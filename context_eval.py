import os
import numpy as np
import argparse
import logging
import json
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from nltk.translate.bleu_score import sentence_bleu


from action_history_GPT_train import EventTokenizer, EVENT_TOKENS
from context_test import GPT_context
from action_history_GPT_eval import calculate_bleu
from utils import STAGE_TOKENS, CARD_TOKENS, SimpleTokenizer



# 写一个函数将json形成Context，顺序需要对应event_history，相当于考虑每个玩家在决策前的视角看到的场上信息作为Context
class EventHistoryContext_LegalActions_Dataset(Dataset):
    def __init__(self, json_files, tokenizer, max_length=24, legal_actions_max_len=10):
        self.event_histories = []
        self.contexts = []  # 保存event对应的context
        self.legal_actions_histories = []  # 保存event对应的legal_actions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.legal_actions_max_len = legal_actions_max_len
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line.strip())
                    event_history, context, legal_actions_history = self.create_event_legal_actions_history(data)
                    event_history.append(tokenizer.eos_token)  # Add end of sequence token
                    legal_actions_history.append([tokenizer.eos_token]) # Add end of sequence token

                    self.event_histories.append(event_history)
                    self.contexts.append(context)  # 保存对应的上下文信息, <eos>已经对应了end_context
                    self.legal_actions_histories.append(legal_actions_history)

    def create_event_legal_actions_history(self, data):
        num_player = len(data["pocket_cards"])
        player_actionIdx = [0] * num_player  # 每个玩家目前的行动索引
        action_history = data["action_history"]
        
        event_history = []
        contexts = []
        legal_actions_history = []

        # 预先定义好各种阶段
        stages = ["PREFLOP", "FLOP", "TURN", "RIVER"]
        stage_actions = {
            "PREFLOP": [],
            "FLOP": "DEAL_THREECARDS",
            "TURN": "DEAL_ONECARD",
            "RIVER": "DEAL_ONECARD"
        }

        current_stage_index = 0
        current_stage = stages[current_stage_index]
        current_agent_state = None
        for action in action_history:
            player_id, action_info = action
            agent_state = data["agent_state"][player_id][player_actionIdx[player_id]]
            # agent_state阶段发生变化，插入对应Dealer行动
            if agent_state["stage"] != current_stage:
                current_stage_index += 1
                current_stage = stages[current_stage_index]
                dealer_action = stage_actions[current_stage]
                if dealer_action:
                    event_history.append(dealer_action)
                    contexts.append(self.extract_context_dealer(data, current_stage, current_agent_state))  # 代理状态的上下文
                    legal_actions_history.append([dealer_action]) # 目前合法的只有dealer_action

            # 添加玩家动作，及其对应的context
            event_history.append(action_info)
            private_cards = data["pocket_cards"][player_id]
            contexts.append(self.extract_context(agent_state, private_cards))  # 从agent_state提取context
            legal_actions_history.append(agent_state["legal_actions"])
            player_actionIdx[player_id] += 1

            # 记录目前的agent_state，便于给dealer加context
            current_agent_state = agent_state
        
        # action_history遍历完以后，根据当前current_stage阶段继续追加dealer的行为
        for i in range(current_stage_index + 1, len(stages)):
            current_stage = stages[i]
            dealer_action = stage_actions[current_stage]
            if dealer_action:
                event_history.append(dealer_action)
                contexts.append(self.extract_context_dealer(data, current_stage, current_agent_state))  # 代理状态的上下文
                legal_actions_history.append([dealer_action])
        
        # 与event_history最后的<eos> token对应的end_contexts
        end_context = {
            "stage": "RIVER",
            "chips_in_pot": current_agent_state["chips_in_pot"],
            "pot": current_agent_state["pot"],
            "public_cards": data["public_cards"],
            "private_cards": self.pad_cards([], 2)
        }
        contexts.append(end_context)

        return event_history, contexts, legal_actions_history


    def pad_cards(self, cards, total_length):
        pad_card = CARD_TOKENS[0]   # 'None'
        if len(cards) < total_length:
            cards += [pad_card] * (total_length - len(cards))
        return cards

    def extract_context(self, agent_state, private_cards):
        context = {
            "stage": agent_state["stage"],
            "chips_in_pot": agent_state["chips_in_pot"],
            "pot": agent_state["pot"],
            "public_cards": self.pad_cards(agent_state["public_cards"], 5),
            "private_cards": self.pad_cards(private_cards, 2)
        }
        return context

    def extract_context_dealer(self, data, current_stage, current_agent_state):
        stage = current_stage
        chips_in_pot = current_agent_state["chips_in_pot"]
        pot = current_agent_state["pot"]
        num_public_cards_stage = {
            "FLOP": 0,
            "TURN": 3,
            "RIVER": 4
        }
        public_cards = data["public_cards"][0:num_public_cards_stage[current_stage]]
        private_cards = []

        dealer_context = {
            "stage": stage,
            "chips_in_pot": chips_in_pot,
            "pot": pot,
            "public_cards": self.pad_cards(public_cards, 5),
            "private_cards": self.pad_cards(private_cards, 2)
        }
        return dealer_context

    def __len__(self):
        return len(self.event_histories)

    def __getitem__(self, idx):
        events = self.event_histories[idx]
        context = self.contexts[idx]
        legal_actions = self.legal_actions_histories[idx]

        # Encode Events
        encoded = self.tokenizer.encode(events)
        len_encoded = len(encoded)

        # Truncate if longer than max_length
        if len_encoded > self.max_length:
            encoded = encoded[:self.max_length]
            context = context[:self.max_length]
            legal_actions = legal_actions[:self.max_length]

        # Pad if shorter than max_length
        if len_encoded < self.max_length:
            padding_length = self.max_length - len_encoded
            encoded = encoded + [self.tokenizer.token_to_id[self.tokenizer.pad_token]] * padding_length
            context = context + [context[-1]] * padding_length # 用最后的end_context来padding，模型会掩码掉loss函数中对<pad>的计算
            legal_actions += [[self.tokenizer.pad_token]] * padding_length

        # Create attention mask (1 for real tokens, 0 for padding tokens)
        attention_mask = [1] * len_encoded
        if len(attention_mask) < self.max_length:
            attention_mask += [0] * (self.max_length - len(attention_mask))
        
        # Pad Legal_actions for Batch Evaluation (Align the length of each element to legal_actions_max_len)
        encoded_legal_actions = []
        for actions in legal_actions:
            encoded_actions = self.tokenizer.encode(actions)
            len_encoded_actions = len(encoded_actions)
            if len_encoded_actions > self.legal_actions_max_len:
                # 截断，避免bug，因此需要将legal_actions_max_len设置的尽可能大
                encoded_actions = encoded_actions[:self.legal_actions_max_len]
            elif len_encoded_actions < self.legal_actions_max_len:
                # 填充
                padding_len = self.legal_actions_max_len - len_encoded_actions
                encoded_actions.extend([self.tokenizer.token_to_id[self.tokenizer.pad_token]] * padding_len)
            encoded_legal_actions.append(encoded_actions)

        input_ids = encoded[:-1]
        labels = encoded[1:]
        attention_mask = attention_mask[1:]  # Attention_mask应该与labels对齐，从而避免计算loss时对<pad>的计算
        context = context[1:]                # context与label对齐，用于辅助input_ids预测labels
        encoded_legal_actions = encoded_legal_actions[1:] # Legal_actions需要与labels对齐，才能用来判断GPT预测的合法性

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "context": context,  # 添加上下文信息
            "encoded_legal_actions": torch.tensor(encoded_legal_actions)
        }



def context_collate_fn(batch):
    # 保持context字典结构，形成堆叠 (batch_size, )
    batch_size = len(batch)
    # 非字典处理
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    encoded_legal_actions = torch.stack([item["encoded_legal_actions"] for item in batch])
    # 字典处理，保持字典列表结构
    context_batch = [item["context"] for item in batch]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "context": context_batch,
        "encoded_legal_actions": torch.tensor(encoded_legal_actions)
    }


def evaluate(model, eval_loader, tokenizer, device):
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    output_sequences = []
    reference_sequences = []
    legal_actions_total = 0
    legal_actions_correct = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluation Batches"):
            input_ids, targets, attention_mask, context = batch["input_ids"], batch["labels"], batch["attention_mask"], batch["context"]
            encoded_legal_actions_list = batch["encoded_legal_actions"]

            input_ids, targets, attention_mask = input_ids.to(device), targets.to(device), attention_mask.to(device)
            logits, loss = model(input_ids, context=context, targets=targets, reduction="none")
            
            # 计算困惑度
            loss = (loss * attention_mask.view(-1)).sum() / attention_mask.sum()
            total_loss += loss.item()
            
            # 预测值与真实标签收集
            preds = torch.argmax(logits, dim=-1)
            preds_masked = torch.masked_select(preds.view(-1), attention_mask.view(-1).bool())
            targets_masked = torch.masked_select(targets.view(-1), attention_mask.view(-1).bool())
            all_preds.extend(preds_masked.cpu().numpy().flatten())
            all_labels.extend(targets_masked.cpu().numpy().flatten())
            
            # 收集用于BLEU计算的预测和参考序列
            output_sequences.append(preds_masked.cpu().numpy().flatten())
            reference_sequences.append(targets_masked.cpu().numpy().flatten())

            # 统计合法动作，计算event预测合法率
            pred_actions = tokenizer.decode(preds_masked.cpu().numpy())
            legal_actions = []
            for actions, mask in zip(encoded_legal_actions_list.cpu(), attention_mask.cpu()):
                # 把padding的legal_actions给筛出来
                for idx, m in enumerate(mask):
                    if m == 1:
                        legal_actions.append(tokenizer.decode(actions[idx].cpu().numpy()))

            for pred, legal in zip(pred_actions, legal_actions):
                legal_actions_total += 1
                if pred in legal:
                    legal_actions_correct += 1
    
    avg_loss = total_loss / len(eval_loader)
    perplexity = np.exp(avg_loss)
    acc = accuracy_score(all_labels, all_preds)
    token_labels = [k for k,v in sorted(tokenizer.token_to_id.items(), key=lambda item: item[1])]
    class_report = classification_report(all_labels, all_preds, labels=list(range(1, len(token_labels))), target_names=token_labels[1:], zero_division=0)
    bleu_score = calculate_bleu(output_sequences, reference_sequences, tokenizer)
    legal_rate = legal_actions_correct / legal_actions_total if legal_actions_total > 0 else None
    
    return perplexity, bleu_score, class_report, acc, legal_rate


if __name__ == "__main__":
    # argparser：指定评测用的模型和GPU
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_player', type=int, default=2, help='num of players')
    parser.add_argument('--model_type', type=str, choices=['gpt-nano', 'gpt-micro', 'gpt-mini', 'gopher-44m', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gopher-44m', help='Model type')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], default='cuda:2', help='Device')
    parser.add_argument('--seq_len', type=int, default=24, help='Maximum length of Action History (Pad + Truncate)')
    args = parser.parse_args()
    
    # 后续可以再生成100万对局数据到test.json，专门用来评测，目前先用这些数据
    batch_size = 4096
    dataset_path = "./reward_agent_dataset"
    dataset_json_idx = [5]
    
    # 加载tokenizer & 数据集设置
    json_files = [f"{dataset_path}/nolimit_{args.num_player}player/hand_results_nolimit_{idx}.json" for idx in dataset_json_idx]
    tokenizer = EventTokenizer(EVENT_TOKENS)
    stage_tokenizer = SimpleTokenizer(STAGE_TOKENS)
    card_tokenizer = SimpleTokenizer(CARD_TOKENS)
    dataset = EventHistoryContext_LegalActions_Dataset(json_files, tokenizer, args.seq_len)
    
    # 用Dataloader，分Batch验证
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=context_collate_fn)
    
    # GPT2-模型设置
    config = GPT_context.get_default_config()
    config.vocab_size = tokenizer.vocab_size
    config.block_size = args.seq_len  # 适当调整block size
    config.model_type = args.model_type  # 使用我们定义的小型GPT配置
    config.num_player = args.num_player  # 添加玩家数量配置
    model = GPT_context(config, stage_tokenizer, card_tokenizer)
    
    # 加载训练好的GPT模型
    save_path = f"./event_history_prediction_context/{args.num_player}player/{args.model_type}/"
    # save_path = f"./event_history_prediction_context_finetune/{args.num_player}player/{args.model_type}/"
    model.load_state_dict(torch.load(os.path.join(save_path, "best_gpt.pt"), map_location=args.device))
    
    # 评估模型
    perplexity, bleu_score, class_report, acc, legal_rate = evaluate(model, test_loader, tokenizer, device=args.device)
    
    # 保存评估日志
    logging.basicConfig(filename=os.path.join(save_path, "evaluate.log"), level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f'Accuracy: {acc}\nPerplexity: {perplexity}\nBLEU Score: {bleu_score}\nLegal Rate of Predicted Events: {legal_rate}\nClassification Report:\n{class_report}')
    
    print(f'Accuracy: {acc}')
    print(f'Perplexity: {perplexity}')
    print(f'BLEU Score: {bleu_score}')
    print(f"Legal Rate of Predicted Events: {legal_rate}")
    print(f'Classification Report:\n{class_report}')