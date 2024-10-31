import numpy as np
import argparse
import logging
import json

import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, accuracy_score
from nltk.translate.bleu_score import sentence_bleu

from action_history_GPT_train import EventTokenizer, EVENT_TOKENS, EventHistoryDataset
from mingpt.model import GPT


class EventHistory_LegalActions_Dataset(Dataset):
    def __init__(self, json_files, tokenizer, max_length=24, legal_actions_max_len=10):
        self.event_histories = []
        self.legal_actions_histories = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.legal_actions_max_len = legal_actions_max_len
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line.strip())
                    event_history, legal_actions_history = self.create_event_legal_actions_history(data)
                    event_history.append(tokenizer.eos_token)  # Add end of sequence token
                    legal_actions_history.append([tokenizer.eos_token])
                    self.event_histories.append(event_history)
                    self.legal_actions_histories.append(legal_actions_history)
    
    def create_event_legal_actions_history(self, data):
        num_player = len(data["pocket_cards"])
        player_actionIdx = [0] * num_player # 每个玩家目前的行动索引
        action_history = data["action_history"]

        event_history = []
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
        for action in action_history:
            player_id, action_info = action
            agent_state = data["agent_state"][player_id][player_actionIdx[player_id]]

            # 阶段发生变化，插入对应Dealer行动
            if agent_state["stage"] != current_stage:
                current_stage_index += 1
                current_stage = stages[current_stage_index]
                dealer_action = stage_actions[current_stage]
                if dealer_action:
                    event_history.append(dealer_action)
                    legal_actions_history.append([dealer_action]) # 目前合法的只有dealer_action

            # 添加玩家动作
            event_history.append(action_info)
            legal_actions_history.append(agent_state["legal_actions"])
            player_actionIdx[player_id] += 1
        
        # action_history遍历完以后，根据当前current_stage阶段继续追加dealer的行为
        for i in range(current_stage_index+1, len(stages)):
            dealer_action = stage_actions[stages[i]]
            if dealer_action:
                event_history.append(dealer_action)
                legal_actions_history.append([dealer_action])
        
        return event_history, legal_actions_history

    def __len__(self):
        return len(self.event_histories)
    
    def __getitem__(self, idx):
        # 返回(input_ids, labels, attention_mask, legal_actions)，legal_actions每个元素是一个list
        events = self.event_histories[idx]
        legal_actions = self.legal_actions_histories[idx]
        
        # Encode Events
        encoded = self.tokenizer.encode(events)
        len_encoded = len(encoded)
        
        # Truncate if longer than max_length
        if len_encoded > self.max_length:
            encoded = encoded[:self.max_length]
            legal_actions = legal_actions[:self.max_length]

        # Pad if shorter than max_length
        if len_encoded < self.max_length:
            padding_length = self.max_length - len_encoded
            encoded = encoded + [self.tokenizer.token_to_id[self.tokenizer.pad_token]] * padding_length
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
        # legal_actions = legal_actions[1:]   # Legal_actions需要与labels对齐，才能用来判断GPT预测的合法性
        encoded_legal_actions = encoded_legal_actions[1:]
        
        return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attention_mask), torch.tensor(encoded_legal_actions)



def calculate_bleu(output_sequences, reference_sequences, tokenizer):
    bleu_scores = []
    for output_seq, reference_seq in zip(output_sequences, reference_sequences):
        # 分词以准备计算BLEU
        output_tokens = tokenizer.decode(output_seq)
        reference_tokens = [tokenizer.decode(reference_seq)]
        bleu_score = sentence_bleu(reference_tokens, output_tokens)
        bleu_scores.append(bleu_score)
    return np.mean(bleu_scores)


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
        for input_ids, targets, attention_mask, encoded_legal_actions_list in eval_loader:
            input_ids, targets, attention_mask = input_ids.to(device), targets.to(device), attention_mask.to(device)
            logits, loss = model(input_ids, targets=targets, reduction="none")
            
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

    # 计算acc，相当于next-token-prediction的任务准确率
    acc = accuracy_score(all_labels, all_preds)

    token_labels = [k for k,v in sorted(tokenizer.token_to_id.items(), key=lambda item: item[1])]
    # Classification排除 <pad>:0
    class_report = classification_report(all_labels, all_preds, labels=list(range(1, len(token_labels))), target_names=token_labels[1:], zero_division=0)
    bleu_score = calculate_bleu(output_sequences, reference_sequences, tokenizer)
    legal_rate = legal_actions_correct / legal_actions_total if legal_actions_total > 0 else None
    
    return perplexity, bleu_score, class_report, acc, legal_rate



if __name__ == "__main__":
    # argparser：指定评测用的模型和GPU
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_player', type=int, default=6, help='num of players')
    parser.add_argument('--model_type', type=str, choices=['gpt-nano', 'gpt-micro', 'gpt-mini', 'gopher-44m', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gopher-44m', help='Model type')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], default='cuda:2', help='Device')
    parser.add_argument('--seq_len', type=int, default=48, help='Maximum length of Action History (Pad + Truncate)')
    args = parser.parse_args()

    # 后续可以再生成100万对局数据到test.json，专门用来评测，目前先用这些数据
    batch_size = 4096
    dataset_path = "./reward_agent_dataset"
    dataset_json_idx = [5]
    test_ratio = 1
    
    # 加载tokenizer & 数据集设置
    json_files = [f"{dataset_path}/nolimit_{args.num_player}player/hand_results_nolimit_{idx}.json" for idx in dataset_json_idx]
    tokenizer = EventTokenizer(EVENT_TOKENS)
    dataset = EventHistory_LegalActions_Dataset(json_files, tokenizer, args.seq_len)
    
    # 用Dataloader，分Batch验证
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # GPT2-模型设置
    config = GPT.get_default_config()
    config.vocab_size = tokenizer.vocab_size
    config.block_size = args.seq_len  # 适当调整block size
    config.model_type = args.model_type  # 使用我们定义的小型GPT配置
    model = GPT(config)

    # 加载训练好的GPT模型
    save_path = "./event_history_prediction/{}player/{}/".format(args.num_player, args.model_type)
    model.load_state_dict(torch.load(save_path + "best_gpt.pt"))
    
    # 评估模型
    perplexity, bleu_score, class_report, acc, legal_rate = evaluate(model, test_loader, tokenizer, device=args.device)


    # 保存评估日志
    logging.basicConfig(filename=save_path+"evaluate.log", level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f'Accuracy: {acc}\nPerplexity: {perplexity}\nBLEU Score: {bleu_score}\nLegal Rate of Predicted Events: {legal_rate}\nClassification Report:\n{class_report}')
    
    print(f'Accuracy: {acc}')
    print(f'Perplexity: {perplexity}')
    print(f'BLEU Score: {bleu_score}')
    print(f"Legal Rate of Predicted Events: {legal_rate}")
    print(f'Classification Report:\n{class_report}')