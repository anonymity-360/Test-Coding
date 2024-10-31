import os
import json
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import argparse

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from mingpt.model import GPT
from poker_hands_GPT_train import loss_figure


ACTION_TOKENS = ['FOLD', 'CHECK_CALL', 'RAISE_HALF_POT', 'RAISE_POT', 'ALL_IN']
DEALER_TOKENS = ['DEAL_THREECARDS', 'DEAL_ONECARD']
EVENT_TOKENS = ACTION_TOKENS + DEALER_TOKENS


class EventTokenizer:
    def __init__(self, event_tokens):
        self.event_tokens = event_tokens
        
        self.pad_token = '<pad>'
        self.eos_token = '<eos>'
        self.special_tokens = {
            self.pad_token: 0,
            self.eos_token: 1
        }
        
        self.token_to_id = {token: i + len(self.special_tokens) for i, token in enumerate(event_tokens)}
        self.token_to_id.update(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
    
    def encode(self, actions):
        return [self.token_to_id[action] for action in actions]

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return [self.id_to_token[id] for id in ids]


class EventHistoryDataset(Dataset):
    def __init__(self, json_files, tokenizer, max_length=24):
        self.event_histories = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line.strip())
                    event_history = self.create_event_history(data)
                    event_history.append(tokenizer.eos_token)  # Add end of sequence token
                    self.event_histories.append(event_history)

    def create_event_history(self, data):
        num_player = len(data["pocket_cards"])
        player_actionIdx = [0] * num_player # 每个玩家目前的行动索引
        action_history = data["action_history"]

        event_history = []
        
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

            # 添加玩家动作
            event_history.append(action_info)
            player_actionIdx[player_id] += 1
        
        # action_history遍历完以后，根据当前current_stage阶段继续追加dealer的行为
        for i in range(current_stage_index+1, len(stages)):
            dealer_action = stage_actions[stages[i]]
            if dealer_action:
                event_history.append(dealer_action)
        
        return event_history


    def __len__(self):
        return len(self.event_histories)
    
    def __getitem__(self, idx):
        events = self.event_histories[idx]
        encoded = self.tokenizer.encode(events)
        len_encoded = len(encoded)
        
        # Truncate if longer than max_length
        if len_encoded > self.max_length:
            encoded = encoded[:self.max_length]

        # Pad if shorter than max_length
        if len_encoded < self.max_length:
            padding_length = self.max_length - len_encoded
            encoded = encoded + [self.tokenizer.token_to_id[self.tokenizer.pad_token]] * padding_length

        # Create attention mask (1 for real tokens, 0 for padding tokens)
        attention_mask = [1] * len_encoded
        if len(attention_mask) < self.max_length:
            attention_mask += [0] * (self.max_length - len(attention_mask))

        input_ids = encoded[:-1]
        labels = encoded[1:]
        attention_mask = attention_mask[1:] # Attention_mask应该与labels对齐，从而避免计算loss时对<pad>的计算

        return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attention_mask)



def train(model, train_loader, eval_loader, optimizer, device, epochs=5, save_path="./"):
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    eval_losses = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_train_loss = 0

        for input_ids, targets, attention_mask in tqdm(train_loader, desc="Training Batches", leave=False):
            input_ids, targets, attention_mask = input_ids.to(device), targets.to(device), attention_mask.to(device)
            optimizer.zero_grad()
            logits, loss = model(input_ids, targets=targets, reduction="none")
            
            # 忽略<pad>的预测loss
            loss = (loss * attention_mask.view(-1)).sum() / attention_mask.sum()
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for input_ids, targets, attention_mask in tqdm(eval_loader, desc="Validation Batches", leave=False):
                input_ids, targets, attention_mask = input_ids.to(device), targets.to(device), attention_mask.to(device)
                logits, loss = model(input_ids, targets=targets, reduction="none")
                
                # 忽略<pad>的预测loss
                loss = (loss * attention_mask.view(-1)).sum() / attention_mask.sum()
                
                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(targets.cpu().numpy().flatten())

        avg_val_loss = total_val_loss / len(eval_loader)
        eval_losses.append(avg_val_loss)
        logging.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path + "best_gpt.pt")
            logging.info("Saved new best model based on validation loss.")

        tqdm.write(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    return train_losses, eval_losses



@torch.no_grad()
def generate(model, tokenizer, input_sequence, max_new_tokens, device="cpu", temperature=1.0, do_sample=False, top_k=None):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(input_sequence)).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        input_ids_cond = input_ids if input_ids.size(1) <= model.block_size else input_ids[:, -model.block_size:]
        logits, _ = model(input_ids_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)

        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            _, next_token = torch.topk(probs, k=1, dim=-1)

        if next_token.item() == tokenizer.token_to_id[tokenizer.eos_token] or next_token.item() == tokenizer.token_to_id[tokenizer.pad_token]:
            break

        input_ids = torch.cat((input_ids, next_token), dim=1)

    generated_sequence = tokenizer.decode(input_ids.squeeze().tolist())
    return generated_sequence



if __name__ == "__main__":
    # argparser
    parser = argparse.ArgumentParser()
        # Required
    parser.add_argument('--num_player', type=int, default=6, help='num of players')
    parser.add_argument('--epoch', type=int, default=50, help='num of epochs')
    parser.add_argument('--model_type', type=str, choices=['gpt-nano', 'gpt-micro', 'gpt-mini', 'gopher-44m', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default="gpt-mini", help='Model type')
    parser.add_argument('--device', type=str, choices=['cuda:0', 'cuda:1', 'cuda:2'], default="cuda:0", help='GPU Device')
        # Default
    parser.add_argument('--seq_len', type=int, default=48, help='Maximum length of Action History (Pad + Truncate)')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    args = parser.parse_args()

    # Settings
    num_player = args.num_player
    dataset_path = "./reward_agent_dataset"
    # dataset_json_idx = [1,2,3,4]
    dataset_json_idx = [1]
    seq_len = args.seq_len
    batch_size = args.batch_size
    train_ratio = 0.9
    num_epochs = args.epoch
    model_type = args.model_type
    save_path = "./event_history_prediction/{}player/{}/".format(num_player, model_type)
    device = args.device
    
    # 生成json_files列表
    json_files = [f"{dataset_path}/nolimit_{num_player}player/hand_results_nolimit_{idx}.json" for idx in dataset_json_idx]
    
    # 初始化tokenizer和数据集
    tokenizer = EventTokenizer(EVENT_TOKENS)
    dataset = EventHistoryDataset(json_files, tokenizer, seq_len)

    # 训练/测试集分割
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # GPT2-模型设置
    config = GPT.get_default_config()
    config.vocab_size = tokenizer.vocab_size
    config.block_size = seq_len  # 适当调整block size
    config.model_type = model_type  # 使用我们定义的小型GPT配置
    model = GPT(config)
    
    # 优化器
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    # 训练
    if not os.path.exists(save_path): os.makedirs(save_path)
    logging.basicConfig(filename=save_path+"training.log", level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    train_losses, eval_losses = train(model, train_loader, val_loader, optimizer, device=device, epochs=num_epochs, save_path=save_path)
    loss_figure(train_losses, eval_losses, save_path)
    
    # 加载已保存模型并生成序列示例
    input_sequence = ["CHECK_CALL"]
    generated_sequence = generate(model, tokenizer, input_sequence, max_new_tokens=10, device=device)
    print("Generated Sequence:", generated_sequence)
    
    print("ENding")