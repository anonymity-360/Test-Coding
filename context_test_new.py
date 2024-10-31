import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import logging
from tqdm import tqdm
import os
import argparse
from torch.optim import Adam

from mingpt.model import GPT, NewGELU
from utils import STAGE_TOKENS, CARD_TOKENS, SimpleTokenizer, loss_figure
from action_history_GPT_train import EventTokenizer, EVENT_TOKENS


stage_tokenizer= SimpleTokenizer(STAGE_TOKENS)
card_tokenizer = SimpleTokenizer(CARD_TOKENS)   # 包含None作Padding，用于public_cards填充


# 继承用于event history prediction的GPT模型，并扩展到能够使用Context信息
class GPT_context(GPT):
    # GPT 特征shape为 (batch, seq_len, n_embd)
    def __init__(self, config, stage_tokenizer, card_tokenizer):
        # config：加入num_player
        super().__init__(config)
        self.stage_tokenizer = stage_tokenizer
        self.card_tokenizer = card_tokenizer

        self.dropout_rate = 0.1
        
        # 定义Context的嵌入层
        self.emb_stage = nn.Embedding(stage_tokenizer.vocab_size, config.n_embd)
        self.emb_chips_in_pot = nn.Linear(config.num_player, config.n_embd)    # chips_in_pot=[2, 1, 0] -> pot=3
        self.emb_pot = nn.Linear(1, config.n_embd)
        self.emb_card = nn.Embedding(card_tokenizer.vocab_size, config.n_embd) # 52张牌 + None用作padding

        # 定义Context的嵌入聚合层
        self.aggregation_layer = nn.Sequential(
            nn.Linear(10*config.n_embd, 4*config.n_embd),
            NewGELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(4*config.n_embd, 2*config.n_embd),
            NewGELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(2*config.n_embd, config.n_embd)
        )

        # 残差连接层
        self.res_layer = nn.Linear(10*config.n_embd, config.n_embd)

    def encode_context(self, context):
        # context [batch, seq_len]
        batch_size, seq_len = len(context), len(context[0])

        # Stage Embedding
        stage_ids = [[self.stage_tokenizer.encode([c["stage"]])[0] for c in seq] for seq in context]
        stage_ids = torch.LongTensor(stage_ids).to(self.emb_stage.weight.device)
        stage_emb = self.emb_stage(stage_ids)

        # Chips_in_pot & Pot Embedding
        chips_in_pot_list = [[c["chips_in_pot"] for c in seq] for seq in context]
        chips_in_pot_tensor = torch.FloatTensor(chips_in_pot_list).to(self.emb_chips_in_pot.weight.device)
        chips_emb = self.emb_chips_in_pot(chips_in_pot_tensor)

        pot_list = [[c["pot"] for c in seq] for seq in context]
        pot_tensor = torch.FloatTensor(pot_list).unsqueeze(-1).to(self.emb_pot.weight.device)
        pot_emb = self.emb_pot(pot_tensor)
        

        # Public Card Embedding （传入之前，public_cards需要用None给Padding到5张）
        public_cards_list = [[self.card_tokenizer.encode(c["public_cards"]) for c in seq] for seq in context]
        public_cards_ids_tensor = torch.LongTensor(public_cards_list).to(self.emb_card.weight.device)  # (batch_size, seq_len-1, 5)
        public_cards_emb = self.emb_card(public_cards_ids_tensor).view(batch_size, seq_len, -1)  # (batch_size, seq_len-1, 5*n_embd)


        # Private Card Embedding（对于Dealer而言，需要用None给Padding到2张）
        private_cards_list = [[self.card_tokenizer.encode(c["private_cards"]) for c in seq] for seq in context]
        private_cards_ids_tensor = torch.LongTensor(private_cards_list).to(self.emb_card.weight.device)  # (batch_size, seq_len-1, 2)
        private_cards_emb = self.emb_card(private_cards_ids_tensor).view(batch_size, seq_len, -1)  # (batch_size, seq_len-1, 2*n_embd)


        # Concatenate （10*config.n_embd）
        context_emb = torch.cat([stage_emb, chips_emb, pot_emb, public_cards_emb, private_cards_emb], dim=-1)

        # Context Embedding
        context_emb = self.aggregation_layer(context_emb) + self.res_layer(context_emb)
        return context_emb
    
    def forward(self, idx, context=None, targets=None, reduction='mean'):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Logits计算前，若有Context，则融入Context相关的Embedding
        if context is not None:
            context_embs = self.encode_context(context) # (b, t, n_embd)
            x = x + context_embs
        
        logits = self.lm_head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)

        return logits, loss
            



# 写一个函数将json形成Context，顺序需要对应event_history，相当于考虑每个玩家在决策前的视角看到的场上信息作为Context
class EventHistoryContextDataset(Dataset):
    def __init__(self, json_files, tokenizer, max_length=24):
        self.event_histories = []
        self.contexts = []  # 保存event对应的context
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line.strip())
                    event_history, context = self.create_event_history(data)
                    event_history.append(tokenizer.eos_token)  # Add end of sequence token
                    self.event_histories.append(event_history)
                    self.contexts.append(context)  # 保存对应的上下文信息, <eos>已经对应了end_context

    def create_event_history(self, data):
        num_player = len(data["pocket_cards"])
        player_actionIdx = [0] * num_player  # 每个玩家目前的行动索引
        action_history = data["action_history"]
        event_history = []
        contexts = []

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

            # 添加玩家动作，及其对应的context
            event_history.append(action_info)
            private_cards = data["pocket_cards"][player_id]
            contexts.append(self.extract_context(agent_state, private_cards))  # 从agent_state提取context
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

        # 与event_history最后的<eos> token对应的end_contexts
        end_context = {
            "stage": "RIVER",
            "chips_in_pot": current_agent_state["chips_in_pot"],
            "pot": current_agent_state["pot"],
            "public_cards": data["public_cards"],
            "private_cards": self.pad_cards([], 2)
        }
        contexts.append(end_context)

        return event_history, contexts

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
        encoded = self.tokenizer.encode(events)
        len_encoded = len(encoded)

        # Truncate if longer than max_length
        if len_encoded > self.max_length:
            encoded = encoded[:self.max_length]
            context = context[:self.max_length]

        # Pad if shorter than max_length
        if len_encoded < self.max_length:
            padding_length = self.max_length - len_encoded
            encoded = encoded + [self.tokenizer.token_to_id[self.tokenizer.pad_token]] * padding_length
            context = context + [context[-1]] * padding_length # 用最后的end_context来padding，模型会掩码掉loss函数中对<pad>的计算

        # Create attention mask (1 for real tokens, 0 for padding tokens)
        attention_mask = [1] * len_encoded
        if len(attention_mask) < self.max_length:
            attention_mask += [0] * (self.max_length - len(attention_mask))

        input_ids = encoded[:-1]
        labels = encoded[1:]
        attention_mask = attention_mask[1:]  # Attention_mask应该与labels对齐，从而避免计算loss时对<pad>的计算
        context = context[1:]                # context与label对齐，用于辅助input_ids预测labels

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "context": context  # 添加上下文信息
        }


def context_collate_fn(batch):
    # 保持context字典结构，形成堆叠 (batch_size, )
    batch_size = len(batch)

    # 非字典处理
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # 字典处理，保持字典列表结构
    context_batch = [item["context"] for item in batch]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "context": context_batch
    }



# 1. 直接context和event联合重新训模型；
def train(model, train_loader, eval_loader, optimizer, device, epochs=5, save_path="./"):
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    eval_losses = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc="Training Batches", leave=False):
            input_ids, targets, attention_mask, context = batch["input_ids"], batch["labels"], batch["attention_mask"], batch["context"]
            input_ids, targets, attention_mask = input_ids.to(device), targets.to(device), attention_mask.to(device)
            optimizer.zero_grad()
            logits, loss = model(input_ids, context=context, targets=targets, reduction="none")
            
            # 忽略<pad>的预测loss
            loss = (loss * attention_mask.view(-1)).sum() / attention_mask.sum()
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # evaluate
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Validation Batches", leave=False):
                input_ids, targets, attention_mask, context = batch["input_ids"], batch["labels"], batch["attention_mask"], batch["context"]
                input_ids, targets, attention_mask = input_ids.to(device), targets.to(device), attention_mask.to(device)
                logits, loss = model(input_ids, context=context, targets=targets, reduction="none")
                
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


# 2. 加载event history预训练模型并固定，只训练context embedding相关的层，看看精度能否进一步增长
def load_pretrained_GPT(model, pretained_model_path):
    pretrained_state_dict = torch.load(pretained_model_path, map_location="cpu")
    model.load_state_dict(pretrained_state_dict, strict=False)

def freeze_eventGPT_params(model):
    for name, param in model.named_parameters():
        # 只开放GPT_context特有的网络参数
        if not name.startswith(("emb_", "aggregation_layer", "res_layer")):
            param.requires_grad = False


if __name__ == "__main__":
    # argparser
    parser = argparse.ArgumentParser()
    # Required
    parser.add_argument('--num_player', type=int, default=2, help='num of players')
    parser.add_argument('--epoch', type=int, default=50, help='num of epochs')
    parser.add_argument('--model_type', type=str, choices=['gpt-nano', 'gpt-micro', 'gpt-mini', 'gopher-44m', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default="gpt-nano", help='Model type')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], default="cpu", help='Device')
    parser.add_argument('--is_pretrained', type=bool, default=False, help='Whether the pure event history GPT pretrained model should be loaded')
    # Default
    parser.add_argument('--seq_len', type=int, default=48, help='Maximum length of Action History (Pad + Truncate)')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    args = parser.parse_args()
    
    # Settings
    num_player = args.num_player
    dataset_path = "./reward_agent_dataset"
    dataset_json_idx = [1]
    seq_len = args.seq_len
    batch_size = args.batch_size
    train_ratio = 0.9
    num_epochs = args.epoch
    model_type = args.model_type
    device = args.device
    
    # 生成json_files列表
    json_files = [f"{dataset_path}/nolimit_{num_player}player/hand_results_nolimit_{idx}.json" for idx in dataset_json_idx]
    
    # 初始化tokenizer和数据集
    tokenizer = EventTokenizer(EVENT_TOKENS)
    dataset = EventHistoryContextDataset(json_files, tokenizer, max_length=seq_len)

    # GPT-模型设置
    config = GPT_context.get_default_config()
    config.vocab_size = tokenizer.vocab_size
    config.block_size = seq_len  # 适当调整block size
    config.model_type = model_type  # 使用我们定义的小型GPT配置
    config.num_player = num_player  # 添加玩家数量配置
    model = GPT_context(config, stage_tokenizer, card_tokenizer)

    # 两种训练方式
    if args.is_pretrained == True:
        # 加载预训练event history预测模型进行fine_tuning
        save_path = f"./event_history_prediction_context_finetune/{num_player}player/{model_type}/"
        pretrained_model_path = f"./event_history_prediction/{num_player}player/{model_type}/best_gpt.pt"
        # 加载模型、冻结预训练参数
        load_pretrained_GPT(model, pretrained_model_path)
        freeze_eventGPT_params(model)
    else:
        # 重新训练 context event history预测模型
        save_path = f"./event_history_prediction_context/{num_player}player/{model_type}/"

    # 创建保存路径
    if not os.path.exists(save_path): os.makedirs(save_path)
    logging.basicConfig(filename=os.path.join(save_path, "training.log"), level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    
    # 训练/测试集分割
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=context_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=context_collate_fn)
    
    # 优化器
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    # 训练
    train_losses, eval_losses = train(model, train_loader, val_loader, optimizer, device=device, epochs=num_epochs, save_path=save_path)
    loss_figure(train_losses, eval_losses, save_path)