import os
import json
from tqdm import tqdm
import logging
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import random_split

from mingpt.model import GPT
from poker_hands_tokenizer import format_poker_hand_sample, PokerHandTokenizer, KNOWN_TOKENS, CARDS_CATEGORY


class PokerHandsDataset(Dataset):
    def __init__(self, json_files, tokenizer, max_length=32, merge_classes=None, unique_classes=None):
        self.samples = []
        self.tokenizer = tokenizer
        self.merge_classes = merge_classes if merge_classes else {}
        self.unique_classes = unique_classes if unique_classes else set(CARDS_CATEGORY)

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line.strip())
                    formatted_text = format_poker_hand_sample(data)
                    token_ids, attention_mask = self.tokenizer.encode(formatted_text, max_length=max_length)
                    
                    # 处理合并类别
                    category = data['cards_category']
                    if category in self.merge_classes:
                        category = self.merge_classes[category]
                        
                    self.samples.append((token_ids, attention_mask, category))
        
        # 对类别进行编码
        self.label_encodings = {label: idx for idx, label in enumerate(self.unique_classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids, attention_mask, category = self.samples[idx]
        category_id = self.label_encodings[category]
        return torch.tensor(token_ids), torch.tensor(attention_mask), category_id


class GPTForCardTypeClassification(GPT):
    def __init__(self, config, num_classes=None):
        super().__init__(config)
        self.num_classes = num_classes if num_classes is not None else len(CARDS_CATEGORY)
        # self.card_type_head = nn.Linear(config.n_embd, len(CARDS_CATEGORY), bias=False)
        self.card_type_head = nn.Linear(config.n_embd, self.num_classes)

    def forward(self, idx):
        # 确保序列长度不超过块大小
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        # 生成位置编码
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)

        # 提取token和位置嵌入，然后相加
        tok_emb = self.transformer.wte(idx)  # token嵌入
        pos_emb = self.transformer.wpe(pos)  # 位置嵌入
        x = self.transformer.drop(tok_emb + pos_emb)  # 应用dropout

        # 通过所有的transformer块
        for block in self.transformer.h:
            x = block(x)

        # 应用最后的层规范化
        x = self.transformer.ln_f(x)

        # 获取序列最后一个时间步的输出
        x = x[:, -1, :]  # 只关心序列的最后一个输出，用于分类

        # 通过牌型分类的线性层
        card_logits = self.card_type_head(x)

        return card_logits



def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    # cm = confusion_matrix(y_true, y_pred, normalize="all")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True Label")
    ax.set_title("confusion matrix")
    plt.tight_layout()
    plt.savefig(save_path+"confusion.png", format="png")


def loss_figure(train_losses, test_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Evaluating Loss')
    plt.title('Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path + "loss_plot.png", dpi=400)
    print("Image saved")


def calculate_class_weights(dataset):
    # 统计每个类别的样本数量
    label_counts = Counter()
    for _, _, category_id in dataset:
        label_counts[category_id] += 1
    
    # 计算总样本数并确保为浮点数以避免整数除法
    total_count = float(sum(label_counts.values()))
    
    # 计算每个类的权重：总样本数除以该类样本数
    class_weights = {cls: total_count / count for cls, count in label_counts.items() if count > 0}

    # 根据label_encodings的大小初始化权重张量
    weights_tensor = torch.zeros(len(dataset.label_encodings), dtype=torch.float32)
    
    # 填充权重张量，按照dataset的label_encodings的顺序
    for label, idx in dataset.label_encodings.items():
        if idx in class_weights:
            weights_tensor[idx] = class_weights[idx]

    return weights_tensor




def train(model, train_loader, eval_loader, optimizer, criterion, device, epochs=5, save_path="./", unique_classes=None, start_epoch=0):
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    eval_losses = []
    
    # These are the unique class labels adjusted for merged classes
    target_names = unique_classes
    labels_indices = {label: idx for idx, label in enumerate(target_names)}


    for epoch in tqdm(range(start_epoch, start_epoch+epochs), desc="Training Epochs"):
        # Train
        model.train()
        total_train_loss = 0
        for token_ids, attention_mask, labels in tqdm(train_loader, desc="Training Batches", leave=False):
            token_ids, labels = token_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            cards_logits = model(token_ids)
            loss = criterion(cards_logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Eval
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for token_ids, attention_mask, labels in tqdm(eval_loader, desc="Validation Batches", leave=False):
                token_ids, labels = token_ids.to(device), labels.to(device)
                cards_logits = model(token_ids)
                loss = criterion(cards_logits, labels)
                total_val_loss += loss.item()
                preds = torch.argmax(cards_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(eval_loader)
        eval_losses.append(avg_val_loss)
        logging.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_val_loss:.4f}')
        
        
        # log classification report
        if (epoch+1)%5 == 0:
            report = classification_report(all_labels, all_preds, target_names=target_names, labels=list(labels_indices.values()), output_dict=True, zero_division=0)
            logging.info(f'Epoch {epoch+1}/{epochs}, Classification Report:\n{report}')

        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path + "best_gpt.pt")
            logging.info("Saved new best model based on validation loss.")

        tqdm.write(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
    
    plot_confusion_matrix(all_labels, all_preds, target_names, save_path)
    
    return train_losses, eval_losses


def generate_json_paths(player_list, idx_list):
    json_paths = []
    base_path = "./reward_agent_data/nolimit_{}player/hands_category/poker_hands_samples_{}.json"
    
    for num_player in player_list:
        for idx in idx_list:
            path = base_path.format(num_player, idx)
            json_paths.append(path)
    
    return json_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training GPT Model')
    parser.add_argument('--model_type', type=str, default="gpt-micro",
                        help='[gpt-nano, gpt-micro, gpt-mini]')
    args = parser.parse_args()
    
    # 参数
    # player_list = [2]
    # idx_list = [1]
    # json_paths = generate_json_paths(player_list, idx_list)
    # json_paths = ["./reward_agent_data/nolimit_2player/hands_category/test.json"]
    json_paths = ["./reward_agent_data/nolimit_2player/hands_category/balanced_poker_hands.json"]
    batch_size = 256
    max_length = 128
    train_ratio = 0.85
    num_epochs = 50

    # 合并类别处理
    merge_classes = {'FULL_HOUSE': 'HUGE_HANDS', 'FOUR_OF_A_KIND': 'HUGE_HANDS', 'STRAIGHT_FLUSH': 'HUGE_HANDS'}
    unique_classes = sorted(set(CARDS_CATEGORY) - set(merge_classes.keys()) | set(merge_classes.values()))
    num_classes = len(unique_classes)

    # 初始化tokenizer和数据集
    tokenizer = PokerHandTokenizer(KNOWN_TOKENS + CARDS_CATEGORY)
    dataset = PokerHandsDataset(json_paths, tokenizer, max_length, merge_classes=merge_classes, unique_classes=unique_classes)
    
    # 训练/测试集分割
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # GPT2-模型设置
    config = GPT.get_default_config()
    config.vocab_size = tokenizer.vocab_size
    config.block_size = max_length  # 适当调整block size
    config.model_type = args.model_type  # 使用我们定义的小型GPT配置
    
    model = GPTForCardTypeClassification(config, num_classes=num_classes)

    # 配置优化器
    optimizer = Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
    
    # 计算类权重
    class_weights = calculate_class_weights(dataset).to('cuda')
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    
    # 训练
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = "./poker_hands_GPT/" + config.model_type + "/" + date_time + "/"
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    logging.basicConfig(filename=save_path+"training.log", level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s') # 设置日志记录器
    train_losses, eval_losses = train(model, train_loader, val_loader, optimizer, criterion, device="cuda", epochs=num_epochs, save_path=save_path, unique_classes=unique_classes)
    loss_figure(train_losses, eval_losses, save_path)