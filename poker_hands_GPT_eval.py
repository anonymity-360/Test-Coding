import torch
from torch.utils.data import random_split
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, classification_report

from poker_hands_GPT_train import PokerHandsDataset, GPTForCardTypeClassification, plot_confusion_matrix
from poker_hands_tokenizer import PokerHandTokenizer, KNOWN_TOKENS, CARDS_CATEGORY
from mingpt.model import GPT


def load_model(model_path, config, num_classes):
    model = GPTForCardTypeClassification(config, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_model(model, data_loader, device='cuda'):
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for token_ids, attention_mask, labels in tqdm(data_loader):
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            cards_logits = model(token_ids)
            predicted = torch.argmax(cards_logits, dim=1)
            y_pred.extend(predicted.view(-1).cpu())
            y_true.extend(labels.view(-1).cpu())
    
    return y_true, y_pred


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "gpt-mini"
    root_path = "./poker_hands_GPT/" + model_type + "/"
    model_path = root_path + "best_gpt.pt"
    # json_files = ["./reward_agent_data/nolimit_2player/hands_category/test.json"]
    json_paths = ["./reward_agent_data/nolimit_2player/hands_category/balanced_poker_hands.json"]
    batch_size = 8192
    max_length = 128
    
    test_ratio = 0.1
    
    # 合并类别处理
    merge_classes = {'FULL_HOUSE': 'HUGE_HANDS', 'FOUR_OF_A_KIND': 'HUGE_HANDS', 'STRAIGHT_FLUSH': 'HUGE_HANDS'}
    unique_classes = sorted(set(CARDS_CATEGORY) - set(merge_classes.keys()) | set(merge_classes.values()))
    num_classes = len(unique_classes)

    # 加载tokenizer和数据集
    tokenizer = PokerHandTokenizer(KNOWN_TOKENS + CARDS_CATEGORY)
    # test_dataset = PokerHandsDataset(json_paths, tokenizer, max_length)
    dataset = PokerHandsDataset(json_paths, tokenizer, max_length, merge_classes=merge_classes, unique_classes=unique_classes)
    
    _, test_dataset = random_split(dataset, [len(dataset)-int(test_ratio*len(dataset)), int(test_ratio*len(dataset))])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型配置
    config = GPT.get_default_config()
    config.vocab_size = tokenizer.vocab_size
    config.block_size = max_length
    config.model_type = model_type

    # 加载模型
    model = load_model(model_path, config, num_classes=num_classes)
    model.to(device)

    # 评估模型
    y_true, y_pred = evaluate_model(model, test_loader, device)

    # 绘制混淆矩阵并计算指标
    plot_confusion_matrix(y_true, y_pred, unique_classes, root_path)
    print(classification_report(y_true, y_pred, target_names=unique_classes))
    
    print("Ended")