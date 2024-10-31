import logging
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import random_split

from poker_hands_GPT_train import PokerHandsDataset, GPTForCardTypeClassification, calculate_class_weights, train
from mingpt.model import GPT
from poker_hands_tokenizer import PokerHandTokenizer, KNOWN_TOKENS, CARDS_CATEGORY
from poker_hands_GPT_train import loss_figure


if __name__ == "__main__":
    json_paths = ["./reward_agent_data/nolimit_2player/hands_category/balanced_poker_hands.json"]
    batch_size = 256
    max_length = 128
    train_ratio = 0.85
    num_epochs = 10
    
    continue_training_path = "./poker_hands_GPT/gpt-mini/"
    start_epoch = 40
    
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
    config.model_type = "gpt-mini"  # 使用我们定义的小型GPT配置
    
    model_path = continue_training_path+"best_gpt.pt"
    model = GPTForCardTypeClassification(config, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    
    # 配置优化器
    optimizer = Adam(model.parameters(), lr=1e-6, weight_decay=1e-5)
    
    # 计算类权重
    class_weights = calculate_class_weights(dataset).to('cuda')
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 配置日志记录器以追加模式续写日志
    logging.basicConfig(filename=continue_training_path + "training.log", level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', filemode='a')
    
    # 继续训练
    train_losses, eval_losses = train(model, train_loader, val_loader, optimizer, criterion, device="cuda", epochs=num_epochs, save_path=continue_training_path, unique_classes=unique_classes, start_epoch=start_epoch)
    loss_figure(train_losses, eval_losses, continue_training_path)