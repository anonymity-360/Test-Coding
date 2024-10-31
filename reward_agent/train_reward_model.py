import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from reward_model import RewardModel
from pokerDataset import PokerDataset, collate_fn

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, save_path):
    model.to(device)
    best_loss = np.inf
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Training"):
            context_next_action_vecs, action_history_vecs, rewards = data
            context_next_action_vecs = context_next_action_vecs.to(device)
            action_history_vecs = action_history_vecs.to(device)
            rewards = rewards.to(device)

            optimizer.zero_grad()
            outputs = model(context_next_action_vecs, action_history_vecs)
            loss = criterion(outputs.squeeze(), rewards)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        

        # Evaluation
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Evaluating"):
                context_next_action_vecs, action_history_vecs, rewards = data
                context_next_action_vecs = context_next_action_vecs.to(device)
                action_history_vecs = action_history_vecs.to(device)
                rewards = rewards.to(device)

                outputs = model(context_next_action_vecs, action_history_vecs)
                loss = criterion(outputs.squeeze(), rewards)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_test_loss:.4f}')

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), save_path+'best_model.pt')
            logging.info(f'Epoch {epoch+1}/{num_epochs}, save best_model.pt')

    return train_losses, test_losses
    

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


if __name__ == "__main__":
    # 使用argparse接收命令行参数
    parser = argparse.ArgumentParser(description='Training Reward Model with Different Number of Agents')
    parser.add_argument('--num_players', type=int, default=2,
                        help='Number of Agents')
    parser.add_argument('--folder', type=str, default='./data/nolimit_2player/',
                        help='Directory where the data is stored')
    args = parser.parse_args()
    
    # 根据命令行参数设置folder
    folder = args.folder
    json_path = folder + 'no-limit_holdem_action_results.json'
    save_path = "reward_model/nolimit_" + str(args.num_players) + "player/"
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    # Settings
    num_players = args.num_players
    batch_size = 128
    num_epochs = 150
    d_model = 128
    hidden_dim = 2*d_model
    dropout_rate = 0.4
    lr = 5e-4
    weight_decay = 1e-5
    training_ratio = 0.8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(filename=save_path+'training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    
    # Dataset
    dataset = PokerDataset(json_path)
    train_size = int(training_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = RewardModel(num_players=num_players, d_model=d_model, hidden_dim=hidden_dim, output_dim=1, dropout_rate=dropout_rate)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Training
    train_losses, test_losses = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, save_path)
    loss_figure(train_losses, test_losses, save_path)