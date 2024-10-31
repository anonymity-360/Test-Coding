from torch.utils.data import Dataset
import torch
import torch.nn as nn

import json
from reward_model import STAGE, ACTION, CARD

def board2vec(board):
    """ board to vector, nolimit Hold'em max to 5 board cards
    [] -> [0, 0, 0, 0, 0]
    ["SA", "S2", "S3"] -> [1, 2, 3, 0, 0]
    """
    board_card = [0] * 5
    n = len(board)
    if n > 0:
        for i in range(n):
            board_card[i] = CARD[board[i]]
    
    return board_card


def legalAction2vec(legalAction):
    """ legalAction to vector, nolimit Hold'em max to 5 action types
    ["FOLD", "CHECK_CALL", "RAISE_HALF_POT", "RAISE_POT", "ALL_IN"] -> [1, 2, 3, 4, 5]
    ["FOLD", "CHECK_CALL", "RAISE_POT", "ALL_IN"] -> [1, 2, 0, 4, 5]
    """
    legalAction_vec = [0] * 5
    for legal_action in legalAction:
        legalAction_vec[ACTION[legal_action]-1] = ACTION[legal_action]
    return legalAction_vec


def actHistory2vec(action_history):
    n = len(action_history)
    if n == 0:
        # "action_history": []
        # 返回一个占位符，表示没有历史行动
        return [[0, 0]]
    else:
        # "action_history": [[1, "CHECK_CALL"], [0, "RAISE_HALF_POT"]]
        # 转化为[seq_len=n, dim=2]
        action_history_vec = []
        for i in range(n):
            id, action = action_history[i]
            action_history_vec.append([id+1, ACTION[action]]) # id从1开始
        return action_history_vec

class PokerDataset(Dataset):
    def __init__(self, filepath):
        self.data = [json.loads(line) for line in open(filepath, 'r')]
        self.STAGE = STAGE
        self.ACTION = ACTION
        self.CARD = CARD


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        action_history = item["action_history"]
        next_action = item["next_action"]
        payoff_reward = item["payoff_reward"]
        
        # context to vector
        position = context["position"] + 1  # 编号从1开始，有利于action_history用0来占位
        stage = self.STAGE[context["stage"]]
        pocket_cards = [CARD[card] for card in context["pocket_cards"]]
        board = board2vec(context["board"])
        chips_in_pot = context["chips_in_pot"]
        pot = context["pot"]
        legal_actions = legalAction2vec(context["legal_actions"])
        
        context_vec = [position, stage] + pocket_cards + board + legal_actions + chips_in_pot + [pot] 
        
        # next_action to vector
        next_action_vec = [self.ACTION[next_action]]
        
        # action_history to vector
        action_history_vec = actHistory2vec(action_history)
        
        return context_vec, next_action_vec, action_history_vec, payoff_reward


def collate_fn(batch):
    context_vecs, next_action_vecs, action_histories, rewards = zip(*batch)
    context_next_action_vecs = torch.tensor([cv + nv for cv, nv in zip(context_vecs, next_action_vecs)], dtype=torch.float32)
    action_history_vecs = nn.utils.rnn.pad_sequence([torch.tensor(ah) for ah in action_histories], batch_first=True, padding_value=0)
    return context_next_action_vecs, action_history_vecs, torch.tensor(rewards).float()