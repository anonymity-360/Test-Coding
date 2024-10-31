import torch
import torch.nn as nn

STAGE = {
    "PREFLOP": 0,
    "FLOP": 1,
    "TURN": 2,
    "RIVER": 3,
    "END_HIDDEN": 4,
    "SHOWDOWN": 5
}

ACTION = {
    "NONE": 0,
    "FOLD": 1,
    "CHECK_CALL": 2,
    "RAISE_HALF_POT": 3,
    "RAISE_POT": 4,
    "ALL_IN": 5
}

CARD = {"NONE": 0, "SA": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "S6": 6, "S7": 7, "S8": 8, "S9": 9, "ST": 10, "SJ": 11, "SQ": 12, "SK": 13, "HA": 14, "H2": 15, "H3": 16, "H4": 17, "H5": 18, "H6": 19, "H7": 20, "H8": 21, "H9": 22, "HT": 23, "HJ": 24, "HQ": 25, "HK": 26, "DA": 27, "D2": 28, "D3": 29, "D4": 30, "D5": 31, "D6": 32, "D7": 33, "D8": 34, "D9": 35, "DT": 36, "DJ": 37, "DQ": 38, "DK": 39, "CA": 40, "C2": 41, "C3": 42, "C4": 43, "C5": 44, "C6": 45, "C7": 46, "C8": 47, "C9": 48, "CT": 49, "CJ": 50, "CQ": 51, "CK": 52}


class RewardModel(nn.Module):
    def __init__(self, num_players, d_model, hidden_dim, output_dim, dropout_rate):
        super(RewardModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.d_model = d_model
        
        # 将position (id)、stage、card、action这些类别变量进行embedding
        self.position_embedding = nn.Embedding(num_embeddings=num_players+1, embedding_dim=d_model) # 一个占位符0，剩下的就是player id
        self.stage_embedding = nn.Embedding(num_embeddings=len(STAGE), embedding_dim=d_model)
        self.card_embedding = nn.Embedding(num_embeddings=len(CARD), embedding_dim=d_model)  # 包括52张牌加一个占位符0
        self.action_embedding = nn.Embedding(num_embeddings=len(ACTION), embedding_dim=d_model)  # 包括5个动作加一个占位符0

        # 将chips_in_pot和pot这些连续变量进行Linear映射为d_model
        self.chips_in_pot_pot_ffn = nn.Linear(num_players+1, d_model)
        
        # # GRU for processing action history embeddings
        # self.action_history_gru
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # MLP Regressor for predicting payoff_reward
        self.regressor = nn.Sequential(
            nn.Linear(17*d_model, hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim, output_dim)
        )
        
       
    
    def forward(self, context_next_action_vec, action_history_vec):
        # 变量
        position = context_next_action_vec[:, 0].long()
        stage = context_next_action_vec[:, 1].long()
        pocket_cards = context_next_action_vec[:, 2:4].long()
        board = context_next_action_vec[:, 4:9].long()
        legal_actions = context_next_action_vec[:, 9:14].long()
        next_action = context_next_action_vec[:, -1].long()
        
        chips_in_pot_pot = context_next_action_vec[:, 14:-1]    # 连续值
        
        
        # 变量作embedding
        position_emb = self.position_embedding(position)
        stage_emb = self.stage_embedding(stage)
        pocket_cards_emb = self.card_embedding(pocket_cards).view(-1, self.d_model * 2)
        board_emb = self.card_embedding(board).view(-1, self.d_model * 5)
        legal_actions_emb = self.action_embedding(legal_actions).view(-1, self.d_model * 5)
        next_action_emb = self.action_embedding(next_action)
        
        chips_in_pot_pot_emb = self.chips_in_pot_pot_ffn(chips_in_pot_pot)
        
        
        # 处理 action_history_vec
        history_positions = action_history_vec[:, :, 0].long()
        history_actions = action_history_vec[:, :, 1].long()
        valid_mask = (history_positions != 0) | (history_actions != 0)  # 非[0,0]作为有效mask
        position_hist_emb = self.position_embedding(history_positions) * valid_mask.unsqueeze(-1)
        action_hist_emb = self.action_embedding(history_actions) * valid_mask.unsqueeze(-1)
        history_emb = position_hist_emb + action_hist_emb
        history_emb = history_emb.sum(dim=1)  # 汇总所有有效的历史嵌入，通过对seq_len求和来完成
        
        # 合并所有特征向量
        input_emb = torch.cat((position_emb, stage_emb, pocket_cards_emb, board_emb, legal_actions_emb, next_action_emb, chips_in_pot_pot_emb, history_emb), dim=1)
        
        # 通过MLP回归模型预测payoff_reward
        output = self.regressor(input_emb)
        return output