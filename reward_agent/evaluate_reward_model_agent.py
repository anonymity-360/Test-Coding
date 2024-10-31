import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import rlcard
from rlcard.agents import RandomAgent
from rlcard.games.nolimitholdem.round import Action
from rlcard.utils import tournament

from reward_model import STAGE, ACTION, CARD
from pokerDataset import board2vec, legalAction2vec, actHistory2vec
from reward_model import RewardModel


class RewardModelBasedAgent:
    def __init__(self, model, device):
        self.use_raw = False
        self.reward_model = model
        self.device = device

    def step(self, state):
        # 训练模式下的动作选择逻辑
        return self._choose_action(state)

    def eval_step(self, state):
        # 评估模式下的动作选择逻辑
        action = self._choose_action(state)
        return action, {}

    def predict_reward(self, context_vec, action_history_vec, next_action_vec):
        # Combine context and next action vector
        context_next_action_vec = torch.tensor(context_vec + next_action_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Prepare action history vector
        action_history_vec = torch.tensor(action_history_vec, dtype=torch.long).unsqueeze(0).to(self.device)  # Add batch dimension
        
        # 进行预测
        with torch.no_grad():
            pred_reward = self.reward_model(context_next_action_vec, action_history_vec)

        return pred_reward.item()
    
    def state_to_input(self, state):
        # legal action
        legal_actions = state['raw_legal_actions'] # [<Action.FOLD: 0>, <Action.CHECK_CALL: 1>, <Action.RAISE_HALF_POT: 2>, <Action.RAISE_POT: 3>, <Action.ALL_IN: 4>]
        legal_actions_str = [action.name for action in legal_actions] # ['FOLD', 'CHECK_CALL', 'RAISE_HALF_POT', 'RAISE_POT', 'ALL_IN']
        
        # context
        num_players = len(state["raw_obs"]["all_chips"])
        position = state["raw_obs"]["current_player"]
        stage = state["raw_obs"]["stage"].name
        pocket_cards = state["raw_obs"]["hand"]
        board = state["raw_obs"]["public_cards"]
        chips_in_pot = state["raw_obs"]["all_chips"]
        pot = state["raw_obs"]["pot"]
        context = {
                "num_players": num_players,
                "position": position,
                "stage": stage,
                "pocket_cards": pocket_cards,
                "board": board,
                "chips_in_pot": chips_in_pot,
                "pot": pot,
                "legal_actions": legal_actions_str,
            }
        # context to vector
        num_player = context["num_players"]
        position = context["position"] + 1  # 编号从1开始，有利于action_history用0来占位
        stage = STAGE[context["stage"]]
        pocket_cards = [CARD[card] for card in context["pocket_cards"]]
        board = board2vec(context["board"])
        chips_in_pot = context["chips_in_pot"]
        pot = context["pot"]
        legal_actions = legalAction2vec(context["legal_actions"])
        # context_vec = [num_player, position, stage] + pocket_cards + board + chips_in_pot + [pot] + legal_actions
        context_vec = [position, stage] + pocket_cards + board + legal_actions + chips_in_pot + [pot] 
        
        # action history
        action_history = state["action_record"]
        action_history = [(id, action.name) for id, action in action_history]
        
        # action_history to vector
        action_history_vec = actHistory2vec(action_history)
        
        return context_vec, action_history_vec, legal_actions_str
        
    
    def _choose_action(self, state):
        # transform state 
        context_vec, action_history_vec, legal_actions_str = self.state_to_input(state)
        
        # 调试用
        pocket_cards = state["raw_obs"]["hand"]
        board = state["raw_obs"]["public_cards"]
        
        # Choose best action
        best_action = None
        max_reward = float('-inf')
        for next_action in legal_actions_str:
            next_action_vec = [ACTION[next_action]]
            pred_reward = self.predict_reward(context_vec, action_history_vec, next_action_vec)
            
            if pred_reward > max_reward:
                max_reward = pred_reward
                best_action = next_action
        
        return Action[best_action].value



if __name__ == "__main__":
    # Settings
    num_players = 6
    game_name = 'no-limit-holdem'
    num_eval_games = 2000 # Game number of hands
    # eval_agent = 1  # 1:random  2:MCCFR

    # Folder path
    folder_path = "./reward_model/nolimit_" + str(num_players) + "player/"
    model_path = folder_path + "best_model.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reward Model Settings
    d_model = 128
    hidden_dim = 2*d_model
    dropout_rate = 0.4

    # Load model
    model = RewardModel(num_players=num_players, d_model=d_model, hidden_dim=hidden_dim, output_dim=1, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Make environment
    env = rlcard.make(game_name, 
                      config={'allow_step_back': False,
                             "game_num_players": num_players}
                     )
    
    # Agent Settings
    rewardbased_agent = RewardModelBasedAgent(model, device)
    eval_agent = [rewardbased_agent]
    # random_agent = RandomAgent(num_actions=env.num_actions)
    for i in range(num_players-1):
        # eval_agent.append(RandomAgent(num_actions=env.num_actions))
        eval_agent.append(RewardModelBasedAgent(model, device))
    
    # if eval_agent == 1:
    # elif eval_agent == 2:
        
    
    env.set_agents(eval_agent)
    
    # Evaluateing
    eval_reward = tournament(env, num_eval_games)
    
    print(eval_reward[0])
    print(eval_reward)