import torch
import rlcard
import os
import time
import json
import numpy as np
import argparse
from enum import Enum

from rlcard.games.limitholdem.utils import Hand

from reward_model import RewardModel
from evaluate_reward_model_agent import RewardModelBasedAgent

class CARD_CATEGORY(Enum):
    # 牌型
    STRAIGHT_FLUSH = 9
    FOUR_OF_A_KIND = 8
    FULL_HOUSE = 7
    FLUSH = 6
    STRAIGHT = 5
    THREE_OF_A_KIND = 4
    TWO_PAIR = 3
    ONE_PAIR = 2
    HIGH_CARD = 1
    INVAVLID = 0

 
def number_to_card_category(number):
    for category in CARD_CATEGORY:
        if category.value == number:
            return category.name
    raise ValueError(f"没有找到对应的CARD_CATEGORY: {number}")


class jsonEncoder(json.JSONEncoder):
    # 避免numpy中的int64无法转化为json文件输出
    def default(self, obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        return super().default(obj)


def Hand_Results_Generate(num_hands, env):
    # Generate Gaming Data - Hand（一整局存一条数据）
    ACTION_LIST = env.actions
    start_time = time.time()
    hand_results = []
    for hand in range(num_hands):
        print(f'Episode {hand + 1}/{num_hands}')
        # 初始化游戏状态
        trajectories, payoffs = env.run(is_training=False) # 注意一下，这里payoff是除以了BB数的，即最后输赢了多少BB
        
        # 取出每一局需要的过程和结果
        action_history = trajectories[0][0]["action_record"] # [(1, 'raise'), (0, 'call')
        action_history = [(action[0], action[1].name) for action in action_history]
        
        # 手牌
        pocket_cards = env.get_perfect_information()["hand_cards"]
        
        # 公共牌
        public_cards_all = env.get_perfect_information()["public_card"]
        if public_cards_all == None:
            # 游戏在中途结束了，还没发完牌，则补全所有公共牌
            public_cards_all = []
            for i in range(5):
                public_cards_all.append(env.game.dealer.deal_card().get_index())
        elif len(public_cards_all) < 5:
            # 游戏在中途结束了，还没发完牌，则补全所有公共牌
            while len(public_cards_all) < 5:
                public_cards_all.append(env.game.dealer.deal_card().get_index())

        
        # 牌型
        cards_category = []
        for card in pocket_cards:
            hand = Hand(card + public_cards_all)
            hand.evaluateHand()
            cards_category.append(number_to_card_category(hand.category))
        
        
        # 状态
        agent_state = []
        for id in range(num_players):
            n = len(trajectories[id])
            state = []
            for i in range(0, n-1, 2):
                
                stage = trajectories[id][i]["raw_obs"]["stage"].name
                public_cards = trajectories[id][i]["raw_obs"]["public_cards"]
                
                chips_in_pot = trajectories[id][i]["raw_obs"]["all_chips"]
                pot = trajectories[id][i]["raw_obs"]["pot"]
                
                legal_actions = trajectories[id][i]["raw_obs"]["legal_actions"]
                legal_actions = [action.name for action in legal_actions]
                
                next_action = ACTION_LIST(trajectories[id][i+1]).name
                tmp_state = {
                    "stage": stage,
                    "chips_in_pot": chips_in_pot,
                    "pot": pot,
                    "public_cards": public_cards,
                    "legal_actions": legal_actions,
                    "next_action": next_action 
                }

                state.append(tmp_state)
            
            agent_state.append(state)
        
        
        hand_result = {
            "pocket_cards": pocket_cards,
            "action_history": action_history,
            "public_cards": public_cards_all,
            "cards_category": cards_category,
            "payoffs": [int(payoff) for payoff in payoffs.tolist()],
            "agent_state": agent_state
        }
        
        hand_results.append(hand_result)

    end_time = time.time()
    print(f"Generating {num_hands} hands has spent {end_time-start_time} seconds")
    
    return hand_results


def Action_Results_Generate(hand_results):
    # 将Hand游戏结果拆分到每次action的行动
    context_list = []
    action_history_list = []
    next_action_list = []
    payoff_reward_list = []

    for hand_result in hand_results:
        n = len(hand_result["pocket_cards"])
        game_action_history = hand_result["action_history"]
        
        for agent_id, states in enumerate(hand_result["agent_state"]):
            action_idx = 0
            
            for agent_state in states:
                context = {
                    "num_players": n,
                    "position": agent_id,
                    "stage": agent_state["stage"],
                    "pocket_cards": hand_result["pocket_cards"][agent_id],
                    "board": agent_state["public_cards"],
                    "chips_in_pot": agent_state["chips_in_pot"],
                    "pot": agent_state["pot"],
                    "legal_actions": agent_state["legal_actions"],
                }
                
                # action_history：确保当前状态的动作与action_history中的动作匹配
                while action_idx < len(hand_result["action_history"]) and hand_result["action_history"][action_idx] != (agent_id, agent_state["next_action"]):
                    action_idx += 1
                
                # 截取当前动作之前的所有动作历史
                action_history = game_action_history[:action_idx]
                
                next_action = agent_state["next_action"]
                
                payoff_reward = hand_result["payoffs"][agent_id]
                
                context_list.append(context)
                action_history_list.append(action_history)
                next_action_list.append(next_action)
                payoff_reward_list.append(payoff_reward)
    
    return context_list, action_history_list, next_action_list, payoff_reward_list

    

if __name__ == "__main__":
    # 使用argparse接收命令行参数
    parser = argparse.ArgumentParser(description='Training Reward Model with Different Number of Agents')
    parser.add_argument('--num_players', type=int, default=2,
                        help='Number of Agents')
    args = parser.parse_args()
    
    # Settings
    num_players = args.num_players
    game_name = 'no-limit-holdem'
    num_hands = 500 * 10000 # Game number of hands
    # num_hands = 1000
    save_every_hands = 100 * 10000 # Save every 1 million hands
    model_path = "./reward_model/nolimit_" + str(num_players) + "player/" + "best_model.pt"
    output_path = "./reward_agent_data/nolimit_" + str(num_players) + "player/"
    if not os.path.exists(output_path): os.makedirs(output_path)
    
    # Reward Model Settings
    d_model = 128
    hidden_dim = 2*d_model
    dropout_rate = 0.4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = RewardModel(num_players=num_players, d_model=d_model, hidden_dim=hidden_dim, output_dim=1, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Make Agents
    agents = []
    for i in range(num_players):
        agents.append(RewardModelBasedAgent(model, device))
    
    # Make environment
    env = rlcard.make(game_name, 
                      config={'allow_step_back': False,
                             "game_num_players": num_players}
                     )
    env.set_agents(agents)
    
    # Generate Gaming Data - Hand（一整局存一条数据）
    hand_results = Hand_Results_Generate(num_hands, env)
    # output_file = output_path + 'hand_results_nolimit.json'
    # with open(output_file, 'w') as f:
    #     for hand_result in hand_results:
    #         json_line = json.dumps(hand_result, cls=jsonEncoder)
    #         f.write(json_line + '\n')
    # print(f'Hand Game results have been saved to {output_file}')
    
    
    file_index = 1
    results_count = 0
    current_file = open(f"{output_path}hand_results_nolimit_{num_players}players_{file_index}.json", 'w')

    try:
        for hand_result in hand_results:
            json_line = json.dumps(hand_result, cls=jsonEncoder)  # Assuming jsonEncoder is defined elsewhere
            current_file.write(json_line + '\n')
            results_count += 1
            
            if results_count >= save_every_hands:
                current_file.close()
                file_index += 1
                current_file = open(f"{output_path}hand_results_nolimit_{file_index}.json", 'w')
                results_count = 0
    finally:
        current_file.close()
    
    print(f'Hand Game results have been saved starting with {output_path}hand_results_nolimit_1.json')
    
    # # Generate Gaming Data - Action（每次行动存一条数据，用于训练Reward Model）
    # context_list, action_history_list, next_action_list, payoff_reward_list = Action_Results_Generate(hand_results)
    # output_file = output_path + 'action_results_nolimit.json'
    # with open(output_file, 'w') as f:
    #     for i,_ in enumerate(context_list):
    #         tmp = {
    #             "context": context_list[i],
    #             "action_history": action_history_list[i],
    #             "next_action": next_action_list[i],
    #             "payoff_reward": payoff_reward_list[i]
    #         }
    #         json_line = json.dumps(tmp, cls=jsonEncoder)
    #         f.write(json_line + '\n')
    # print(f'Action Game results have been saved to {output_file}')