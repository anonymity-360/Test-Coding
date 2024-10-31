import rlcard
from rlcard.agents import RandomAgent

import json
import time
import numpy as np

class jsonEncoder(json.JSONEncoder):
    # 避免numpy中的int64无法转化为json文件输出
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)

# Settings
num_players = 6
game_name = 'no-limit-holdem'
num_hands = 100000 # Game number of hands
output_path = "./results/nolimit_6player/"

# Make environment
env = rlcard.make(game_name, 
                  config={'allow_step_back': True,
                          "game_num_players": num_players}
                  )

# Make CFR Agent
agent1 = RandomAgent(num_actions=env.num_actions)
agent2 = RandomAgent(num_actions=env.num_actions)
agent3 = RandomAgent(num_actions=env.num_actions)
agent4 = RandomAgent(num_actions=env.num_actions)
agent5 = RandomAgent(num_actions=env.num_actions)
agent6 = RandomAgent(num_actions=env.num_actions)

env.set_agents([agent1, agent2, agent3, agent4, agent5, agent6])
ACTION_LIST = env.actions

start_time = time.time()
# Gaming
game_results = []
for hand in range(num_hands):
    print(f'Episode {hand + 1}/{num_hands}')
    # 初始化游戏状态
    trajectories, payoffs = env.run(is_training=False) # 注意一下，这里payoff是除以了BB数的，即最后输赢了多少BB
    
    # 取出每一局需要的过程和结果
    action_history = trajectories[0][0]["action_record"] # [(1, 'raise'), (0, 'call')
    action_history = [(action[0], action[1].name) for action in action_history]
    
    # 手牌
    pocket_cards = env.get_perfect_information()["hand_cards"]
    
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
    
    
    game_result = {
        "pocket_cards": pocket_cards,
        "action_history": action_history,
        "payoffs": [int(payoff) for payoff in payoffs.tolist()],
        "agent_state": agent_state
    }
    
    game_results.append(game_result)

# 将hand游戏结果保存到JSON文件中
output_file = output_path + 'no-limit_holdem_hand_results.json'
with open(output_file, 'w') as f:
    for game_result in game_results:
        json_line = json.dumps(game_result, cls=jsonEncoder)
        f.write(json_line + '\n')

print(f'Hand Game results have been saved to {output_file}')

end_time = time.time()
print(f"Generating {num_hands} hands has spent {end_time-start_time} seconds")


# 将Hand游戏结果拆分到每次action的行动
context_list = []
action_history_list = []
next_action_list = []
payoff_reward_list = []

for game_result in game_results:
    n = len(game_result["pocket_cards"])
    game_action_history = game_result["action_history"]
    
    for agent_id, states in enumerate(game_result["agent_state"]):
        action_idx = 0
        
        for agent_state in states:
            context = {
                "num_players": n,
                "position": agent_id,
                "stage": agent_state["stage"],
                "pocket_cards": game_result["pocket_cards"][agent_id],
                "board": agent_state["public_cards"],
                "chips_in_pot": agent_state["chips_in_pot"],
                "pot": agent_state["pot"],
                "legal_actions": agent_state["legal_actions"],
            }
            
            # action_history：确保当前状态的动作与action_history中的动作匹配
            while action_idx < len(game_result["action_history"]) and game_result["action_history"][action_idx] != (agent_id, agent_state["next_action"]):
                action_idx += 1
            
            # 截取当前动作之前的所有动作历史
            action_history = game_action_history[:action_idx]
            
            next_action = agent_state["next_action"]
            
            payoff_reward = game_result["payoffs"][agent_id]
            
            context_list.append(context)
            action_history_list.append(action_history)
            next_action_list.append(next_action)
            payoff_reward_list.append(payoff_reward)

# 将每一步游戏结果保存到JSON文件中
output_file = output_path + 'no-limit_holdem_action_results.json'
with open(output_file, 'w') as f:
    for i,_ in enumerate(context_list):
        tmp = {
            "context": context_list[i],
            "action_history": action_history_list[i],
            "next_action": next_action_list[i],
            "payoff_reward": payoff_reward_list[i]
        }
        json_line = json.dumps(tmp, cls=jsonEncoder)
        f.write(json_line + '\n')

print(f'Action Game results have been saved to {output_file}')

end_time = time.time()
print(f"Generating {i} actions has spent {end_time-start_time} seconds")