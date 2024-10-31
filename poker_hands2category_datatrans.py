import json
import argparse
import os


def handResults_to_pokerHandsSamples(hand_results):
    # 将Hand游戏结果拆分为牌型预测样本
    pokerHandsSamples = []
    
    for hand_result in hand_results:
        n = len(hand_result["pocket_cards"])
        game_action_history = hand_result["action_history"]
        final_cards_category = hand_result['cards_category']  # 最终的牌型
        
        for agent_id, states in enumerate(hand_result["agent_state"]):
            action_idx = 0
            
            for agent_state in states:
                # action_history：确保当前状态的动作与action_history中的动作匹配
                while action_idx < len(hand_result["action_history"]) and hand_result["action_history"][action_idx] != [agent_id, agent_state["next_action"]]:
                    action_idx += 1
                
                # 截取当前动作之前的所有动作历史
                action_history = game_action_history[:action_idx]
                
                sample = {
                    "num_players": n,
                    "position": agent_id,
                    "stage": agent_state["stage"],
                    "pocket_cards": hand_result["pocket_cards"][agent_id],
                    "board": agent_state["public_cards"],
                    "chips_in_pot": agent_state["chips_in_pot"],
                    "pot": agent_state["pot"],
                    "action_history": action_history,
                    "legal_actions": agent_state["legal_actions"],
                    "cards_category": final_cards_category[agent_id]
                }
                pokerHandsSamples.append(sample)
    return pokerHandsSamples
                

def file_process(num_players, file_index):
    input_path = f'./reward_agent_data/nolimit_{num_players}player/hand_results_nolimit_{file_index}.json'
    output_path = f'./reward_agent_data/nolimit_{num_players}player/hands_category/poker_hands_samples_{file_index}.json'

    # 读取并存储所有hand_results
    hand_results = []
    with open(input_path, 'r') as file:
        for line in file:
            hand_result = json.loads(line)
            hand_results.append(hand_result)

    # 转换为pokerHandsSamples
    all_samples = handResults_to_pokerHandsSamples(hand_results)
    
    # 将所有样本写入输出文件
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    with open(output_path, 'w') as out_file:
        for sample in all_samples:
            out_file.write(json.dumps(sample) + '\n')



if __name__ == "__main__":
    for num_players in range(2,7):
        for file_index in range(1,6):
            file_process(num_players, file_index) 