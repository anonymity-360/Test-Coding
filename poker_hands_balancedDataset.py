import json
import random
from collections import defaultdict

from poker_hands_GPT_train import generate_json_paths

def load_and_balance_data_by_ratio(json_paths, max_ratio=3, merge_classes=None):
    class_samples = defaultdict(list)
    merge_classes = merge_classes or {}

    # 1. 加载数据并按类别分组
    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                category = data['cards_category']
                
                # 处理类别合并
                if category in merge_classes:
                    category = merge_classes[category]
                
                class_samples[category].append(data)
    
    # 2. 计算每个类的样本数量
    sample_counts = {category: len(samples) for category, samples in class_samples.items()}
    max_samples = max(sample_counts.values())
    min_samples = min(sample_counts.values())

    # 确保最大样本数和最小样本数的比例不超过 max_ratio
    target_samples = min(max_samples, max_ratio * min_samples)

    # 3. 采样形成均衡数据集
    balanced_samples = []
    for category, samples in class_samples.items():
        if len(samples) > target_samples:
            balanced_samples.extend(random.sample(samples, target_samples))
        else:
            # # 如果样本数量少于目标，进行上采样
            # repeats = target_samples // len(samples)
            # remainder = target_samples % len(samples)
            # balanced_samples.extend(samples * repeats + random.sample(samples, remainder))
            balanced_samples.extend(samples)

    # 打乱数据集以增加随机性
    random.shuffle(balanced_samples)
    return balanced_samples

def save_balanced_samples(balanced_samples, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for sample in balanced_samples:
            f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    player_list = [2]
    idx_list = [1, 2, 3, 4, 5]
    # idx_list = [1]
    json_paths = generate_json_paths(player_list, idx_list)
    
    # 形成平衡数据集
    max_ratio = 2
    merge_classes = {'FULL_HOUSE': 'HUGE_HANDS', 'FOUR_OF_A_KIND': 'HUGE_HANDS', 'STRAIGHT_FLUSH': 'HUGE_HANDS'}
    balanced_samples = load_and_balance_data_by_ratio(json_paths, max_ratio=max_ratio, merge_classes=merge_classes)
    save_path = "./reward_agent_data/nolimit_2player/hands_category/" + 'balanced_poker_hands.json'
    save_balanced_samples(balanced_samples, save_path)