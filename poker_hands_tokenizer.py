import json
from collections import Counter
import re


KNOWN_TOKENS = [
    # STAGE
    'PREFLOP', 'FLOP', 'TURN', 'RIVER', 'END_HIDDEN', 'SHOWDOWN',
    # ACTION
    'FOLD', 'CHECK_CALL', 'RAISE_HALF_POT', 'RAISE_POT', 'ALL_IN',
    # CARD
    'None', 'SA', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'ST', 'SJ', 'SQ', 'SK', 'HA', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'HT', 'HJ', 'HQ', 'HK', 'DA', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'DT', 'DJ', 'DQ', 'DK', 'CA', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'CT', 'CJ', 'CQ', 'CK',
    # NAME
    "num_players", "position", "stage", "pocket_cards", "board", "chips_in_pot", "pot", "action_history", "legal_actions", "cards_category",
    # position/number of players
    "0", "1", "2", "3", "4", "5", "6",
    # others
    "[", "]", ","
]

CARDS_CATEGORY = ['HIGH_CARD', 'ONE_PAIR', 'TWO_PAIR', 'THREE_OF_A_KIND', 'STRAIGHT', 'FLUSH', 'FULL_HOUSE', 'FOUR_OF_A_KIND', 'STRAIGHT_FLUSH']


def read_samples_from_json(file_path):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析JSON行为字典
            data = json.loads(line.strip())
            # 调整action_history中的索引
            formatted_action_history = [[str(action[0]), action[1]] for action in data.get('action_history', [])]
            # 构建sample字典
            sample = {
                "num_players": str(data['num_players']),
                "position": str(data['position']),
                "stage": data['stage'],
                "pocket_cards": data['pocket_cards'],
                "board": data['board'],
                "action_history": formatted_action_history,
                "legal_actions": data['legal_actions']
            }
            samples.append(sample)
    return samples


def format_poker_hand_sample(sample):
    # 处理公共牌信息，如果为空，则填充None
    board = sample["board"] if sample["board"] else [None] * 5
    # 如果公共牌部分填充，确保长度总为5
    board += [None] * (5 - len(board))

    # 转换公共盘列表为字符串，保留方括号
    board_str = ' '.join(b if b else 'None' for b in board)

    # 转换手牌为字符串，保留方括号
    pocket_cards_str = ' '.join(card for card in sample["pocket_cards"])

    # 转换行动历史为字符串，保留方括号
    action_history_str = ' '.join(
        f'[ position {action[0]} {action[1]} ]' for action in sample["action_history"])

    # 转换法律行动为字符串，保留方括号
    legal_actions_str = ' '.join(action for action in sample["legal_actions"])

    # 组合整个样本的描述
    input_text = (
        f"num_players {sample['num_players']} , position {sample['position']} , "
        f"stage {sample['stage']} , pocket_cards [ {pocket_cards_str} ] , board [ {board_str} ] , "
        f"action_history [ {action_history_str} ] , legal_actions [ {legal_actions_str} ]"
    )

    return input_text


class PokerHandTokenizer:
    def __init__(self, known_tokens):
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unknown_token = '<unk>'
        self.special_tokens = {
            self.eos_token: 0,
            self.pad_token: 1,
            self.unknown_token: 2
        }
        # 初始化词汇表
        self.token_to_idx = {token: idx + len(self.special_tokens) for idx, token in enumerate(known_tokens)}
        self.token_to_idx.update(self.special_tokens)
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.vocab_size = len(self.token_to_idx)

    def tokenize(self, text):
        # 使用正则表达式来匹配词汇
        tokens = re.findall(r'\[|\]|\,|None|\w+', text)
        return tokens

    def encode(self, sample, max_length=None):
        tokens = self.tokenize(sample)
        token_ids = [self.token_to_idx.get(token, self.token_to_idx[self.unknown_token]) for token in tokens]
        attention_mask = [1] * len(token_ids)

        # Padding
        if max_length is not None and len(token_ids) < max_length:
            padding_length = max_length - len(token_ids)
            token_ids.extend([self.token_to_idx[self.pad_token]] * padding_length)
            attention_mask.extend([0] * padding_length)

        return token_ids, attention_mask

    def decode(self, token_ids):
        tokens = [self.idx_to_token.get(idx, self.unknown_token) for idx in token_ids if idx != self.token_to_idx[self.pad_token]]
        return ' '.join(tokens)

    def update_vocab(self, samples):
        new_tokens = set()
        for sample in samples:
            tokens = self.tokenize(sample)
            new_tokens.update(tokens)

        # 筛选未知的新词汇
        new_tokens = new_tokens - set(self.token_to_idx.keys())

        # 添加新词汇到词汇表
        for token in new_tokens:
            new_index = len(self.token_to_idx)
            self.token_to_idx[token] = new_index
            self.idx_to_token[new_index] = token

        # 更新词汇表大小
        self.vocab_size = len(self.token_to_idx)


# Example usage
if __name__ == "__main__":
    num_player = 2
    file_idx = 1
    file_path = "./reward_agent_data/nolimit_"+str(num_player)+"player/hands_category/poker_hands_samples_"+str(file_idx)+".json"
    samples = read_samples_from_json(file_path)
    
    sample = samples[5]

    formatted_text = format_poker_hand_sample(sample)
    print(formatted_text)
    # 'num_players 2 , position 0 , stage PREFLOP , pocket_cards [ D7 HT ] , board [ None None None None None ] , action_history [ [ position 0 CHECK_CALL ] [ position 1 RAISE_POT ] ] , legal_actions [ FOLD CHECK_CALL RAISE_POT ALL_IN ]'
    
    
    # Initialize tokenizer with predefined tokens
    known_tokens = KNOWN_TOKENS + CARDS_CATEGORY
    tokenizer = PokerHandTokenizer(known_tokens)

    
    token_ids, attention_mask = tokenizer.encode(formatted_text, max_length=100)
    decoded_text = tokenizer.decode(token_ids)
    print("Encoded tokens:", token_ids)
    print("Decoded text:", decoded_text)

    # Update vocab in the tokenizer
    sample_new = {
        "num_players": '7', 
        "position": '0', 
        "stage": "PREFLOP", 
        "pocket_cards": ["D7", "HT"], 
        "board": ["Nihao"], 
        "action_history": [['0', "CHECK_CALL"], ['1', "RAISE_POT"]],
        "legal_actions": ["FOLD", "CHECK_CALL", "RAISE_POT", "ALL_IN"]
    }
    formatted_text_new = format_poker_hand_sample(sample_new)
    print(formatted_text_new)
    tokenizer.update_vocab([formatted_text_new])
    token_ids, attention_mask = tokenizer.encode(formatted_text_new, max_length=100)
    decoded_text = tokenizer.decode(token_ids)
    print("Encoded tokens:", token_ids)
    print("Decoded text:", decoded_text)