import torch
from mingpt.model import GPT
import argparse
import logging

from action_history_GPT_train import EVENT_TOKENS, EventTokenizer, generate


def load_and_generate(model_path, config, tokenizer, input_sequence, max_new_tokens=10, device="cpu", temperature=1.0, do_sample=False, top_k=None):
    # Load the saved model
    model = GPT(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Generate sequence
    generated_sequence = generate(model, tokenizer, input_sequence, max_new_tokens, device, temperature, do_sample, top_k)
    return generated_sequence


if __name__ == "__main__":
    # argparser：指定评测用的模型和GPU
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_player', type=int, default=2, help='num of players')
    parser.add_argument('--model_type', type=str, choices=['gpt-nano', 'gpt-micro', 'gpt-mini', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default="gpt-mini", help='Model type')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], default="cuda:0", help='Device')
    parser.add_argument('--seq_len', type=int, default=24, help='Maximum length of Action History (Pad + Truncate)')
    args = parser.parse_args()

    # Tokenizer
    tokenizer = EventTokenizer(EVENT_TOKENS)

    # 参数
    save_path = "./event_history_prediction/{}player/{}/".format(args.num_player, args.model_type)
    model_path = save_path + "best_gpt.pt"
    config = GPT.get_default_config()
    config.vocab_size = tokenizer.vocab_size
    config.block_size = args.seq_len  # 适当调整block size
    config.model_type = args.model_type  # 使用我们定义的小型GPT配置
    model = GPT(config)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.to(args.device)

    # 保存评估日志
    logging.basicConfig(filename=save_path+"evaluate.log", level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

    # 遍历输入序列并记录生成的序列
    input_sequences = [["CHECK_CALL"], ["RAISE_HALF_POT"], ["RAISE_POT"], ["ALL_IN"], ["FOLD"]]
    for seq in input_sequences:
        generated_sequence = generate(model, tokenizer, seq, args.seq_len, args.device)
        logging.info(f"Generated Sequence for '{seq}': {generated_sequence}\n")