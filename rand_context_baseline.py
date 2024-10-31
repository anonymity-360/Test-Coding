from torch.utils.data import DataLoader
import logging
import os
import argparse

from context_test import GPT_context
from context_eval import EventHistoryContext_LegalActions_Dataset, context_collate_fn, evaluate
from utils import STAGE_TOKENS, CARD_TOKENS, SimpleTokenizer
from action_history_GPT_train import EventTokenizer, EVENT_TOKENS

if __name__ == "__main__":
    # argparser：指定评测用的模型和GPU
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_player', type=int, default=6, help='num of players')
    parser.add_argument('--model_type', type=str, choices=['gpt-nano', 'gpt-micro', 'gpt-mini', 'gopher-44m', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gopher-44m', help='Model type')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], default='cuda:2', help='Device')
    parser.add_argument('--seq_len', type=int, default=48, help='Maximum length of Action History (Pad + Truncate)')
    args = parser.parse_args()
    
    # 后续可以再生成100万对局数据到test.json，专门用来评测，目前先用这些数据
    batch_size = 4096
    dataset_path = "./reward_agent_dataset"
    dataset_json_idx = [5]
    
    # 加载tokenizer & 数据集设置
    json_files = [f"{dataset_path}/nolimit_{args.num_player}player/hand_results_nolimit_{idx}.json" for idx in dataset_json_idx]
    tokenizer = EventTokenizer(EVENT_TOKENS)
    stage_tokenizer = SimpleTokenizer(STAGE_TOKENS)
    card_tokenizer = SimpleTokenizer(CARD_TOKENS)
    dataset = EventHistoryContext_LegalActions_Dataset(json_files, tokenizer, args.seq_len)
    
    # 用Dataloader，分Batch验证
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=context_collate_fn)
    
    # GPT2-模型设置
    config = GPT_context.get_default_config()
    config.vocab_size = tokenizer.vocab_size
    config.block_size = args.seq_len  # 适当调整block size
    config.model_type = args.model_type  # 使用我们定义的小型GPT配置
    config.num_player = args.num_player  # 添加玩家数量配置
    model = GPT_context(config, stage_tokenizer, card_tokenizer)
    
    # 评估模型
    perplexity, bleu_score, class_report, acc, legal_rate = evaluate(model, test_loader, tokenizer, device=args.device)
    
    # 保存评估日志
    logging.basicConfig(filename=os.path.join("./rand_evaluate.log"), level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f'Accuracy: {acc}\nPerplexity: {perplexity}\nBLEU Score: {bleu_score}\nLegal Rate of Predicted Events: {legal_rate}\nClassification Report:\n{class_report}')
    
    print(f'Accuracy: {acc}')
    print(f'Perplexity: {perplexity}')
    print(f'BLEU Score: {bleu_score}')
    print(f"Legal Rate of Predicted Events: {legal_rate}")
    print(f'Classification Report:\n{class_report}')