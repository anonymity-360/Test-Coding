import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import math
import torch
import torch.nn as nn

# Tokens Definition
STAGE_TOKENS = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
ACTION_TOKENS = ['FOLD', 'CHECK_CALL', 'RAISE_HALF_POT', 'RAISE_POT', 'ALL_IN']
DEALER_TOKENS = ['DEAL_THREECARDS', 'DEAL_ONECARD']
CARD_TOKENS = ['None', 'SA', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'ST', 'SJ', 'SQ', 'SK', 'HA', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'HT', 'HJ', 'HQ', 'HK', 'DA', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'DT', 'DJ', 'DQ', 'DK', 'CA', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'CT', 'CJ', 'CQ', 'CK']
CARDS_CATEGORY = ['HIGH_CARD', 'ONE_PAIR', 'TWO_PAIR', 'THREE_OF_A_KIND', 'STRAIGHT', 'FLUSH', 'FULL_HOUSE', 'FOUR_OF_A_KIND', 'STRAIGHT_FLUSH']


def loss_figure(train_losses, test_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Evaluating Loss')
    plt.title('Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path + "loss_plot.png", dpi=400)
    print("Image saved")


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    # cm = confusion_matrix(y_true, y_pred, normalize="all")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True Label")
    ax.set_title("confusion matrix")
    plt.tight_layout()
    plt.savefig(save_path+"confusion.png", format="png")


class SimpleTokenizer:
    # 根据tokens产生基本Tokenizer，不包含special tokens
    def __init__(self, tokens):
        self.tokens = tokens

        self.token_to_id = {token: i for i, token in enumerate(tokens)}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
    
    def encode(self, tokens):
        return [self.token_to_id[token] for token in tokens]
    
    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return [self.id_to_token[id] for id in ids]

