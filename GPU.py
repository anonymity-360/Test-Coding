import argparse
import torch
import time

parser = argparse.ArgumentParser(description='Matrix Multiplication with CUDA')
parser.add_argument('--size', type=int, default=8096, help='Size of the matrix')
parser.add_argument('--device', type=str, default='cuda:0', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'], help='Device to use for computation')
args = parser.parse_args()

# Create random tensors on specified device
a = torch.randn(args.size, args.size, device=args.device)
b = torch.randn(args.size, args.size, device=args.device)

start_time = time.time()

while True:
    # Perform matrix multiplication
    c = a @ b