import torch
import torch.nn as nn
import torch.nn.functional as F
from timepred import TimePredModel
from util import get_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time-LLM')

    # 

    # model define
    parser.add_argument('--window_size', type=int, default=3, help='window size')
    parser.add_argument('--route_dim', type=int, default=3, help='route dimension')
    parser.add_argument('--space_dim', type=int, default=1, help='space dimension')
    parser.add_argument('--time_dim', type=int, default=1, help='time dimension')
    parser.add_argument('--route_hidden', type=int, default=16, help='route hidden dimension')
    parser.add_argument('--state_hidden', type=int, default=8, help='state hidden dimension')
    parser.add_argument('--block_dims', type=int, nargs='+', default=[512, 256, 128, 64, 32, 16, 1], help='block dimensions')
    args = parser.parse_args()

    model = get_model(args)
    
    B = 32
    route = torch.randn(B, args.window_size, args.route_dim)
    space_state = torch.randn(B, args.window_size, args.space_dim)
    time_state = torch.randn(B, args.time_dim)
    out = model(route, space_state, time_state)#.squeeze()
    print(out.shape)
    print(out)
    print(out[0])