import torch
from torch import nn
import torch.nn.functional as F
import os
from data_provider.data_provider import get_dataloader
from models.timepredmodel import TimePredModel
from util import get_model
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time-LLM')

    # data loader
    parser.add_argument('--data_name', type=str, default='shenzhen_8_6', help='data name')
    parser.add_argument('--downsample', type=bool, default=False, help='downsample')

    # model define
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--vocab_size', type=int, default=27910, help='vocab size')
    parser.add_argument('--window_size', type=int, default=2, help='window size')
    parser.add_argument('--route_dim', type=int, default=1, help='route dimension')
    parser.add_argument('--space_dim', type=int, default=1, help='space dimension')
    parser.add_argument('--time_dim', type=int, default=1, help='time dimension')
    parser.add_argument('--route_hidden', type=int, default=16, help='route hidden dimension')
    parser.add_argument('--state_hidden', type=int, default=8, help='state hidden dimension')
    parser.add_argument('--input_dim', type=int, default=5, )
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--block_dims', type=int, nargs='+', default=[64, 64, 32, 16], help='block dimensions')

    # optimization
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')

    args = parser.parse_args()
    device = torch.device(args.device)
    model = get_model(args).to(device)

    # define and load model
    model = get_model(args).to(device)
    file_name = "data_name_shenzhen_8_6 vocab_size_27910 window_size_2 route_dim_1 space_dim_1 time_dim_1 route_hidden_16 state_hidden_8 block_dims_[64, 64, 32, 16] train_epochs_100 batch_size_128 learning_rate_0.005.pth"
    print(file_name)
    load_path = os.path.join('checkpoints', file_name)
    model.load_state_dict(torch.load(load_path))

    test_data = get_dataloader(args, data_type='test')
    model.eval()

    total_num = 0
    correct_num = 0
    for i, (batch_x, batch_y) in tqdm(enumerate(test_data)):
        batch_x = batch_x[:,2:-1].to(device)
        batch_y = batch_y.to(torch.float32).to(device)

        out = model(batch_x).squeeze() 
        out = nn.functional.sigmoid(out)   
        out = (out > 0.5).float()
        correct_num += torch.sum(out == batch_y).item()
        total_num += len(batch_y)
    print("total_num: ", total_num)
    print('correct_num: ', correct_num)
    print('accuracy: ', correct_num / total_num)