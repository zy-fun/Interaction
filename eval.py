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
    parser.add_argument('--downsample', type=bool, default=True, help='downsample')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle')

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
    parser.add_argument('--block_dims', type=int, nargs='+', default=[64, 32, 16, 8], help='block dimensions')

    # evaluation
    parser.add_argument('--load_model', type=bool, default=True, help='load model')
    parser.add_argument('--load_path', type=str, default="checkpoints/241112_1616 dataname_shenzhen_8_6 windowsize_2 blockdims_[64, 64, 32, 16] trainepochs_200 batchsize_256 learningrate_0.01/epoch_199.pth", help='load path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')

    args = parser.parse_args()
    device = torch.device(args.device)
    model = get_model(args).to(device)
    model.train()

    for data_type in ['val', 'test']:
        data = get_dataloader(args, data_type=data_type)

        total_num = 0
        correct_num = 0
        criterion = nn.BCEWithLogitsLoss()
        total_loss = []
        for i, (batch_x, batch_y) in tqdm(enumerate(data)):
            batch_x = batch_x[:,2:-1].to(device)
            batch_y = batch_y.to(torch.float32).to(device)

            out = model(batch_x).squeeze() 
            loss = criterion(out, batch_y)
            total_loss.append(loss.item())
            out = nn.functional.sigmoid(out)   
            out = (out > 0.5).float()
            correct_num += torch.sum(out == batch_y).item()
            total_num += len(batch_y)

        print(batch_x)
        print(out)
        print(batch_y)
        print(data_type, "total_num: ", total_num)
        print(data_type, 'correct_num: ', correct_num)
        print(data_type, 'accuracy: ', correct_num / total_num)
        print(data_type, 'loss: ', sum(total_loss) / len(total_loss))