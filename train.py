import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from data_provider.data_provider import get_dataloader
from models.timepredmodel import TimePredModel
from util import get_model
from tqdm import tqdm
import argparse
import time
import pickle
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time-LLM')

    # data loader
    parser.add_argument('--data_name', type=str, default='shenzhen_8_6', help='data name')
    parser.add_argument('--downsample', type=bool, default=True, help='downsample')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle')

    # model define
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    # parser.add_argument('--vocab_size', type=int, default=27910, help='vocab size')
    parser.add_argument('--window_size', type=int, default=2, help='window size')
    # parser.add_argument('--route_dim', type=int, default=1, help='route dimension')
    # parser.add_argument('--space_dim', type=int, default=1, help='space dimension')
    # parser.add_argument('--time_dim', type=int, default=1, help='time dimension')
    # parser.add_argument('--route_hidden', type=int, default=16, help='route hidden dimension')
    # parser.add_argument('--state_hidden', type=int, default=8, help='state hidden dimension')
    parser.add_argument('--input_dim', type=int, default=5, )
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--block_dims', type=int, nargs='+', default=[64, 64, 32, 16], help='block dimensions')

    # optimization
    parser.add_argument('--load_model', type=bool, default=True, help='load model')
    parser.add_argument('--load_path', type=str, default="checkpoints/data_name_shenzhen_8_6 vocab_size_27910 window_size_2 route_dim_1 space_dim_1 time_dim_1 route_hidden_16 state_hidden_8 block_dims_[64, 64, 32, 16] train_epochs_100 batch_size_128 learning_rate_0.005.pth", help='load path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--pos_weight', type=float, default=None)

    # evaluation
    parser.add_argument('--epoch_eval', type=int, default=20, help='evaluation epochs')

    args = parser.parse_args()
    starttime = time.time()
    print(vars(args))

    device = torch.device(args.device)
    model = get_model(args).to(device)

    now = datetime.now()
    now = now.strftime("%y%m%d_%H%M")
    run_name = now + ' dataname_{data_name} windowsize_{window_size} blockdims_{block_dims} trainepochs_{train_epochs} batchsize_{batch_size} learningrate_{learning_rate}'.format(**vars(args))
    save_dir = os.path.join('checkpoints', run_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    train_data = get_dataloader(args, data_type='val', )
    # val_data = get_dataloader(args, data_type='test')
    device = torch.device(args.device)
    model = get_model(args).to(device)
    if args.load_model:
        print('Load model from: ', args.load_path)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Save dir: ', save_dir)
    print("Total params: ", total_params)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight])).to(device) if args.pos_weight is not None else nn.BCEWithLogitsLoss()

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    window_size = args.window_size
    train_loss = []

    for epoch in tqdm(range(args.train_epochs), unit='epoch'):
        # train
        model.train()
        for i, (batch_x, batch_y) in tqdm(enumerate(train_data), leave=False, unit='batch'):
            batch_x = batch_x[:,2:-1].to(device)
            batch_y = batch_y.to(torch.float32).to(device)

            out = model(batch_x).squeeze()    
            loss = criterion(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        print('Epoch: ', epoch)
        print('Loss: ', np.mean(train_loss[-1000:]))
        save_path = os.path.join(save_dir, 'epoch_{}.pth'.format(epoch))
        torch.save(model.state_dict(), save_path)

        # eval
        model.eval()
        if epoch % args.epoch_eval == 0:
            total_num = 0
            correct_num = 0
            for i, (batch_x, batch_y) in tqdm(enumerate(train_data), leave=False, unit='batch'):
                batch_x = batch_x[:,2:-1].to(device)
                batch_y = batch_y.to(torch.float32).to(device)

                out = model(batch_x).squeeze()  
                out = nn.functional.sigmoid(out)   
                out = (out > 0.5).float()
                correct_num += torch.sum(out == batch_y).item()
                total_num += len(batch_y)  
            print('Epoch: ', epoch)
            print('Train Accuracy: ', correct_num / total_num)

    save_path = os.path.join(save_dir, 'loss.pkl')
    with open('data.pkl', 'wb') as file:
        pickle.dump(train_loss, file)