import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from data_provider.data_provider import get_dataloader
from models.timepredmodel import TimePredModel
from util import get_model
from tqdm import tqdm
import argparse
import time

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
    parser.add_argument('--block_dims', type=int, nargs='+', default=[64, 64, 32, 16], help='block dimensions')

    # optimization
    parser.add_argument('--load_model', type=bool, default=False, help='load model')
    parser.add_argument('--load_path', type=str, default="checkpoints/data_name_shenzhen_8_6 vocab_size_27910 window_size_2 route_dim_1 space_dim_1 time_dim_1 route_hidden_16 state_hidden_8 block_dims_[64, 64, 32, 16] train_epochs_100 batch_size_128 learning_rate_0.005.pth", help='load path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--pos_weight', type=float, default=None)

    # evaluation
    parser.add_argument('--iter_eval', type=int, default=100, help='evaluation epochs')

    args = parser.parse_args()
    starttime = time.time()
    print(vars(args))

    device = torch.device(args.device)
    model = get_model(args).to(device)

    run_name = 'data_name_{data_name} vocab_size_{vocab_size} window_size_{window_size} route_dim_{route_dim} space_dim_{space_dim} time_dim_{time_dim} route_hidden_{route_hidden} state_hidden_{state_hidden} block_dims_{block_dims} train_epochs_{train_epochs} batch_size_{batch_size} learning_rate_{learning_rate}'.format(**vars(args))
    save_path = os.path.join('checkpoints', run_name+'.pth')

    train_data = get_dataloader(args, data_type='val', )
    # val_data = get_dataloader(args, data_type='test')
    device = torch.device(args.device)
    model = get_model(args).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Save path: ', save_path)
    print("Total params: ", total_params)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight])).to(device) if args.pos_weight is not None else nn.BCEWithLogitsLoss()

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    window_size = args.window_size
    train_loss = []

    for epoch in tqdm(range(args.train_epochs), unit='epoch'):
        for i, (batch_x, batch_y) in tqdm(enumerate(train_data), leave=False, unit='batch'):
            batch_x = batch_x[:,2:-1].to(device)
            batch_y = batch_y.to(torch.float32).to(device)

            out = model(batch_x).squeeze()    
            loss = criterion(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            print(loss.item())
        torch.save(model.state_dict(), save_path)
    print(train_loss)
    print('Time: ', time.time()-starttime)
    exit()
    for epoch in range(args.train_epochs):
        for i, (batch_x, batch_y) in tqdm(enumerate(train_data)):
            route_id, route, space_state, time_state = batch_x[:, :window_size], batch_x[:, window_size:window_size*2], batch_x[:, window_size*2:window_size*3], batch_x[:, window_size*3:-1]
            route_id = route_id.to(device, torch.int64)
            route = route.to(device)
            space_state = space_state.to(device)
            time_state = time_state.to(device)
            batch_y = batch_y.to(torch.float32).to(device)


            out = model(route, space_state, time_state, route_id).squeeze()         
            print(out)
            loss = criterion(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            print(loss.item())

        torch.save(model.state_dict(), save_path)
    print(train_loss)