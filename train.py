import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_provider.data_provider import get_dataloader
from timepred import TimePredModel
from util import get_model
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time-LLM')

    # data loader
    parser.add_argument('--data_name', type=str, default='shenzhen_8_6', help='data name')
    # parser.add_argument('--data_type', type=str, default='train', help='data type')

    # model define
    parser.add_argument('--vocab_size', type=int, default=27910, help='vocab size')
    parser.add_argument('--window_size', type=int, default=2, help='window size')
    parser.add_argument('--route_dim', type=int, default=1, help='route dimension')
    parser.add_argument('--space_dim', type=int, default=1, help='space dimension')
    parser.add_argument('--time_dim', type=int, default=1, help='time dimension')
    parser.add_argument('--route_hidden', type=int, default=16, help='route hidden dimension')
    parser.add_argument('--state_hidden', type=int, default=8, help='state hidden dimension')
    parser.add_argument('--block_dims', type=int, nargs='+', default=[512, 256, 128, 64, 32, 16, 1], help='block dimensions')

    # optimization
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--percent', type=int, default=100)

    # evaluation
    parser.add_argument('--iter_eval', type=int, default=100, help='evaluation epochs')

    args = parser.parse_args()

    train_data = get_dataloader(args, data_type='test')
    # val_data = get_dataloader(args, data_type='val')
    device = torch.device(args.device)
    model = get_model(args).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    window_size = args.window_size
    for epoch in range(args.train_epochs):
        train_loss = []
        for i, (batch_x, batch_y) in tqdm(enumerate(train_data)):
        # while True:
            # print(batch_x.shape)
            route_id, route, space_state, time_state = batch_x[:, :window_size], batch_x[:, window_size:window_size*2], batch_x[:, window_size*2:window_size*3], batch_x[:, window_size*3:-1]
            route_id = route_id.to(device, torch.int64)
            route = route.to(device)
            space_state = space_state.to(device)
            time_state = time_state.to(device)
            batch_y = batch_y.to(torch.float32).to(device)
            # route_id = torch.randint(0, args.vocab_size, (args.batch_size, window_size)).to(device)
            # space_state = torch.randn(args.batch_size, window_size, args.space_dim).to(device)
            # time_state = torch.randn(args.batch_size, args.time_dim).to(device)
            # route = torch.randn(args.batch_size, window_size, args.route_dim).to(device)
            # batch_y = torch.randint(0, 2, (args.batch_size, 1)).to(torch.float32).to(device)

            print(route_id.shape, route.shape, space_state.shape, time_state.shape)
            print(route_id[0], route[0], space_state[0], time_state[0])
            out = model(route, space_state, time_state, route_id).squeeze()         

            loss = criterion(out, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            print(loss.item())

            # if i % args.iter_eval == 0:
            #     for val_batch_x, val_batch_y in val_data:
            #         route_id, route, space_state, time_state = val_batch_x[:, :window_size], val_batch_x[:, window_size:window_size*2], val_batch_x[:, window_size*2:window_size*2+1], val_batch_x[:, window_size*2+1:-1]
            #         current_time = val_batch_x[:, -1]
            #         out = model(route, space_state, time_state, route_id)
            #         loss = criterion(out, val_batch_y)
            #     print(f'Epoch {epoch} Iter {i} Train Loss {sum(train_loss) / len(train_loss)} Val Loss {sum(val_loss) / len(val_loss)}')
            #     train_loss = []
