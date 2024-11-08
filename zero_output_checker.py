import torch
from torch import nn
import torch.nn.functional as F
import os
from data_provider.data_provider import get_dataloader
from models.timepredmodel import TimePredModel
from util import get_model
from tqdm import tqdm
import argparse

batch_x = torch.tensor([[1.6460e+02, 1.1760e+02, 2.6675e-01, 0.0000e+00, 1.0000e+00],
    [5.8900e+02, 1.6993e+03, 2.6675e-01, 0.0000e+00, 4.3000e+01],
    [1.8600e+02, 1.6993e+03, 2.6675e-01, 0.0000e+00, 2.1000e+01],
    [3.4130e+02, 1.3130e+02, 2.6675e-01, 0.0000e+00, 2.2600e+02],
    [3.5600e+01, 1.6993e+03, 5.3350e-01, 0.0000e+00, 8.3000e+01],
    [7.9200e+01, 1.6993e+03, 2.6675e-01, 0.0000e+00, 1.3000e+01],
    [2.8580e+02, 2.2370e+02, 2.6675e-01, 0.0000e+00, 0.0000e+00],
    [1.1720e+02, 7.8300e+01, 2.6675e-01, 0.0000e+00, 5.0000e+00],
    [4.1310e+02, 1.6993e+03, 1.0670e+00, 0.0000e+00, 5.8100e+02],
    [3.1200e+02, 1.0092e+03, 2.6675e-01, 0.0000e+00, 3.0000e+00],
    [1.4120e+02, 7.3100e+01, 2.6675e-01, 0.0000e+00, 2.0000e+00],
    [2.0900e+01, 2.7870e+02, 2.6675e-01, 0.0000e+00, 9.0000e+00],
    [1.9140e+02, 1.6993e+03, 1.0670e+00, 0.0000e+00, 1.9600e+02],
    [5.8460e+02, 9.6000e+01, 1.3338e+00, 2.6675e-01, 4.0000e+00],
    [4.1310e+02, 1.6993e+03, 2.1340e+00, 0.0000e+00, 1.2700e+02],
    [7.1960e+02, 3.6990e+02, 2.6675e-01, 5.3350e-01, 1.8000e+01],
    [3.1360e+02, 4.1960e+02, 2.1340e+00, 1.6005e+00, 6.3000e+01],
    [1.4810e+02, 5.8560e+02, 2.6675e-01, 2.6675e-01, 5.0000e+00],
    [2.4540e+02, 5.0400e+01, 2.6675e-01, 0.0000e+00, 0.0000e+00],
    [3.4530e+02, 7.9800e+01, 1.6005e+00, 2.1340e+00, 1.0000e+00],
    [1.8570e+02, 6.8000e+01, 2.6675e-01, 2.6675e-01, 2.4000e+01],
    [6.2900e+01, 1.0950e+02, 2.6675e-01, 0.0000e+00, 2.0000e+00],
    [2.4540e+02, 3.2380e+02, 8.0026e-01, 1.0670e+00, 7.0000e+01],
    [8.8000e+01, 2.8830e+02, 2.6675e-01, 1.6005e+00, 1.1000e+01],
    [5.5300e+01, 1.6993e+03, 2.6675e-01, 0.0000e+00, 0.0000e+00],
    [3.2160e+02, 2.4540e+02, 2.1340e+00, 3.2010e+00, 1.6100e+02],
    [1.8470e+02, 3.1400e+02, 5.3350e-01, 0.0000e+00, 3.0000e+00],
    [2.3270e+02, 2.2970e+02, 2.6675e-01, 0.0000e+00, 2.2000e+01],
    [1.9140e+02, 3.2160e+02, 1.6005e+00, 2.1340e+00, 1.4000e+01],
    [4.4820e+02, 2.4540e+02, 2.4008e+00, 8.0026e-01, 4.5000e+01],
    [3.0120e+02, 4.3940e+02, 2.6675e-01, 2.6675e-01, 4.0000e+01],
    [3.6110e+02, 1.8580e+02, 2.6675e-01, 0.0000e+00, 5.1000e+01],
    [7.8900e+01, 1.6993e+03, 1.8673e+00, 0.0000e+00, 3.1900e+02],
    [3.2380e+02, 3.4530e+02, 2.9343e+00, 2.6675e+00, 1.3500e+02],
    [2.2570e+02, 2.7730e+02, 2.6675e-01, 0.0000e+00, 2.5000e+01],
    [1.1920e+02, 4.6830e+02, 2.6675e-01, 0.0000e+00, 4.0000e+00],
    [3.8730e+02, 2.5260e+02, 1.3338e+00, 2.6675e-01, 3.6000e+01],
    [2.4480e+02, 8.8700e+01, 8.0026e-01, 0.0000e+00, 1.2300e+02],
    [2.7000e+02, 9.5600e+01, 8.0026e-01, 0.0000e+00, 5.0000e+00],
    [5.2200e+01, 6.8800e+01, 2.6675e-01, 2.6675e-01, 8.0000e+00],
    [3.5930e+02, 6.0900e+01, 2.6675e-01, 0.0000e+00, 4.7000e+01],
    [1.2910e+02, 5.6400e+01, 2.6675e-01, 2.6675e-01, 2.0000e+00],
    [1.4560e+02, 3.5160e+02, 2.6675e-01, 0.0000e+00, 6.0000e+00],
    [3.3610e+02, 1.9400e+01, 8.0026e-01, 0.0000e+00, 1.4300e+02]],)

batch_y = torch.tensor([1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1.,
    0., 0., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 0., 0., 1., 0.,
    0., 0., 0., 1., 0., 1., 1., 1])

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
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')

    args = parser.parse_args()
    device = torch.device(args.device)
    model = get_model(args).to(device)

    file_name = "data_name_shenzhen_8_6 vocab_size_27910 window_size_2 route_dim_1 space_dim_1 time_dim_1 route_hidden_16 state_hidden_8 block_dims_[64, 64, 32, 16] train_epochs_100 batch_size_128 learning_rate_0.005.pth"
    print(file_name)
    load_path = os.path.join('checkpoints', file_name)
    model.load_state_dict(torch.load(load_path))

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}")
    #     print(f"Parameter value: {param}\n")
    
    def hook_fn(module, input, output):
        intermediate_output = output
        print(module)
        print(output.squeeze())
        print()

    for module in model.blocks:
        model.blocks[1].fc1.register_forward_hook(hook_fn)
        model.blocks[1].bn1.register_forward_hook(hook_fn)
        model.blocks[1].relu1.register_forward_hook(hook_fn)
        model.blocks[1].fc2.register_forward_hook(hook_fn)
        model.blocks[1].bn2.register_forward_hook(hook_fn)
        # model.blocks[1].fc_res.register_forward_hook(hook_fn)
        model.blocks[1].bn_res.register_forward_hook(hook_fn)
        model.blocks[1].relu2.register_forward_hook(hook_fn)
        break
    out = model(batch_x).squeeze()
    print()
    print(out)

    out = nn.functional.sigmoid(out)
    out = out > 0.5
    print()
    print(out)

    print()
    print(batch_y.squeeze())

