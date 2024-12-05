import torch
from torch import nn
import torch.nn.functional as F
import os
from data_provider.data_provider import get_dataloader
from models.timepredmodel import TimePredModel
from util import get_model
from tqdm import tqdm
import argparse

batch_x = torch.tensor([[ 1.7815,  1.1445, -0.4611, -0.6619, -0.7471, -0.7471, -0.7468,  0.4380],
        [ 2.0787, -0.5749, -0.4996, -0.2951, -0.5749, -0.5749, -0.5713,  1.0121],
        [ 1.9539,  1.0588, -0.6325, -0.6366, -0.6393, -0.6393, -0.6390,  0.1740],
        [ 2.3753, -0.4837, -0.4782, -0.2982, -0.4837, -0.4837, -0.4789,  0.3309],
        [ 1.5399,  1.4962, -0.6776, -0.6498, -0.6820, -0.6820, -0.6819,  0.3372],
        [ 1.0758,  1.7872, -0.6540, -0.6730, -0.6975, -0.6975, -0.6969,  0.5560],
        [ 1.4635,  1.4639, -0.6777, -0.6914, -0.7027, -0.7027, -0.7004,  0.5474],
        [ 1.8650,  1.2453, -0.6133, -0.6321, -0.6341, -0.6341, -0.6249,  0.0281],
        [ 1.5747,  1.5750, -0.6365, -0.6277, -0.6427, -0.6427, -0.6425,  0.0424],
        [ 2.1010, -0.5524, -0.5433, -0.3565, -0.5524, -0.5524, -0.5282,  0.9843],
        [ 2.1938, -0.5465, -0.5259, -0.2810, -0.5465, -0.5465, -0.5460,  0.7986],
        [ 1.1508,  1.1448, -0.6929, -0.6883, -0.7624, -0.7624, -0.7103,  1.3207],
        [-0.0955, -0.0946, -0.4406, -0.3795, -0.4772, -0.4772, -0.4769,  2.4415],
        [ 2.3794, -0.4908, -0.4375, -0.2918, -0.4908, -0.4908, -0.4908,  0.3133],
        [ 1.6010,  1.5552, -0.6036, -0.6152, -0.6530, -0.6530, -0.6497,  0.0185],
        [ 0.3484,  2.2028, -0.5828, -0.6037, -0.6091, -0.6091, -0.6069,  0.4604],
        [ 1.2001,  1.2032, -0.6471, -0.6732, -0.7663, -0.7663, -0.7663,  1.2158],
        [-0.3377,  2.3231, -0.4976, -0.3997, -0.5280, -0.5280, -0.5278,  0.4958],
        [ 0.9556,  1.7996, -0.6480, -0.6842, -0.7057, -0.7057, -0.7001,  0.6885],
        [ 1.8424,  1.1985, -0.4240, -0.6588, -0.7130, -0.7130, -0.7115,  0.1795],
        [ 2.1564, -0.5530, -0.4664, -0.3610, -0.5530, -0.5530, -0.5526,  0.8826],
        [ 1.5458,  1.5406, -0.6369, -0.6402, -0.6781, -0.6781, -0.6733,  0.2203],
        [ 1.3799,  1.4083, -0.6758, -0.6556, -0.7404, -0.7404, -0.7394,  0.7635],
        [ 1.3359, -0.1501, -0.5673, -0.5403, -0.6388, -0.6388, -0.6340,  1.8333],
        [ 1.9442,  1.0929, -0.6359, -0.4050, -0.6954, -0.6954, -0.6944,  0.0890],
        [ 0.7634,  1.5737, -0.6069, -0.7208, -0.7500, -0.7500, -0.7053,  1.1960],
        [ 0.9626,  1.8336, -0.6675, -0.6647, -0.6938, -0.6938, -0.6934,  0.6169],
        [ 0.9170,  0.9217, -0.6661, -0.6802, -0.7255, -0.7255, -0.7231,  1.6816],
        [ 0.2258,  2.2600, -0.5759, -0.5728, -0.5845, -0.5845, -0.5761,  0.4079],
        [ 1.4155, -0.5923, -0.5876, -0.2453, -0.5923, -0.5923, -0.5915,  1.7857],
        [ 1.6921,  1.1160, -0.6944, -0.6938, -0.6995, -0.6995, -0.6992,  0.6784],
        [-0.2986,  2.1208, -0.5667, -0.4544, -0.5795, -0.5795, -0.5793,  0.9372]])

batch_y = torch.tensor([False,  True, False, False,  True,  True, False, False, False,  True,
         True,  True,  True, False,  True, False, False,  True,  True,  True,
        False, False,  True,  True,  True,  True,  True, False,  True,  True,
         True,  True])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time-LLM')

    # model define
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--vocab_size', type=int, default=27910, help='vocab size')
    parser.add_argument('--window_size', type=int, default=2, help='window size')
    parser.add_argument('--route_dim', type=int, default=1, help='route dimension')
    parser.add_argument('--space_dim', type=int, default=1, help='space dimension')
    parser.add_argument('--time_dim', type=int, default=1, help='time dimension')
    parser.add_argument('--route_hidden', type=int, default=16, help='route hidden dimension')
    parser.add_argument('--state_hidden', type=int, default=8, help='state hidden dimension')
    parser.add_argument('--input_dim', type=int, default=8, )
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--block_dims', type=int, nargs='+', default=[64, 32], help='block dimensions')

    # optimization
    parser.add_argument('--load_model', type=bool, default=True, help='load model')
    parser.add_argument('--load_path', type=str, default="checkpoints/241114_1650 dataname_mini_data windowsize_2 blockdims_[64, 32] trainepochs_300 batchsize_256 learningrate_0.01/epoch_125.pth", help='load path')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')

    args = parser.parse_args()
    device = torch.device(args.device)
    print(args.device)
    model = get_model(args).to(device)
    model.train()

    # batch_x = batch_x[:,2:-1].to(device)
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    print(batch_x.shape)
    print(batch_y.shape)
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}")
    #     print(f"Parameter value: {param}\n")

    # fc1 = model.blocks[0].fc1
    # print(fc1.weight)
    # print(fc1.bias)
    # print(fc1(batch_x).squeeze())
    # exit()
    
    def hook_fn(module, input, output):
        intermediate_output = output
        # try:
        #     print(module.weight)
        #     print(module.bias)
        # except:
        #     pass
        print(module)
        print(output.squeeze())
        print()

    for i, block in enumerate(model.blocks):
        for layer in block.children():
            layer.register_forward_hook(hook_fn)

    # for module in model.blocks:
    #     model.blocks[1].fc1.register_forward_hook(hook_fn)
    #     model.blocks[1].bn1.register_forward_hook(hook_fn)
    #     model.blocks[1].relu1.register_forward_hook(hook_fn)
    #     model.blocks[1].fc2.register_forward_hook(hook_fn)
    #     model.blocks[1].bn2.register_forward_hook(hook_fn)
    #     # model.blocks[1].fc_res.register_forward_hook(hook_fn)
    #     model.blocks[1].bn_res.register_forward_hook(hook_fn)
    #     model.blocks[1].relu2.register_forward_hook(hook_fn)
    #     break

    out = model(batch_x).squeeze()
    print()
    print(out)

    out = nn.functional.sigmoid(out)
    out = out > 0.5
    print()
    print(out.float())

    print()
    print(batch_y.squeeze())

    acc = torch.sum(out == batch_y).item() / len(batch_y)
    print()
    print(acc) 

