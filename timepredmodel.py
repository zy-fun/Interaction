import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, activation=True):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

        if input_dim != output_dim:
            self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        identity = x  

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)

        if x.size(-1) != out.size(-1):
            identity = self.fc(identity)
        out += identity
        return self.relu(out) if self.activation else out

class TimePredModel(nn.Module):
    def __init__(
            self, 
            route_dim: int = 1, 
            space_dim: int = 1, 
            time_dim: int = 1, 
            vocab_size: int | None = None,
            route_hidden: int = 16,
            # route_feature_hidden: int = 16,
            state_hidden: int = 16,
            window_size: int = 2, 
            block_dims: list = [], 
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.window_size = window_size
        if self.vocab_size:
            self.road_seg_emb = nn.Embedding(vocab_size, route_hidden)
            self.road_seg_emb_proj = nn.Linear(route_hidden * self.window_size, route_hidden)

        self.route_encoder = Block(route_dim * window_size, route_hidden)
        self.state_encoder = Block(space_dim * window_size + time_dim, state_hidden)

        if self.vocab_size: 
            block_dims = [2 * route_hidden + state_hidden] + block_dims
        else:
            block_dims = [route_hidden + state_hidden] + block_dims
        block_dims = zip(block_dims[:-1], block_dims[1:])
        self.blocks = nn.ModuleList([Block(in_dim, out_dim) for in_dim, out_dim in block_dims])
        # self.sigmoid = nn.Sigmoid() 

    def forward(self, route, space_state, time_state, route_id=None):
        '''
        input:
            route: (batch_size, window_size, route_dim)
            space_state: (batch_size, window_size, space_dim)
            time_state: (batch_size, time_dim)
            route_id: (batch_size, window_size)
        '''
        B = route.size(0)

        route = route.reshape(B, -1)
        route = self.route_encoder(route)
        if route_id is not None:
            route_id_emb = self.road_seg_emb(route_id).reshape(B, -1)
            route_id_emb = self.road_seg_emb_proj(route_id_emb)

        space_state = space_state.reshape(B, -1)
        state = torch.cat([space_state, time_state], dim=-1)
        state = self.state_encoder(state)

        if route_id is not None:
            out = torch.cat([route, state, route_id_emb], dim=-1)
        else:
            out = torch.cat([route, state], dim=-1)

        for block in self.blocks:
            out = block(out)

        # out = self.sigmoid(out)   # move the sigmoid from model to loss function nn.BCEWithlogitsLoss
        return out   

class MLPModel(nn.Module):
    def __init__(
            self, 
            input_dim: int = 5,
            hidden_dims: list = [], 
            output_dim: int = 1,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        dims = [input_dim] + hidden_dims
        self.blocks = nn.ModuleList([Block(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])]
                                    + [Block(hidden_dims[-1], output_dim, activation=False)])

    def forward(self, x):
        '''
        input:
            x: (batch_size, input_dim)
        '''
        for block in self.blocks:
            x = block(x)
        return x
    
def main_test(test_program: str):
    if test_program == 'mlp':
        B = 128
        input_dim = 5
        hidden_dims = [64, 64, 32, 16]
        output_dim = 1
        model = MLPModel(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
        print(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total params: ", total_params)
        x = torch.randn(B, input_dim)
        out = model(x)
        print(out.shape)
        print(out.squeeze())
        print(nn.functional.sigmoid(out).squeeze())
    else:
        B = 32
        window_size = 2
        route_dim = 1
        space_dim = 1
        time_dim = 1
        route_hidden = 16
        state_hidden = 8
        block_dims = [512, 256, 128, 64, 32, 16, 1]
        vocab_size = 27910
        route = torch.randn(B, window_size, route_dim)
        route_id = torch.randint(0, vocab_size, (B, window_size))
        space_state = torch.randn(B, window_size, space_dim)
        time_state = torch.randn(B, time_dim)
        model = TimePredModel(route_dim=route_dim, space_dim=space_dim, time_dim=time_dim, route_hidden=route_hidden, state_hidden=state_hidden, window_size=window_size, block_dims=block_dims,vocab_size=vocab_size)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(total_params)
        out = model(route, space_state, time_state, route_id)
        print(out.shape)
        print(out)


if __name__ == "__main__":
    main_test('mlp')