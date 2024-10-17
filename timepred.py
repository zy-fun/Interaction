import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

        if input_dim != output_dim:
            self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        identity = x  

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        if x.size(-1) != out.size(-1):
            identity = self.fc(identity)
        out += identity
        out = self.relu(out)
        return out 

# class RouteEncoder(nn.Module):
#     def __init__(self, 
#                  input_dim: int = 3,
#                  hidden_dim: int = 16,):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.relu2 = nn.ReLU()

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         return out

class TimePredModel(nn.Module):
    def __init__(
            self, 
            route_dim: int = 3, 
            space_dim: int = 1, 
            time_dim: int = 1, 
            route_hidden: int = 16,
            state_hidden: int = 16,
            window_size: int = 3, 
            block_dims: list = [], 
        ):
        super().__init__()
        self.route_encoder = Block(route_dim * window_size, route_hidden)
        self.state_encoder = Block(space_dim * window_size + time_dim, state_hidden)
        block_dims = [route_hidden + state_hidden] + block_dims
        block_dims = zip(block_dims[:-1], block_dims[1:])
        self.blocks = nn.ModuleList([Block(in_dim, out_dim) for in_dim, out_dim in block_dims])
        self.sigmoid = nn.Sigmoid() 

    def forward(self, route, space_state, time_state):
        '''
        input:
            route: (batch_size, window_size, route_dim)
            space_state: (batch_size, window_size, space_dim)
            time_state: (batch_size, time_dim)
        '''
        B, W, _ = route.size()

        route = route.reshape(B, -1)
        route = self.route_encoder(route)

        space_state = space_state.reshape(B, -1)
        state = torch.cat([space_state, time_state], dim=-1)
        state = self.state_encoder(state)

        out = torch.cat([route, state], dim=-1)
        for block in self.blocks:
            out = block(out)

        out = self.sigmoid(out)
        return out   

if __name__ == "__main__":
    B = 32
    window_size = 3
    route_dim = 3
    space_dim = 1
    time_dim = 1
    route_hidden = 16
    state_hidden = 8
    block_dims = [512, 256, 128, 64, 32, 16, 1]
    route = torch.randn(B, window_size, route_dim)
    space_state = torch.randn(B, window_size, space_dim)
    time_state = torch.randn(B, time_dim)
    model = TimePredModel(route_dim=route_dim, space_dim=space_dim, time_dim=time_dim, route_hidden=route_hidden, state_hidden=state_hidden, window_size=window_size, block_dims=block_dims)
    out = model(route, space_state, time_state)
    print(out.shape)
    print(out[0])