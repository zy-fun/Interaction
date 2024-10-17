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

class TimePredModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        pass

    def forward(self, loc_state, traffic_state, time_state):
        '''
        input:
            loc_state: (batch_size, seq_len, loc_dim)
            traffic_state: (batch_size, seq_len, traffic_dim)
            time_state: (batch_size, seq_len, time_dim)
        '''
        x = torch.cat([loc_state, traffic_state, time_state], dim=-1)   # (batch_size, seq_len, feature_in)
        B, T, C = x.size()

        x = x.reshape(B, T * C)
        x = self.fc1(x)
        pass

if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    block = Block(4, 5)
    out = block(x)
    print(out.shape)
