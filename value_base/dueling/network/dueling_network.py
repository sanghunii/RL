import torch 
import torch.nn as nn

class DuelingQNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.in_channels = 4
        self.n_actions = n_actions
        dim_head = 128
        
        self.shared = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten() 
        )
        
        cnn_out_dim = 64 * 7 * 7

        self.value_head = nn.Sequential(
            nn.Linear(cnn_out_dim, dim_head),
            nn.ReLU(),

            nn.Linear(dim_head, dim_head//2),
            nn.ReLU(),

            nn.Linear(dim_head//2, 1)
        )

        self.adv_head = nn.Sequential(
            nn.Linear(cnn_out_dim, dim_head),
            nn.ReLU(),

            nn.Linear(dim_head, dim_head//2),
            nn.ReLU(),

            nn.Linear(dim_head//2, n_actions)
        )

    def forward(self, state):
        shared = self.shared(state)
        value = self.value_head(shared)
        advs = self.adv_head(shared)
        return value, advs

    def get_q_values(self, state):
        if state.dtype == torch.uint8:
            state = state.float() / 255.0
            
        value, advs = self.forward(state)
        advs_mean = advs.mean(dim=-1, keepdim=True)
        q = value + (advs - advs_mean)
        return q