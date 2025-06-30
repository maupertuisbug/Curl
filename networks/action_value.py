import torch

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class QFunction(torch.nn.Module):

    def __init__(self, obs_dim, action_dim, output_dim):
        super().__init__()
        self.obs_dim = obs_dim 
        self.action_dim = action_dim 
        self.output_dim = output_dim
        self.net = torch.nn.Sequential(
                    torch.nn.Linear(self.obs_dim+self.action_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, self.output_dim)
        )
        self.net.apply(init_weights)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = self.net(x)
        return x