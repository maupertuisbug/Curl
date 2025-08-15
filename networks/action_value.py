import torch

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class QFunction(torch.nn.Module):

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim 
        self.action_dim = action_dim 
        self.net = torch.nn.Sequential(
                    torch.nn.Linear(self.obs_dim+self.action_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 1)
        )
        self.net.apply(init_weights)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        x = self.net(x)
        return x
    



class QFunctionImg(torch.nn.Module):

    def __init__(self, input_channels, action_dim):
        super().__init__()
        self.action_dim  = action_dim 
        
        self.conv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels = input_channels, out_channels = 32, kernel_size = 8, stride = 4),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
                    torch.nn.ReLU(), 
                   
        )
        flatten_size = self.get_output_size((input_channels, 100, 100))        
        self.linear = torch.nn.Sequential(
                    torch.nn.Linear(flatten_size+action_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 1))
        
        self.conv.apply(init_weights)
        self.linear.apply(init_weights)

    def forward(self, obs, action):
        x = torch.flatten(self.conv(obs.float()/255.0), start_dim=1)
        x =  torch.cat([x, action], dim = 1)
        return x

    def get_output_size(self, shape):
        x_out = self.conv(torch.zeros(1, *shape))
        x_out = torch.tensor(x_out.shape[1:])
        return int(torch.prod(x_out))
    

    


