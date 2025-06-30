import torch 

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class GaussianMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim  # you have this in the latent space
        self.output_dim = output_dim
        self.net = torch.nn.Sequential(
                    torch.nn.Linear(self.input_dim, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 2 * self.output_dim)
        )
        self.log_min = -10
        self.log_max = 2
        self.net.apply(init_weights)

    def forward(self, obs):
        output = self.net(obs)
        mean = output[:,:self.output_dim]
        log_var = output[:,self.output_dim:]
        log_var = torch.clamp(log_var, self.log_min, self.log_max)
        std = torch.exp(log_var)

        # you need to use reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z)
        log_prob = log_prob.sum(dim = -1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim = -1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)

