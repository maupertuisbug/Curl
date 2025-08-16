import torch 
import random
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from collect_data import RB
import numpy as np

# lets define a single image as [1, 3, 84, 84, ]
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)




class ImgEncoder(torch.nn.Module):
    def __init__(self, latent_dim = 50):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=9, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size= 3, stride=1),
            torch.nn.ReLU(),
        )
        self.output_dim = self.get_output_size((9, 84, 84))    # stack of 3 images
        self.linear = torch.nn.Linear(self.output_dim, latent_dim)
        self.encoder.apply(init_weights)
        self.linear.apply(init_weights)

    def forward(self, x):
        x = torch.flatten(self.encoder(x.float() / 255.0), start_dim=1)
        x = self.linear(x)
        x = torch.nn.functional.relu(x)
        return x

    def get_output_size(self, shape):
        x_out = self.encoder(torch.zeros(1, *shape))
        x_out = torch.tensor(x_out.shape[1:])
        return int(torch.prod(x_out))

class BilinearProd(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.W.requires_grad = True

    def forward(self, x):
        # x to be a column vector ([1, latent_dim])
        return torch.transpose(torch.matmul(x, self.W), 0, 1)


class CURLWrapper:
    def __init__(self, replay_buffer, wandb_run, stack_size=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.storage = replay_buffer.experience
        self.stack_size = stack_size
        self.keyEncoder = ImgEncoder(latent_dim = 50).to(self.device)
        self.queryEncoder = ImgEncoder(latent_dim = 50).to(self.device)
        self.queryEncoderOptim = torch.optim.Adam(self.queryEncoder.parameters(), lr = 0.002)
        self.keyEncoderOptim = torch.optim.Adam(self.keyEncoder.parameters(), lr = 0.002)
        self.blp  = BilinearProd(latent_dim=50).to(self.device)
        self.blpOptim  = torch.optim.Adam(self.blp.parameters(), lr = 0.001)
        self.logger = wandb_run

    def preprocess_batch(self, batch):    # a batch of dataset, 
        obs = batch['obs_img']
        b, s, h, w, c, = obs.shape
        obs = obs.permute(0, 1, 4, 2, 3)
        obs = obs.reshape(b, s*c, h, w)
        return obs
    
    def preprocess(self, obs):
        b, s, h, w, c, = obs.shape
        obs = obs.permute(0, 1, 4, 2, 3)
        obs = obs.reshape(b, s*c, h, w)
        return obs

    def encode(self,  x):
        return self.queryEncoder(x)

    def train_repr(self, epochs, batch_size):
        tf = transforms.RandomResizedCrop([84,84], scale=(0.8, 0.84), ratio=(0.85, 0.89))
        rs_one = 65
        rs_two = 89
        lossl = []
        for epoch in range(0, epochs):
            obs = self.storage.sample(batch_size)
            obs = self.preprocess_batch(obs)
            obs = obs.to(self.device)
            torch.manual_seed(rs_one)
            query_encoded = tf(obs)
            torch.manual_seed(rs_two)
            key_encoded = tf(obs)
            query_encoded = self.queryEncoder(query_encoded)
            key_encoded   = self.keyEncoder(key_encoded)
            key_encoded   = key_encoded.detach() # no gradient prob 
            proj_key      = self.blp(key_encoded)
            logits        = torch.matmul(query_encoded, proj_key)
            logits        = logits - logits.max(dim=1, keepdim=True).values
            labels        =  torch.arange(logits.shape[0]).to(self.device).long()
            loss = torch.nn.functional.cross_entropy(logits, labels)
            self.queryEncoderOptim.zero_grad()
            self.blpOptim.zero_grad()
            loss.backward()
            self.queryEncoderOptim.step()
            self.blpOptim.step()
            m = 0.99
            lossl.append(loss.item())
            for param_k, param_q in zip(self.keyEncoder.parameters(), self.queryEncoder.parameters()):
                param_k.data = m * param_k.data + (1 - m) * param_q.data
        
        self.logger.log({'loss' : np.mean(lossl)})











