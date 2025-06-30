import torch 
import random
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class ImgEncoder(torch.nn.Mpdule):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(64*64, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.encoder(x)
        return logits

class BilinearProd(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.W.requires_grad = True

    def forward(self, x):
        # x to be a column vector ([1, latent_dim])
        return torch.matmul(self.W, x)



seed = 42
random.seed(seed)
torch.manual_seed(seed)

tf = transforms.RandomResizedCrop([84,84], scale=(0.8, 0.84), ratio=(0.85, 0.89))


keyEncoder = ImgEncoder()
queryEncoder = ImgEncoder()
queryEncoderOptim = torch.optim.Adam(queryEncoder.parameters(), lr = 0.002)
keyEncoderOptim = torch.optim.Adam(keyEncoder.parameters(), lr = 0.002)
blp          = BilinearProd(latent_dim=10)
blpOptim     = torch.optim.Adam(blp.parameters(), lr = 0.001)

for img in []:
    query_encoded = tf(img)
    key_encoded   = tf(img)
    query_encoded = queryEncoder(query_encoded)
    key_encoded   = keyEncoder(key_encoded)
    key_encoded   = key_encoded.detach() # no gradient prob 
    proj_key      = blp(key_encoded)
    logits        = torch.matmul(query_encoded, proj_key)
    logits        = logits - max(logits, axis=1)
    labels        = arange(logits.shape[0])
    loss = torch.nn.functional.cross_entropy(logits, labels)
    queryEncoderOptim.zero_grad()
    blpOptim.zero_grad()
    loss.backward()
    queryEncoderOptim.step()
    blpOptim.step()


