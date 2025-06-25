import torch
from networks.gaussian_policy import GaussianMLP
from networks.action_value import QFunction
import copy


class SAC:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replayy_buffer = replay_buffer
        self.obs_dim = self.env.observation_spec().shape
        self.action_dim = self.env.action_spec().shape
        self.policy = GaussianMLP(self.obs_dim, self.action_dim)
        self.qfunctionA = QFunction(self.obs_dim, self.action_dim, 1)
        self.qfunctionAtarget = copy.deepcopy(self.qfunctionA) 
        self.qfunctionB = QFunction(self.obs_dim, self.action_dim, 1)
        self.qfunctionBtarget = copy.deepcopy(self.qfunctionB) 
        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=0.001)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = 3e-4)
        self.qfunctionA_optimzer = torch.optim.Adam(self.qfunctionA.parameters(), lr = 0.01)
        self.qfunctionB_optimzer = torch.optim.Adam(self.qfunctionB.parameters(), lr = 0.01)
        
    def train(self, episodes, max_steps):

        




