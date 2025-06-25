import torch 
import numpy as np
from torchrl.data import ReplayBuffer, ListStorage

torch.manual_seed(0)

def flatten_observation(obs_dict):

    return np.concatenate([v.ravel() for v in obs_dict.values()])


class RB():
    def __init__(self, max_size, batch_size, wandb_run):
        self.experience = ReplayBuffer(
            storage=ListStorage(max_size=max_size), batch_size = batch_size)
        self.wandb_run = wandb_run

    def collect_init(self, env, episodes, max_steps):
        
        action_spec = env.action_spec()
        obs_spec = env.observation_spec()
        for ep in range(episodes):
            rewards = []
            state = env.reset()
            for step in range(max_steps):
                obs = flatten_observation(state.observation)
                action = np.random.uniform(
                    low = action_spec.minimum,
                    high = action_spec.maximum,
                    size = action_spec.shape,
                )
                state = env.step(action)
                reward = state.reward or 0.0
                transition = {
                    "obs" : torch.tensor(obs),
                    "action" : torch.tensor(action),
                    "next_obs" : torch.tensor(flatten_observation(state.observation)),
                    "reward" : torch.tensor(reward)
                }
                self.experience.add(transition)
                rewards.append(reward)
            self.wandb_run.log({'average reward' : np.mean(rewards)}, step = ep)
            


    

