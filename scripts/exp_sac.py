import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np 
from dm_control import suite 
from collect_data import RB
import wandb
from action_spaces.continousAgent import SAC
import torch 
import torch.nn as nn 
import wandb 
import argparse
from omegaconf import OmegaConf
import pickle 
import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt

class DMCWrapper:
    def __init__(self, env):
        self.env = env
        self.ts = None

    def reset(self):
        self.ts = self.env.reset()
        return self._flatten_obs(self.ts.observation)

    def step(self, action):
        self.ts = self.env.step(action)
        obs = self._flatten_obs(self.ts.observation)
        reward = self.ts.reward or 0.0
        done = self.ts.last()
        return obs, reward, done, {}
    
    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self.env.action_spec()

    def _flatten_obs(self, obs_dict):
        return np.concatenate([np.ravel(v) for v in obs_dict.values()])


def run_exp():
    wandb_run = wandb.init(project="CURL")
    config = wandb.config 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_name = "reacher"
    task_name   = "easy"
    seed = config.seed
    env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})
    pixels = np.ascontiguousarray(env.physics.render(camera_id=0, height=100, width=100))
    pstensor = torch.tensor(pixels, dtype=torch.float64).to(device)
    plt.imshow(pixels)
    plt.savefig('test.png')

    wandb_run = wandb.init(project="CURL")
    rb = RB(1000000, 32, wandb_run)
    rb.collect_init(env, 5, 100)
    data = rb.sample(2)
    ob = data['obs_img']
    envw = DMCWrapper(env)
    agent = SAC(envw, rb, wandb_run)
   # representation_learneer = repr_learner.RepresentationLearner()

    agent.train(episodes=1000, max_steps=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    project_name = "CURL"
    sweep_id = wandb.sweep(sweep=config_dict, project=project_name)
    agent = wandb.agent(sweep_id, function=run_exp, count=10)
