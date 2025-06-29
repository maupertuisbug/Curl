import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np 
from dm_control import suite 
from collect_data import RB
import wandb
from sac import SAC

class DMCWrapper:
    def __init__(self, env):
        self.env = env
        self.ts = None
        self._action_spec = env.action_spec()
        self._obs_spec = env.observation_spec()

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
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def _flatten_obs(self, obs_dict):
        return np.concatenate([np.ravel(v) for v in obs_dict.values()])



domain_name = "reacher"
task_name   = "easy"

env = suite.load(domain_name=domain_name, task_name=task_name)

wandb_run = wandb.init(project="CURL")

rb = RB(100000, 32, wandb_run)
rb.collect_init(env, 500, 200)
envw = DMCWrapper(env)
agent = SAC(envw, rb)
agent.train(episodes=100, max_steps=1000)
