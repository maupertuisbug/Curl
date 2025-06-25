import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np 
from dm_control import suite 
from collect_data import RB
import wandb





domain_name = "reacher"
task_name   = "easy"

env = suite.load(domain_name=domain_name, task_name=task_name)

wandb_run = wandb.init(project="CURL")

rb = RB(1000, 32, wandb_run)
rb.collect_init(env, 500, 200)