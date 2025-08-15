import torch 
import numpy as np
from torchrl.data import ReplayBuffer, ListStorage, TensorDictReplayBuffer
from tensordict import TensorDict

torch.manual_seed(0)

def flatten_observation(obs_dict):

    return np.concatenate([v.ravel() for v in obs_dict.values()])


class RB():
    def __init__(self, max_size, batch_size, wandb_run):
        self.experience = TensorDictReplayBuffer(
            storage=ListStorage(max_size=max_size), batch_size = batch_size)
        self.wandb_run = wandb_run
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collect_init(self, env, episodes, max_steps, train_with_curl):
        
        action_spec = env.action_spec()
        obs_spec = env.observation_spec()
        if train_with_curl :
            frame_skip = 3
        else :
            frame_skip = 1
        for ep in range(episodes):
            rewards = []
            state = env.reset()
            obs_img =  np.ascontiguousarray(env.physics.render(camera_id=0, height=100, width=100))
            obs_list = []
            next_obs_list = []
            obs_img_list = [] 
            next_obs_img_list = []
            for step in range(max_steps):
                action = np.random.uniform(
                        low = action_spec.minimum,
                        high = action_spec.maximum,
                        size = action_spec.shape,
                    )
                reward = 0
                for frame in range(0, frame_skip):
                    obs_img_list.append(torch.tensor(obs_img))
                    obs = flatten_observation(state.observation)
                    obs_list.append(torch.tensor(obs))
                    
                    state = env.step(action)
                    
                    next_obs = flatten_observation(state.observation)
                    next_obs_list.append(torch.tensor(next_obs))
                    next_obs_img = np.ascontiguousarray(env.physics.render(camera_id=0, height=100, width=100))
                    next_obs_img_list.append(torch.tensor(next_obs_img))
                    reward = reward + (state.reward or 0.0)
                    
                    obs_img = next_obs_img
                if frame_skip > 1 : 
                    if len(obs_list) == frame_skip and len(next_obs_list) == frame_skip and len(obs_img_list) == frame_skip and len(next_obs_img_list) == frame_skip:
                        transition = TensorDict({
                            "obs" : torch.stack(obs_list).to("cpu"),
                            "obs_img" : torch.stack(obs_img_list).to("cpu"),
                            "action" : torch.tensor(action).to("cpu"),
                            "next_obs" : torch.stack(next_obs_list).to("cpu"),
                            "next_obs_img" : torch.stack(next_obs_img_list).to("cpu"),
                            "reward" : torch.tensor(reward, device="cpu"),
                            "done"  : torch.tensor(int(state.last()),device="cpu")
                        }, batch_size=[])
                    
                        self.experience.add(transition)
                        rewards.append(reward)
                else :
                    if len(obs_list) == frame_skip and len(next_obs_list) == frame_skip and len(obs_img_list) == frame_skip and len(next_obs_img_list) == frame_skip:
                        transition = TensorDict({
                            "obs" : torch.stack(obs_list).to("cpu").squeeze(0),
                            "obs_img" : torch.stack(obs_img_list).to("cpu").squeeze(0),
                            "action" : torch.tensor(action, device="cpu"),
                            "next_obs" : torch.stack(next_obs_list).to("cpu").squeeze(0),
                            "next_obs_img" : torch.stack(next_obs_img_list).to("cpu").squeeze(0),
                            "reward" : torch.tensor(reward, device="cpu"),
                            "done"  : torch.tensor(int(state.last()), device="cpu")
                        }, batch_size=[])
                    
                        self.experience.add(transition)
                        rewards.append(reward)

                obs_list = []
                next_obs_list = []
                obs_img_list = [] 
                next_obs_img_list = [] 
            # self.wandb_run.log({'Init Data' : np.mean(rewards)}, step = ep)


    def sample(self, batch_size):
        batch = self.experience.sample(batch_size)
        return batch
            


    

