import torch
from networks.gaussian_policy import GaussianMLP, GaussianMLPImg
from networks.action_value import QFunction, QFunctionImg
import copy
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from tensordict import TensorDict

def flatten_observation(obs_dict):

    return np.concatenate([v.ravel() for v in obs_dict.values()])


def valid(l1, l2, l3, l4, size):
    return len(l1) == len(l2) == len(l3) == len(l4) == size 


class SAC:
    def __init__(self, env, replay_buffer, latent_dim, wandb_run, encoder,train_with_curl):
        self.env = env
        self.replay_buffer = replay_buffer.experience
        self.wandb_run = wandb_run
        self.action_dim = self.env.action_spec().shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_with_repr = train_with_curl
        if self.train_with_repr:
            self.obs_dim = latent_dim 
            self.encoder = encoder 
            self.policy = GaussianMLP(self.obs_dim, self.action_dim).to(self.device)
            self.qfunctionA = QFunction(self.obs_dim, self.action_dim).to(self.device)
            self.qfunctionAtarget = copy.deepcopy(self.qfunctionA).to(self.device)
            self.qfunctionB = QFunction(self.obs_dim, self.action_dim).to(self.device)
            self.qfunctionBtarget = copy.deepcopy(self.qfunctionB).to(self.device)
        else :
            in_ch = 3
            self.obs_dim = latent_dim
            self.encoder = encoder
            self.policy = GaussianMLPImg(input_channels = in_ch, output_dim = self.action_dim).to(self.device)
            self.qfunctionA = QFunctionImg(input_channels = in_ch, action_dim = self.action_dim).to(self.device)
            self.qfunctionAtarget = copy.deepcopy(self.qfunctionA).to(self.device)
            self.qfunctionB = QFunctionImg(input_channels = in_ch, action_dim = self.action_dim).to(self.device)
            self.qfunctionBtarget = copy.deepcopy(self.qfunctionB).to(self.device)
        
        self.log_alpha = torch.tensor(0.0, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=0.001)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = 3e-4)
        self.qfunctionA_optimizer = torch.optim.Adam(self.qfunctionA.parameters(), lr = 0.01)
        self.qfunctionB_optimizer = torch.optim.Adam(self.qfunctionB.parameters(), lr = 0.01)
        self.target_entropy = -np.prod(self.env.action_spec().shape).item()
        self.tau = 0.001
        self.gamma = 0.99


  
    def train(self, episodes, max_steps):
        batch_size = 32
        rewardsl = []
        tf = transforms.RandomResizedCrop([84,84], scale=(0.8, 0.84), ratio=(0.85, 0.89))
        if self.train_with_repr:
            frame_skip = 3
        else :
            frame_skip  = 1

        for ep in range(episodes):
            
            state = self.env.reset()
            obs_img = np.ascontiguousarray(self.env.physics.render(camera_id = 0, height=100, width = 100))
            
            total_reward = 0
            obs_img_list = []

            for _ in range(frame_skip) :
                obs_img_list.append(torch.tensor(obs_img, device = self.device))
            
            if frame_skip > 1:
                with torch.no_grad():
                    encoded_img = self.encoder.preprocess(torch.stack(obs_img_list).unsqueeze(0))
                    encoded_img = tf(encoded_img).to(self.device)
                    encoded_img = self.encoder.encode(encoded_img)
                    action, _, _  = self.policy(encoded_img)
                    action = action.squeeze(0).cpu().numpy()
            else :
                with torch.no_grad():
                    action, _, _ = self.policy(torch.stack(obs_img_list).permute(0, 3, 1, 2))
                    action = action.squeeze(0).cpu().numpy()

            
            obs_img_list = [] 
            next_obs_img_list = []
            obs_list = []
            next_obs_list = []
            done = False
            for sp in range(max_steps-1):
                if self.train_with_repr:
                    self.encoder.train_repr(1,32)
                reward = 0
                for _ in range(frame_skip):
                    with torch.no_grad():
                        obs_img_list.append(torch.tensor(obs_img))
                        obs = flatten_observation(state.observation)
                        obs_list.append(torch.tensor(obs))

                        state = self.env.step(action)

                        next_obs = flatten_observation(state.observation)
                        next_obs_list.append(torch.tensor(next_obs))
                        next_obs_img =np.ascontiguousarray(self.env.physics.render(camera_id = 0, height=100, width = 100))
                        next_obs_img_list.append(torch.tensor(next_obs_img))

                        reward = reward + (state.reward or 0.0)
                        obs_img = next_obs_img
                        done = state.last()
                
                if frame_skip > 1 :

                    if valid(obs_list, obs_img_list, next_obs_list, next_obs_img_list, frame_skip):
                        transition = TensorDict({
                            "obs" : torch.stack(obs_list).to("cpu"),
                            "obs_img" : torch.stack(obs_img_list).to("cpu"),
                            "action" : torch.tensor(action, device="cpu"),
                            "next_obs" : torch.stack(next_obs_list).to("cpu"),
                            "next_obs_img" : torch.stack(next_obs_img_list).to("cpu"),
                            "reward" : torch.tensor(reward, device="cpu"),
                            "done"  : torch.tensor(int(state.last()), device="cpu")
                        }, batch_size=[])
                        self.replay_buffer.add(transition)
                        with torch.no_grad():
                            encoded_img = self.encoder.preprocess(torch.stack(obs_img_list).unsqueeze(0))
                            encoded_img = tf(encoded_img).to(self.device)
                            encoded_img = self.encoder.encode(encoded_img)
                            action, _, _  = self.policy(encoded_img)
                            action = action.squeeze(0).cpu().numpy()

                    
                else :
                    if valid(obs_list, obs_img_list, next_obs_list, next_obs_img_list, frame_skip):
                        transition = TensorDict({
                            "obs" : torch.stack(obs_list).to("cpu").squeeze(0),
                            "obs_img" : torch.stack(obs_img_list).to("cpu").squeeze(0),
                            "action" : torch.tensor(action, device="cpu"),
                            "next_obs" : torch.stack(next_obs_list).to("cpu").squeeze(0),
                            "next_obs_img" : torch.stack(next_obs_img_list).to("cpu").squeeze(0),
                            "reward" : torch.tensor(reward, device="cpu"),
                            "done"  : torch.tensor(int(state.last()), device="cpu")
                        },batch_size=[])
                        self.replay_buffer.add(transition)
                        with torch.no_grad():
                            action, _, _ = self.policy(torch.stack(obs_img_list).to(self.device).permute(0, 3, 1, 2))
                            action = action.squeeze(0).cpu().numpy()


                obs_list = []
                next_obs_list = []
                obs_img_list = [] 
                next_obs_img_list = [] 
                total_reward += reward
                # Start updates after buffer fills
                if len(self.replay_buffer) < batch_size:
                    continue
                
                with torch.no_grad():
                    # Sample a batch
                    batch = self.replay_buffer.sample(batch_size)
                    states = batch["obs"].to(self.device)
                    states_img  = batch["obs_img"].to(self.device)
                    actions = batch["action"].to(self.device)
                    rewards = batch["reward"].to(self.device)
                    next_states = batch["next_obs"].to(self.device)
                    next_states_img = batch["next_obs_img"].to(self.device)
                    dones = batch["done"].to(self.device)

                    if self.train_with_repr:
                        s = torch.tensor(states_img, dtype=torch.float64, device = self.device)
                        s = self.encoder.preprocess(s)
                        s = tf(s).to(self.device)
                        s = self.encoder.encode(s)

                        s_next = torch.tensor(next_states_img, dtype = torch.float64, device = self.device)
                        s_next = self.encoder.preprocess(s_next)
                        s_next = tf(s_next).to(self.device)
                        s_next = self.encoder.encode(s_next)
                    else :
                        s = torch.tensor(states_img.permute(0, 3, 1, 2), dtype=torch.float64, device = self.device)
                        s_next = torch.tensor(next_states_img.permute(0, 3, 1, 2), dtype=torch.float64, device = self.device)
                    
                    a = torch.tensor(actions, dtype=torch.float32).to(self.device)
                    r = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
                    d = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

                    next_action, log_prob_next_action, _ = self.policy(s_next)
                    q1_next = self.qfunctionAtarget(s_next, next_action)
                    q2_next = self.qfunctionBtarget(s_next, next_action)
                    min_q_next = torch.min(q1_next, q2_next)
                    alpha = self.log_alpha.exp()
                    target_q = r + self.gamma * (1 - d) * (min_q_next - alpha * log_prob_next_action)

                # Q-function A update
                q1 = self.qfunctionA(s, a)
                q1_loss = F.mse_loss(q1, target_q.detach())
                self.qfunctionA_optimizer.zero_grad()
                q1_loss.backward(retain_graph=True)
                self.qfunctionA_optimizer.step()

                # Q-function B update
                q2 = self.qfunctionB(s, a)
                q2_loss = F.mse_loss(q2, target_q.detach())
                self.qfunctionB_optimizer.zero_grad()
                q2_loss.backward(retain_graph=True)
                self.qfunctionB_optimizer.step()

                # Policy update
                action_new, log_prob_new, _ = self.policy(s)
                q1_pi = self.qfunctionA(s, action_new)
                q2_pi = self.qfunctionB(s, action_new)
                min_q_pi = torch.min(q1_pi, q2_pi)
                policy_loss = (alpha * log_prob_new - min_q_pi).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                self.policy_optimizer.step()

                # Alpha (entropy) update
                alpha_loss = -(self.log_alpha * (log_prob_new + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward(retain_graph=True)
                self.alpha_optimizer.step()

                # Update target networks
                if sp%100 == 0:
                    with torch.no_grad():
                        for param, target_param in zip(self.qfunctionA.parameters(), self.qfunctionAtarget.parameters()):
                            target_param.data.mul_(1 - self.tau)
                            target_param.data.add_(self.tau * param.data)

                        for param, target_param in zip(self.qfunctionB.parameters(), self.qfunctionBtarget.parameters()):
                            target_param.data.mul_(1 - self.tau)
                            target_param.data.add_(self.tau * param.data)

                if done:
                    break 
            
            rewardsl.append(total_reward)  
            self.wandb_run.log({'average reward' : np.mean(rewardsl)}, step = ep)









