import torch
from networks.gaussian_policy import GaussianMLP
from networks.action_value import QFunction
import copy
import torch.nn.functional as F
import numpy as np


class SAC:
    def __init__(self, env, replay_buffer, latent_dim, wandb_run):
        self.env = env
        self.replay_buffer = replay_buffer.experience
        self.wandb_run = wandb_run
        self.action_dim = self.env.action_spec().shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_with_repr = True
        self.obs_dim = sum(np.prod(spec.shape) for spec in self.env.observation_spec().values())
        if self.train_with_repr:
            self.obs_dim = latent_dim
        else :
            self.obs_dim = self.obs_dim
        self.policy = GaussianMLP(self.obs_dim, self.action_dim).to(self.device)
        self.qfunctionA = QFunction(self.obs_dim, self.action_dim, 1).to(self.device)
        self.qfunctionAtarget = copy.deepcopy(self.qfunctionA).to(self.device)
        self.qfunctionB = QFunction(self.obs_dim, self.action_dim, 1).to(self.device)
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

        for ep in range(episodes):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            

            total_reward = 0

            for _ in range(max_steps):
                with torch.no_grad():
                    action, _, _ = self.policy(state.unsqueeze(0))  # [1, action_dim]
                    action = action.squeeze(0).cpu().numpy()
                next_state, reward, done, _ = self.env.step(action)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                transition = {
                    "obs" : torch.tensor(state, device=self.device),
                    "action" : torch.tensor(action, device=self.device),
                    "next_obs" : torch.tensor(next_state_tensor, device=self.device),
                    "reward" : torch.tensor(reward, device=self.device), 
                    "done" : torch.tensor(int(done), device=self.device)

                }
                self.replay_buffer.add(transition)
                state = next_state_tensor
                total_reward += reward

                # Start updates after buffer fills
                if len(self.replay_buffer) < batch_size:
                    continue

                # Sample a batch
                batch = self.replay_buffer.sample(batch_size)
                states = batch["obs"]
                states_img  = batch["obs_img"]
                actions = batch["action"]
                rewards = batch["reward"]
                next_states = batch["next_obs"]
                next_states_img = batch["next_obs_img"]
                dones = batch["done"]

                if self.train_with_repr:
                    s = torch.tensor(states, dtype=torch.float32).to(self.device)
                    s_next = torch.tensor(next_states, dtype=torch.float32).to(self.device)
                else :
                    s = torch.tensor(states_img, dtype=torch.float64, device = self.device)
                    s_next = torch.tensor(next_states_img, dtype=torch.float64, device = self.device)
                a = torch.tensor(actions, dtype=torch.float32).to(self.device)
                r = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
                d = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

                # Sample actions from current policy at next state
                with torch.no_grad():
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
                q1_loss.backward()
                self.qfunctionA_optimizer.step()

                # Q-function B update
                q2 = self.qfunctionB(s, a)
                q2_loss = F.mse_loss(q2, target_q.detach())
                self.qfunctionB_optimizer.zero_grad()
                q2_loss.backward()
                self.qfunctionB_optimizer.step()

                # Policy update
                action_new, log_prob_new, _ = self.policy(s)
                q1_pi = self.qfunctionA(s, action_new)
                q2_pi = self.qfunctionB(s, action_new)
                min_q_pi = torch.min(q1_pi, q2_pi)
                policy_loss = (alpha * log_prob_new - min_q_pi).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Alpha (entropy) update
                alpha_loss = -(self.log_alpha * (log_prob_new + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # Update target networks
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









