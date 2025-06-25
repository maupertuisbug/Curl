import torch
from networks.gaussian_policy import GaussianMLP
from networks.action_value import QFunction
import copy
import torch.functional as F


class SAC:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer
        self.obs_dim = self.env.observation_spec().shape
        self.action_dim = self.env.action_spec().shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = GaussianMLP(self.obs_dim, self.action_dim).to(self.device)
        self.qfunctionA = QFunction(self.obs_dim, self.action_dim, 1).to(self.device)
        self.qfunctionAtarget = copy.deepcopy(self.qfunctionA).to(self.device) 
        self.qfunctionB = QFunction(self.obs_dim, self.action_dim, 1).to(self.device)
        self.qfunctionBtarget = copy.deepcopy(self.qfunctionB).to(self.device)
        self.log_alpha = torch.tensor(0.0, requires_grad=True).to(self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=0.001)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = 3e-4)
        self.qfunctionA_optimzer = torch.optim.Adam(self.qfunctionA.parameters(), lr = 0.01)
        self.qfunctionB_optimzer = torch.optim.Adam(self.qfunctionB.parameters(), lr = 0.01)
        self.tau = 0.001
        self.gamma = 0.99

  
    def train(self, episodes, max_steps):
        batch_size = 256

        for ep in range(episodes):
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            total_reward = 0

            for _ in range(max_steps):
                with torch.no_grad():
                    action, _, _ = self.policy.sample(state.unsqueeze(0))  # [1, action_dim]
                    action = action.squeeze(0).cpu().numpy()
                next_state, reward, done, _ = self.env.step(action)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)

                self.replay_buffer.push(state.cpu().numpy(), action, reward, next_state, done)
                state = next_state_tensor
                total_reward += reward

                # Start updates after buffer fills
                if len(self.replay_buffer) < batch_size:
                    continue

                # Sample a batch
                s, a, r, s_next, d = self.replay_buffer.sample(batch_size)

                s = torch.tensor(s, dtype=torch.float32).to(self.device)
                a = torch.tensor(a, dtype=torch.float32).to(self.device)
                r = torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self.device)
                s_next = torch.tensor(s_next, dtype=torch.float32).to(self.device)
                d = torch.tensor(d, dtype=torch.float32).unsqueeze(1).to(self.device)

                # Sample actions from current policy at next state
                with torch.no_grad():
                    next_action, log_prob_next_action, _ = self.policy.sample(s_next)
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
                action_new, log_prob_new, _ = self.policy.sample(s)
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

            print(f"Episode {ep+1}/{episodes} - Total Reward: {total_reward:.2f}")









