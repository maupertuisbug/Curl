import random
import gym 
from sac import SAC
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


env = gym.make("Pendulum-v1")
replay_buffer = ReplayBuffer()
agent = SAC(env, replay_buffer)
agent.train(episodes=10, max_steps=200)
