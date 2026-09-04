import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import colorize

class Policy:
    def __init__(self, env, name, training=True, gamma = 0.99, deterministic=False):
        self.env = env
        self.name = name
        self.training = training
        self.gamma = gamma

        if deterministic:
            np.random.seed(1)
            torch.manual_seed(1)

    @property
    def act_size(self):
        # discrete space
        if isinstance(self.env.action_space, Discrete):
            return self.env.action_space.n
        # continuous space
        elif isinstance(self.env.action_space, Box):
            return list(self.env.action_space.shape)
    
    @property
    def state_dim(self):
        return self.env.observation_space.shape
    
    def obs_to_inputs(self, ob):
        return ob.flatten()
    
    def act(self, state, **kwargs):
        pass

    def build(self):
        pass

    def train(self, *args, **kwargs):
        pass

    def evaluate(self, n_episodes):
        reward_history = []
        reward = 0.

        for i in range(n_episodes):
            ob = self.env.reset()
            done = False
            while not done:
                a = self.act(ob)
                new_ob, r, done, _ = self.env.step(a)
                self.env.render()
                reward += r
                ob = new_ob

            reward_history.append(reward)
            reward = 0.

        print("Avg. reward over {} episodes: {:.4f}".format(n_episodes, np.mean(reward_history)))




        
    
        


