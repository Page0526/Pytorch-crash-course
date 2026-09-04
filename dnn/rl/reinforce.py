import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

class Reinforce(nn.Module):
    def __init__(self, inputs, outputs, hidden_state = 128, lr=1e-2, gamma=0.99):
        super().__init__()
        self.lr = lr
        self.gamma = gamma

        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_state),
            nn.ReLU(),
            nn.Linear(hidden_state, outputs)
        )

    def forward(self, x):
        return self.model(x)
    
    def act(self, state):
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        return action.item(), dist.log_prob(action)

def train(env, episodes):
    policy = Reinforce(env.observation_space.n, env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=policy.lr)

    for episode in range(episodes):
        ob = env.reset() # state, prob
        
        log_probs = []
        rewards = []
        actions = []

        done = False
        while not done:

            action, log_prob = policy.act(ob)

            next_ob, reward, done, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_ob
        
        returns = []
        Gt = 0
        for r in reversed(rewards):
            Gt = r +  policy.gamma*Gt
            returns.insert(0, Gt)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize

        loss = -torch.sum(torch.stack(log_probs) * returns)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Episode {episode}, Reward: {sum(rewards)}")
        

if __name__ == "__main__":
    # Create the environment
    env=gym.make('FrozenLake-v1',render_mode='human')
    env.reset()
    env.render()
    train(env, episodes=1000)
    # env.observation_space, env.action_space
    env.close()
    