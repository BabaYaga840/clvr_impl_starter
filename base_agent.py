import gym
import sprites_env
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
#from utility.networks import Actor, Critic
from utility import networks
import wandb

environment = 'Sprites-v0'
def ppo_step(policy_net, value_net, states, actions, old_log_probs, returns, advantages, optimizer, ppo_clip_param=0.2):
    dists = policy_net(states)
    new_log_probs = dists.log_prob(actions)
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - ppo_clip_param, 1.0 + ppo_clip_param) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = 0.5 * (returns - value_net(states)).pow(2).mean()
    loss = actor_loss + critic_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train(env, policy_net, value_net, episodes, optimizer):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            dist = policy_net(state)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.numpy())
            # Store state, action, reward, etc., for training
            # Train using ppo_step at appropriate intervals



class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Process the 64x64 observation
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # mean of the action distribution
            nn.Tanh()  # normalize mean between -1 and 1
        )
        self.log_std = nn.Parameter(torch.zeros(1, 2))  # log std parameter

    def forward(self, x):
        mean = self.network(x)
        std = torch.exp(self.log_std.expand_as(mean))  # expand std to match mean's shape
        return Normal(mean, std)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    env = gym.make(environment)

    # Initialize networks and optimizer
    
    policy_net = Actor()
    value_net = Critic()
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=3e-4)

    train(env, policy_net, value_net, 1000, optimizer)    
    # Hyperparameters
    steps_per_epoch = 4000
    epochs = 50
    gamma = 0.99
    lam = 0.95
    ppo_steps = 4
    ppo_clip = 0.2
    wandb.init(
        project="RL_algo_sprites",
        config={
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "ppo_clip": ppo_clip,
        "ppo_steps": ppo_steps,
        "environment": environment
        })

    # Main training loop
    for epoch in range(epochs):
        state = env.reset()
        done = False
