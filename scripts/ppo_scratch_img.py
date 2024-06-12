import os
import random
import time
from dataclasses import dataclass
import gym
import sprites_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
import wandb

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    torch_deterministic: bool = True
    track: bool = True
    wandb_project_name: str = "PPO_sprites"
    save_model: bool = True
    device: str = "cpu"

    env_id: str = "Sprites-v0"
    total_timesteps: int = 100000000
    learning_rate: float = 3e-5
    num_steps: int = 40
    num_plays: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 5
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.1
    ent_coef: float = 0.0001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 10)),
            nn.Tanh(),
            layer_init(nn.Linear(10, 10)),
            nn.Tanh(),
            layer_init(nn.Linear(10, np.prod(envs.action_space.shape)), std=0.0),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def get_value(self, x):
        x=torch.reshape(x,(1,-1))
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        #x=torch.reshape(x,(-1,6))
        x=torch.reshape(x,(-1,64*64))
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_steps * args.num_plays)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{int(time.time())}"

    wandb.init(
        project=args.wandb_project_name,
        config=vars(args),
        name=run_name,
    )

    device = torch.device(args.device)
    envs = gym.make(args.env_id)
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs = torch.zeros((args.num_steps * args.num_plays,) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps * args.num_plays,) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps * args.num_plays)).to(device)
    rewards = torch.zeros((args.num_steps * args.num_plays)).to(device)
    dones = torch.zeros((args.num_steps * args.num_plays)).to(device)
    values = torch.zeros((args.num_steps * args.num_plays)).to(device)

    global_step = 0
    next_obs= envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(1).to(device)


    for iteration in range(1, args.num_iterations + 1):
        for play in range(0, args.num_plays):
            treward=0
            n_o=envs.reset()
            next_obs = torch.Tensor(n_o).to(device)
            for step in range(0, args.num_steps):
                global_step += 1
                
                with torch.no_grad():
                    obs[play * args.num_steps + step] = next_obs
                    dones[play * args.num_steps + step] = next_done
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[play * args.num_steps + step] = value.flatten()
                actions[play * args.num_steps + step] = action
                logprobs[play * args.num_steps + step] = logprob

                print(action)
                next_obs, reward, terminations, _ = envs.step(action.cpu().numpy())
                next_done = np.array(terminations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.Tensor(next_done).to(device)
                treward+=reward
                if terminations:
                    break
            wandb.log({"reward": treward})
          

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[play * args.num_steps + t + 1]
                    delta = rewards[play * args.num_steps + t] + args.gamma * nextvalues * nextnonterminal - values[play * args.num_steps + t]
                    advantages[play * args.num_steps + t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

        #print(obs, "\n", actions, "\n", rewards, "\n", logprobs, "\n", dones, "\n", values, "\n ---------------------------------")
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            tloss=0
            tv_loss=0
            tpg_loss=0
            ten_loss=0
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs[mb_inds], actions[mb_inds])
                logratio = newlogprob - logprobs[mb_inds]
                logprobs[mb_inds] = newlogprob.clone().detach()
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                """wandb.log({"ratio": ratio})
                wandb.log({"approx_kl": approx_kl})"""

                newvalue = newvalue.view(-1)
                
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                tloss+=abs(loss)
                tv_loss+=abs(v_loss)
                tpg_loss+=abs(pg_loss)
                ten_loss+=abs(entropy_loss)
                
                
                loss.backward()
                wandb.log({"policy_loss": pg_loss})
                wandb.log({"value_loss": v_loss})
                wandb.log({"loss": loss})
                wandb.log({"entropy_loss": entropy_loss})
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
    
    envs.close()
