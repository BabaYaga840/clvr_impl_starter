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
from torch.distributions import Beta
from utility import recon
import wandb

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    torch_deterministic: bool = True
    track: bool = True
    wandb_project_name: str = "PPO_sprites"
    save_model: bool = True
    device: str = "cpu"
    freeze_encoder: bool = False

    # Algorithm specific arguments
    env_id: str = "Sprites-v0"
    total_timesteps: int = 100000000
    learning_rate: float = 1e-4
    num_steps: int = 40
    num_plays: int = 128
    gamma: float = 0.99
    gae_lambda: float = 1.0
    num_minibatches: int = 40
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.1
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    neck: int = 64
    checkpoint: str = "../model_weights/encoder_weights_18-00.pth"

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(args.neck, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 28)),
            nn.Tanh(),
            layer_init(nn.Linear(28, 8)),
            nn.Tanh(),
            layer_init(nn.Linear(8, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(args.neck, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 8)),
            nn.Tanh(),
            layer_init(nn.Linear(8, np.prod(envs.action_space.shape))),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def get_value(self, x):
        x=torch.reshape(x,(1,-1))
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x=torch.reshape(x,(-1,args.neck))
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        epsilon = 1e-6
        #print(action_mean, "\n", action_std, "\n\n\n")
        #probs = Normal(action_mean, action_std)
        action_mean = torch.exp(action_mean)
        probs = Beta(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:
            action = (action + 1)/2
        x=torch.reshape(x,(-1,args.neck))
        action = action.clamp(min = epsilon, max = 1 - epsilon)
        #wandb.log({"action": action})
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_steps * args.num_plays)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__pretrained__{int(time.time())}"

    wandb.init(
        project=args.wandb_project_name,
        config=vars(args),
        name=run_name,
    )

    device = torch.device(args.device)
    envs = gym.make(args.env_id)
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs = torch.zeros((args.num_steps * args.num_plays,) + (args.neck,)).to(device)
    actions = torch.zeros((args.num_steps * args.num_plays,) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps * args.num_plays)).to(device)
    rewards = torch.zeros((args.num_steps * args.num_plays)).to(device)
    dones = torch.zeros((args.num_steps * args.num_plays)).to(device)
    values = torch.zeros((args.num_steps * args.num_plays)).to(device)

    enc=recon.Encoder(args.neck)
    encoder_weights = torch.load(args.checkpoint)
    enc.load_state_dict(encoder_weights)

    global_step = 0
    next__obs= envs.reset()
    next__obs = torch.Tensor(next__obs).to(device)
    next__obs = torch.Tensor(next__obs).unsqueeze(0)
    next__obs = torch.Tensor(next__obs).unsqueeze(0)
    print(next__obs.size())
    if args.freeze_encoder:
        next_obs = enc(next__obs).to(device).clone().detach()
    else:
        next_obs = enc(next__obs).to(device)
    next_done = torch.zeros(1).to(device)

    i=0
    for iteration in range(1, args.num_iterations + 1):
        for play in range(0, args.num_plays):
            treward=0
            n_o=envs.reset()
            next__obs = torch.Tensor(n_o).to(device)
            next__obs = torch.Tensor(next__obs).to(device)
            next__obs = next__obs.unsqueeze(0)
            next__obs = next__obs.unsqueeze(0)
            if args.freeze_encoder:
                next_obs = enc(next__obs).to(device).clone().detach()
            else:
                next_obs = enc(next__obs).to(device)
            for step in range(0, args.num_steps):
                global_step += 1
                
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    obs[play * args.num_steps + step] = next_obs
                    dones[play * args.num_steps + step] = next_done
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    action = 2 * action - 1
                    values[play * args.num_steps + step] = value.flatten()
                actions[play * args.num_steps + step] = action
                logprobs[play * args.num_steps + step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next__obs, reward, terminations, _ = envs.step(action.cpu().numpy())   
                next_done = np.array(terminations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs = torch.Tensor(next__obs).to(device)
                next_done = torch.Tensor(next_done).to(device)
                next_obs = next_obs.unsqueeze(0)
                next_obs = next_obs.unsqueeze(0)
                if args.freeze_encoder:
                    next_obs = enc(next_obs).detach().clone()        
                else:
                    next_obs = enc(next_obs).to(device)
                treward+=reward
                if terminations:
                    break
            treward+= i
            i+=3/500000
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
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                
                mb_advantages = advantages[mb_inds]
                
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                #wandb.log({"ratio": ratio})
                wandb.log({"approx_kl": approx_kl})

                # Value loss
                newvalue = newvalue.view(-1)
                
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                tloss+=abs(loss)
                tv_loss+=abs(v_loss)
                tpg_loss+=abs(pg_loss)
                ten_loss+=abs(entropy_loss)
                
                
                loss.backward()

            """for name, param in agent.named_parameters():
                wandb.log({f"weights/{name}": wandb.Histogram(param.data.cpu().numpy())})
                if param.grad is not None:
                    wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.data.cpu().numpy())})
"""
            wandb.log({"policy_loss": tpg_loss})
            wandb.log({"value_loss": tv_loss})
            wandb.log({"loss": tloss})
            wandb.log({"entropy_loss": ten_loss})
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
    
    envs.close()
