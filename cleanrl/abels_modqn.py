# 
# Inspired by Conditioned DQN, by Abels et al
# 
# We have separated the randomization components into:
# - on-policy randomization ~ [fixed, random_per_episode, random_periodic]
# - hindsight replay buffer ~ [original, random, ...]
#
# This diverges from Abels, who used a mix of on-policy and off-policy preferences
# when fetching from the replay buffer. We are less concerned with balancing
# on-policy learning with off policy learning, and are focussed instead on rapidly
# identifying the Pareto front.

import os
import random
import time
import sys
from dataclasses import dataclass

import gymnasium as gym
import mo_gymnasium as mo_gym

from gymnasium.utils.step_api_compatibility import convert_to_done_step_api

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard.writer import SummaryWriter

from cleanrl_utils.mo_buffers import MOReplayBuffer

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str|None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "minecart-deterministic-v0"
    """the id of the environment"""

    reward_weights: list[float] | None = None
    """Linear scalarization weights for the rewards. Default behaviour is equal weighting for all rewards"""

    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name, gamma=1.0):
    def thunk():
        if capture_video and idx == 0:
            env = mo_gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = mo_gym.make(env_id)
        env = mo_gym.wrappers.MORecordEpisodeStatistics(env, gamma=gamma)
        env.action_space.seed(seed)
        #DEBUG - calc and display Pareto Front and CCS
        #for res in env.unwrapped.pareto_front(0.98):
        #    print(np.array2string(res, separator=","))
        #sys.exit()

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        reward_dimension = env.reward_space.shape[0]

        # Split into a "state representation" network, and an action head
        # Practically, this is a single large 2-hidden-layer NN,
        # however, the split allows us to inject preference values after the
        # first network, allowing potential future extension to CNN, DuellingDQN, etc
        
        # Note: Compared to linear_modqn, this archicture adds an additional 120x120
        # linear layer

        self.state_network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU()
        )

        self.action_head = nn.Sequential(
            nn.Linear(120 + reward_dimension, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n * reward_dimension),
        )

    def forward(self, x, prefs):
        state_rep = self.state_network(x)
        return self.action_head(torch.cat( (state_rep, prefs), dim=1))


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    assert args.num_envs == 1, "Only supports one environment at this time"

    # env setup
    envs = mo_gym.wrappers.vector.MOSyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, gamma=args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    reward_dimension = envs.reward_space.shape[0]

    reward_weights = args.reward_weights
    if reward_weights is None:
        reward_weights = [1.0] * reward_dimension
    reward_weights = torch.as_tensor(reward_weights, dtype=torch.float32).to(device)
    

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = MOReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        reward_dimension,
        True,       # Store preferences
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    autoreset = np.zeros(envs.num_envs)
    
    for global_step in range(args.total_timesteps):

        # Select the greedy action, and then override with e-greedy
        # This lets us log the greedy action in all cases, even if we choose a
        # random action
        q_values = q_network(torch.Tensor(obs).to(device), reward_weights.repeat(args.num_envs, 1))

        # Therefore, we need to find the maximum *per reward* here
        # reshape to (A, R)
        q_values = q_values.reshape(envs.single_action_space.n, reward_dimension)

        # Convert to scalar weighted so we can find the max
        # shape = (A)
        q_scalar = (q_values * reward_weights).sum(dim=-1)

        # Then select the vector that corresponds to max scalarized
        # @TODO: For parallel environments this needs to be a vector for n environments
        greedy_actions = actions = torch.atleast_1d(q_scalar.argmax()).cpu().numpy()
        
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])

        if autoreset[0] or global_step == 0:
            # If we start an episode, log the Q value for the greedy action (regardless of what it was)
            start_q_values = q_values[greedy_actions[0]]
            start_q_scalar = q_scalar[greedy_actions[0]]
        
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # Caution: As elsewhere, the plotting of overestimation is based on a single environment.
        # If there are multiple environments, the same starting q value is plotted multiple times.
        if infos and "_episode" in infos:
            episode = infos['episode']
            for env_idx, ep in enumerate(infos["_episode"]):
                if ep:
                    print(f"global_step={global_step}, episodic_return={episode['r'][env_idx]}")
                    for i, r in enumerate(episode['r'][env_idx]):
                        writer.add_scalar(f"charts/episodic_return_{i}", r, global_step)

                    for i, r in enumerate(episode['dr'][env_idx]):
                        writer.add_scalar(f"charts/discounted_episodic_return_{i}", r, global_step)

                    for n in range(reward_dimension):
                        writer.add_scalar(f"qvalues/greedy_{n}", start_q_values[n], global_step )
                    
                    writer.add_scalar("qvalues/greedy_scalarised_q", start_q_scalar, global_step)

                    overestimation = start_q_values.detach().numpy() - episode['dr'][env_idx]
                    for i, r in enumerate(overestimation):
                        writer.add_scalar(f"overestimation/q_{i}", r, global_step)

                    writer.add_scalar("charts/scalar_episodic_return", (episode['r'][env_idx] * reward_weights.numpy()).sum(), global_step)

                    discounted_scalar_episodic_return = (episode['dr'][env_idx] * reward_weights.numpy()).sum()

                    writer.add_scalar("charts/discounted_scalar_episodic_return", discounted_scalar_episodic_return, global_step)
                    writer.add_scalar("overestimation/q_scalar", start_q_scalar - discounted_scalar_episodic_return, global_step)
                                      
                    
                    writer.add_scalar("charts/episodic_length", episode["l"][env_idx], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()

        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         # real_next_obs[idx] = infos["terminal_observation"][idx]
        #         real_next_obs[idx] = infos["final_obs"][idx]

        # IMPORTANT: ONLY SUPPORTS A SINGLE ENVIRONMENT
        # (or, rather, does not store transitions if any environment terminated and reset)
        if not autoreset[0]:
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos, reward_weights)
        
        autoreset = np.logical_or(terminations, truncations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():

                    # Note that the output layer of the QNetwork is of dimension
                    # actions * rewards

                    # Steps
                    
                    # Therefore, we need to find the maximum *per reward* here
                    next_obs_values = target_network(data.next_observations, data.preferences)
                    # reshape to (B, A, R)
                    next_obs_values = next_obs_values.reshape(-1, envs.single_action_space.n, reward_dimension)

                    # Convert to scalar weighted so we can find the max
                    # shape = (B, A)
                    next_obs_scaled = (next_obs_values * reward_weights)
                    next_obs_scalar = next_obs_scaled.sum(dim=-1)

                    # Then select the vector that is corresponds to max scalarized
                    target_argmax = next_obs_scalar.argmax(dim=1)

                    # Convert the argmax (which is now (B,)) to (B, 1, R) to allow tensor gather
                    idx = target_argmax.view(-1, 1, 1).expand(-1, 1, reward_dimension)
                    target_max = next_obs_values.gather(dim=1, index=idx).squeeze(1)

                    # Multiplying by 1-data.dones.flatten is to ignore the transitions from end of
                    # one episode to the start of the next
                    # shape = (B, R)
                    td_target = data.rewards + args.gamma * target_max * (1 - data.dones)

                old_val = q_network(data.observations, data.preferences)      # shape = (B, A*R)

                # reshape to (B, A, R)
                old_val = old_val.reshape(-1, envs.single_action_space.n, reward_dimension)
                
                # choose only actions that were actually taken
                # need shape (B, R). data.actions is shape (B, 1) for single actions
                # need to convert to (B, 1, R)
                idx = data.actions.view(-1, 1, 1).expand(-1, 1, reward_dimension)
                old_action_val = old_val.gather(dim=1, index=idx).squeeze(1)

                loss = F.mse_loss(td_target, old_action_val)



                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    
                    print(f"global_step: {global_step}")
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
