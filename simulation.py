import torch
import numpy as np
import pandas as pd
import gymnasium as gym

# from environment import *
from sampleEnv import *
from a2c import BatchMaskA2C
from network import LSTMRecurrentActorCriticPolicy
from replaybuffer import *


if __name__ == '__main__':
    # Initialize vectorized environment
    # env = gym.vector.AsyncVectorEnv([
    #     lambda: MetaLearningWrapper(
    #         ExploreExploitEnv(
    #             trial_length=180,
    #             mean=3.0,
    #             std=0.6,
    #             reward_range=(1.0, 5.0),
    #             seed=None
    #         )
    #     )
    #     for _ in range(16)  # batch_size
    # ])
    # env = gym.vector.AsyncVectorEnv([
    #     lambda: MetaLearningWrapper(
    #         HarlowEnv(
    #             num_trials=180,
    #             flip_prob=0.1,
    #             seed=None,
    #         )
    #     )
    #     for _ in range(16)  # batch_size
    # ])

    decision_problems = [
        {"safe_reward": 7, "risky_max": 16.5, "risky_min": 6.9, "p": 0.01},
        {"safe_reward": -4.1, "risky_max": 1.3, "risky_min": -4.3, "p": 0.05},
        {"safe_reward": 11.5, "risky_max": 25.6, "risky_min": 8.1, "p": 0.10},
        {"safe_reward": 2.2, "risky_max": 3.0, "risky_min": -7.2, "p": 0.93},
        {"safe_reward": 6.8, "risky_max": 7.3, "risky_min": -8.5, "p": 0.96},
        {"safe_reward": 11, "risky_max": 11.4, "risky_min": 1.9, "p": 0.97},
        {"safe_reward": 0.7, "risky_max": 8.9, "risky_min": -6.9, "p": 0.34},
        {"safe_reward": 100, "risky_max": 1, "risky_min": 0, "p": 1},
    ]

    n = len(decision_problems)
    data = []
    for i, problem in enumerate(decision_problems):
        
        risk_max = problem["risky_max"]
        risk_min = problem["risky_min"]
        p = problem["p"]
        safe_reward = problem["safe_reward"]

        env = gym.vector.AsyncVectorEnv([
            lambda: MetaLearningWrapper(
                PartialFeedbackEnv(
                    num_trials=100,
                    seed=None,
                    risky_max=risk_max,
                    risky_min=risk_min,
                    p=p,
                    safe_reward=safe_reward
                )
            )
            for _ in range(1)  # batch_size
        ])  

        model_path = "/Users/lijialin/Desktop/NYU/meta_rl_template-main-2/results/exp_0/net.pth"
        net = torch.load(model_path)
        net.eval()  # Set network to evaluation mode

        # Run simulation with the vectorized environment
        num_episodes = 10  # Number of episodes to simulate per environment in batch
        batch_size = 1  # Number of environments to simulate in parallel
        problem_data = []

        for episode in range(num_episodes):
            
            buffer = BatchReplayBuffer()
                
            # initialize a trial
            dones = np.zeros(batch_size, dtype = bool) # no reset once turned to 1
            mask = torch.ones(batch_size)
            states_hidden = None

            obs, info = env.reset()
            obs = torch.Tensor(obs)
            action_mask = torch.tensor(np.stack(info['mask'])) # (batch_size, action_dim), bool

            episode_data = []

            # iterate through a trial
            while not all(dones):
                # step the net
                action, policy, log_prob, entropy, value, states_hidden = net(
                    obs, states_hidden, action_mask,
                )
                value = value.view(-1) # (batch_size,)

                # step the env
                obs, reward, done, truncated, info = env.step(action)
                obs = torch.Tensor(obs) # (batch_size, feature_dim)
                reward = torch.Tensor(reward) # (batch_size,)
                action_mask = torch.tensor(np.stack(info['mask'])) # (batch_size, action_dim), bool

                # push results (make sure shapes are (batch_size,))
                buffer.push(
                    masks = mask, # (batch_size,)
                    log_probs = log_prob, # (batch_size,)
                    entropies = entropy, # (batch_size,)
                    values = value, # (batch_size,)
                    rewards = reward, # (batch_size,)
                )

                # Append trial data to episode_data
                episode_data.append({
                    "trial": info['trial'],
                    "action": action.item(),
                    "reward": round(reward.item(), 1),
                    "risky_max": info['risky_max'].item(),
                    "risky_min": info['risky_min'].item(),
                    "p": info['p'].item(),
                    "safe_reward": info['safe_reward'].item(),
                    "episode": episode,
                    "decision_problem": i + 1
                })

                # update mask and dones
                # note: the order of the following two lines is crucial
                dones = np.logical_or(dones, done)
                mask = (1 - torch.Tensor(dones)) # keep 0 once a batch is done

            # process the last timestep
            value = torch.zeros((batch_size,)) # zero padding for the last time step
            buffer.push(values = value) # push value for the last time step

            # reformat rollout data into (batch_size, seq_len) and mask finished time steps
            buffer.reformat()

            # compute reward and length of the episode
            episode_length = buffer.rollout['masks'].sum(axis = 1).mean(axis = 0)
            episode_reward = (buffer.rollout['rewards'] * buffer.rollout['masks']).sum(axis = 1).mean(axis = 0)

            print(f"Episode {episode} | Length: {episode_length} | Reward: {episode_reward}")
            
            problem_data.extend(episode_data)
    
        data.extend(problem_data)

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv("simulation_results.csv", index=False)
    print("Simulation data saved to simulation_results.csv")