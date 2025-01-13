import os
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from environment import *
# from sampleEnv import *
# from banditEnv import *
# from EpTwoStepEnv import *
from network import *
from a2c import *


# note: vectorizing wrapper only works under this protection
if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()

    # job parameters
    parser.add_argument('--jobid', type = str, default = '0', help = 'job id')
    parser.add_argument('--path', type = str, default = os.path.join(os.getcwd(), 'results'), help = 'path to store results')

    # nework parameters
    parser.add_argument('--hidden_size', type = int, default = 64, help = 'lstm hidden size')

    # environment parameters
    parser.add_argument('--num_trials', type = int, default = 100, help = 'number of trials per episode')
    parser.add_argument('--flip_prob', type = float, default = 0.2, help = 'flip probability')
    # parser.add_argument('--reward_prob', type = float, default = 0.8, help = 'reward probability')
    # parser.add_argument('--reward_prob', type = eval, default = [(1, 1), (0.1, 9)], help = 'reward probabilities for each action')
    
    # training parameters
    parser.add_argument('--num_episodes', type = int, default = 20000, help = 'training episodes')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'batch_size')
    parser.add_argument('--gamma', type = float, default = 0.9, help = 'temporal discount') #0.9
    parser.add_argument('--lamda', type = float, default = 1.0, help = 'generalized advantage estimation coefficient')
    parser.add_argument('--beta_v', type = float, default = 0.1, help = 'value loss coefficient')
    parser.add_argument('--beta_e', type = float, default = 0.05, help = 'entropy regularization coefficient')
    parser.add_argument('--max_grad_norm', type = float, default = 1.0, help = 'gradient clipping')

    args = parser.parse_args()

    # set experiment path
    exp_path = os.path.join(args.path, f'exp_{args.jobid}')
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # set environment
    seeds = [random.randint(0, 1000) for _ in range(args.batch_size)]
    env = gym.vector.AsyncVectorEnv([
        lambda: MetaLearningWrapper(
            HarlowEnv(
                num_trials = args.num_trials,
                flip_prob = args.flip_prob,
                seed = seeds[i],
            )
        )
        for i in range(args.batch_size)
    ])

    # env = gym.vector.AsyncVectorEnv([
    #     lambda: MetaLearningWrapper(
    #         EpTwoStepEnv(
    #             num_trials = args.num_trials,
    #             common_prob = 0.9,
    #             ctx_len = 10,
    #             seed = seeds[i],
    #         )
    #     )   
    #     for i in range(args.batch_size)
    # ])

    # env = gym.vector.AsyncVectorEnv([
    #     lambda: MetaLearningWrapper(
    #         PartialFeedbackEnv(
    #             num_trials = args.num_trials,
    #             seed = seeds[i],
    #         )
    #     )
    #     for i in range(args.batch_size)
    # ])

    # env = gym.vector.AsyncVectorEnv([
    #     lambda: MetaLearningWrapper(
    #         IowaGamblingTask(
    #             num_trials = args.num_trials,
    #             seed = seeds[i],
    #         )
    #     )
    #     for i in range(args.batch_size)
    # ])
    # env = gym.vector.AsyncVectorEnv([
    #     lambda: MetaLearningWrapper(
    #         IndependentBanditEnv(
    #             num_trials = args.num_trials,
    #             num_arms = 2,
    #             seed = seeds[i],
    #         )
    #     )
    #     for i in range(args.batch_size)
    # ])
    # env = gym.vector.AsyncVectorEnv([
    #     lambda: MetaLearningWrapper(
    #         ExploreExploitEnv(
    #             trial_length = args.num_trials,
    #             mean = 3.0,
    #             std = 0.6,
    #             reward_range = (1.0, 5.0),
    #             seed = None
    #         )
    #     )
    #     for i in range(args.batch_size)
    # ])


    # set net
    # net = LSTMRecurrentActorCriticPolicy(
    #     feature_dim = env.single_observation_space.shape[0],
    #     action_dim = env.single_action_space.n,
    #     lstm_hidden_dim = args.hidden_size,
    # )

    net = EpLSTMRecurrentActorCriticPolicy(
        feature_dim = env.single_observation_space.shape[0],
        action_dim = env.single_action_space.n,
        lstm_hidden_dim = args.hidden_size,
    )

    # set model
    model = BatchMaskA2C(
        net = net,
        env = env,
        lr = args.lr,
        batch_size = args.batch_size,
        gamma = args.gamma,
        lamda = args.lamda,
        beta_v = args.beta_v,
        beta_e = args.beta_e,
        max_grad_norm = args.max_grad_norm,
    )

    # train network
    data = model.learn(
        num_episodes = args.num_episodes,
        print_frequency = 10
    )
    
    # save net and data
    model.save_net(os.path.join(exp_path, f'net.pth'))
    model.save_data(os.path.join(exp_path, f'data_training.p'))

    # visualization
    plt.figure()
    num_segments = 100
    segment_length = len(data['episode_reward']) // num_segments
    segment_rewards = np.array(data['episode_reward'][:num_segments * segment_length])
    reshaped_rewards = segment_rewards.reshape(num_segments, segment_length)

    plt.plot(reshaped_rewards.mean(axis = 1))
    plt.xlabel('Episodes Percentage')
    plt.ylabel('Episodes Reward')
    # plt.plot(np.array(data['episode_reward']).reshape(100, -1).mean(axis = 1))
    plt.show()