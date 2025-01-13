import numpy as np
import random

import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete

import numpy as np

def generate_problem(problem_domain):

    # Step 1: Draw probability p from one of three intervals with equal probability
    interval_choice = np.random.choice([0, 1, 2])
    if interval_choice == 0:
        p = np.round(np.random.uniform(0.01, 0.09), 2)
    elif interval_choice == 1:
        p = np.round(np.random.uniform(0.1, 0.9), 2)
    else:
        p = np.round(np.random.uniform(0.91, 0.99), 2)

    # Step 2: Generate risky option values
    Xmin = np.round(np.random.uniform(0, 1), 1) # [-10, 0]
    Xmax = np.round(np.random.uniform(0, 5), 1) # [0, 10]
    H_prime = round(Xmax, 1)
    L_prime = round(Xmin, 1)

    # Step 3: Calculate expected value for the risky option and create the safe option
    m = round(H_prime * p + L_prime * (1 - p), 1)
    SD = min(abs(m - L_prime) / 2, abs(m - H_prime) / 2, 2)
    e = np.random.normal(0, SD)
    m = m + e  # Adjusted expected value for the safe option

    # Step 4: Adjust for payoff domains
    if problem_domain == 0:
        con = -Xmax + Xmin
    elif problem_domain == 1:
        con = 0
    else:
        con = Xmax - Xmin

    # Final values for risky and safe options after adding the constant
    L = round(L_prime + con, 1)
    M = round(m + con, 1)
    H = round(H_prime + con, 1)

    return {
        "problem_domain": problem_domain,
        "p": p,
        "risky_option": {"Xmax": H, "Xmin": L},
        "safe_option": M,
        "expected_value": round(H * p + L * (1 - p), 1)
    }


class SamplingEnv(gym.Env):
    """
    A sampling paradigm environment for decision-making under risk.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, num_trials=100, seed=None, risky_max=None, risky_min=None, p=None, safe_reward=None):
        super().__init__()
        
        self.num_trials = num_trials
        self.risky_max = risky_max
        self.risky_min = risky_min
        self.p = p # p_min = 1 - p
        self.safe_reward = safe_reward

        self.set_random_seed(seed)

        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.trial_count = 0
        
        if self.risky_max is None or self.risky_min is None or self.p is None:
            problem = generate_problem(np.random.choice([0, 1, 2]))
            domain = problem["problem_domain"]
            self.risky_max = problem["risky_option"]["Xmax"]
            self.risky_min = problem["risky_option"]["Xmin"]
            self.p = problem["p"]
            self.safe_reward = problem["safe_option"]

        # initialize the observation
        obs = np.array([0.])

        info = {'trial': None, 'action': None,'reward': None, 'mask': self.action_mask(), 
                'risky_max': self.risky_max, 'risky_min': self.risky_min, 'p': self.p, 'safe_reward': self.safe_reward}
        return obs, info

    def sample(self, action):
        if action == 0:  # risk option
            return self.risky_reward if np.random.rand() < self.risky_prob else 0
        else:            # safe option
            return self.safe_reward

    def step(self, action):
        if self.sample_phase:
            # sample phase, return 0 reward
            reward = self.sample(action)
            obs = np.array([0.])
            info = {'trial': self.trial_count, 'action': action, 'reward': reward, 'mask': self.action_mask(), 
                'risky_max': self.risky_max, 'risky_min': self.risky_min, 'p': self.p, 'safe_reward': self.safe_reward}
            return obs, reward, done, False, info
        else:
            # feedback phase, return the reward
            reward = self.sample(action)
            done = True
            obs = np.array([0.])
            info = {'trial': self.trial_count, 'action': action, 'reward': reward, 'mask': self.action_mask(), 
                'risky_max': self.risky_max, 'risky_min': self.risky_min, 'p': self.p, 'safe_reward': self.safe_reward}
            return obs, reward, done, False, info

    def render(self, mode='human'):
        print(f"Trial: {self.trial_count}, Risky Option: (p={self.risky_prob}, r={self.risky_reward}), Safe Option: (r={self.safe_reward})")

    def one_hot_coding(self, num_classes, labels=None):
        """
        One-hot code nodes.
        """
        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]
        return labels_one_hot

    def action_mask(self):
        """
        Get action mask.
        """
        mask = np.ones((self.action_space.n,), dtype=bool)
        return mask
    
    def set_random_seed(self, seed):
        """
        Set random seed.
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

class PartialFeedbackEnv(gym.Env):
    """
    A partial-feedback paradigm environment for decision-making under risk.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, num_trials=100, seed=None, risky_max=None, risky_min=None, p=None, safe_reward=None):
        super().__init__()
        
        self.num_trials = num_trials
        self.risky_max = risky_max
        self.risky_min = risky_min
        self.p = p
        self.safe_reward = safe_reward

        self.set_random_seed(seed)

        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.trial_count = 0
        
        # if self.risky_max is None or self.risky_min is None or self.p is None:
        # problem = generate_problem(np.random.choice([2])) # 0, 1, 2
        # domain = problem["problem_domain"]
        # self.risky_max = problem["risky_option"]["Xmax"]
        # self.risky_min = problem["risky_option"]["Xmin"]
        # self.p = problem["p"]
        # self.safe_reward = problem["safe_option"]

        # initialize the observation
        obs = np.array([-1])

        info = {'trial': None, 'action': None,'reward': None, 'mask': self.action_mask(), 
                'risky_max': self.risky_max, 'risky_min': self.risky_min, 'p': self.p, 'safe_reward': self.safe_reward, 
                'expected_value': self.risky_max * self.p + self.risky_min * (1 - self.p)}
        return obs, info

    def step(self, action):

        if action == 0:  # risk option
            reward = self.risky_max if np.random.rand() < self.p else self.risky_min
        else:            # safe option
            reward = self.safe_reward

        # update the trial count
        self.trial_count += 1
        done = self.trial_count >= self.num_trials

        # # 增加对高风险选项的额外奖励或惩罚
        # if action == 0 and reward == self.risky_max:
        #     reward += 0.5  # 奖励高风险成功
        # elif action == 0 and reward == self.risky_min:
        #     reward -= 0.5  # 惩罚高风险失败

        obs = np.array([action])
        info = {'trial': self.trial_count, 'action': action, 'reward': reward, 'mask': self.action_mask(), 
                'risky_max': self.risky_max, 'risky_min': self.risky_min, 'p': self.p, 'safe_reward': self.safe_reward,
                'expected_value': round(self.risky_max * self.p + self.risky_min * (1 - self.p), 1)}

        return obs, reward, done, False, info

    def render(self, mode='human'):
        print(f"Trial: {self.trial_count}, Safe Option: (r={self.safe_reward}), "
              f"Risky Option: (max={self.risky_max}, min={self.risky_min}, p={self.p})",
              f"Expected Value: {self.risky_max * self.p + self.risky_min * (1 - self.p)}")
    
    def one_hot_coding(self, num_classes, labels=None):
        """
        One-hot code nodes.
        """
        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]
        return labels_one_hot

    def action_mask(self):
        """
        Get action mask.
        """
        mask = np.ones((self.action_space.n,), dtype=bool)
        return mask
    
    def set_random_seed(self, seed):
        """
        Set random seed.
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

class FullFeedbackEnv(gym.Env):
    """
    A full-feedback paradigm environment for decision-making under risk.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, num_trials=100, seed=None):
        super().__init__()
        
        self.num_trials = num_trials
        self.set_random_seed(seed)

        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.trial_count = 0
        
        # 随机生成收益和概率，确保预期收益一致
        self.risky_prob = np.random.uniform(0.1, 1.0)
        self.risky_reward = np.random.uniform(1, 10)
        expected_value = self.risky_prob * self.risky_reward
        self.safe_reward = expected_value

        obs = np.array([0.])
        info = {'mask': self.action_mask()}

        return obs, info

    def step(self, action):
        # 计算所选和未选选项的奖励
        chosen_prob, chosen_reward_if_success = (self.risky_prob, self.risky_reward) if action == 0 else (1, self.safe_reward)
        not_chosen_prob, not_chosen_reward_if_success = (1, self.safe_reward) if action == 0 else (self.risky_prob, self.risky_reward)

        chosen_reward = chosen_reward_if_success if np.random.rand() < chosen_prob else 0
        not_chosen_reward = not_chosen_reward_if_success if np.random.rand() < not_chosen_prob else 0

        self.trial_count += 1
        done = self.trial_count >= self.num_trials

        obs = np.array([0.])
        info = {
            'chosen_action': action,
            'reward_received': chosen_reward,
            'not_chosen_reward': not_chosen_reward
        }
        return obs, chosen_reward, done, False, info

    def render(self, mode='human'):
        print(f"Trial: {self.trial_count}, Risky Option: (p={self.risky_prob}, r={self.risky_reward}), Safe Option: (r={self.safe_reward})")

    def one_hot_coding(self, num_classes, labels=None):
        """
        One-hot code nodes.
        """
        if labels is None:
            labels_one_hot = np.zeros((num_classes,))
        else:
            labels_one_hot = np.eye(num_classes)[labels]
        return labels_one_hot

    def action_mask(self):
        """
        Get action mask.
        """
        mask = np.ones((self.action_space.n,), dtype=bool)
        return mask
    
    def set_random_seed(self, seed):
        """
        Set random seed.
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

class MetaLearningWrapper(Wrapper):
    """
    A meta-RL wrapper.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, env):
        """
        Construct an wrapper.
        """

        super().__init__(env)

        self.env = env
        self.one_hot_coding = env.get_wrapper_attr('one_hot_coding')

        self.init_prev_variables()

        new_observation_shape = (
            self.env.observation_space.shape[0] +
            self.env.action_space.n + # previous action
            1, # previous reward
        )
        self.observation_space = Box(low = -np.inf, high = np.inf, shape = new_observation_shape)


    def step(self, action):
        """
        Step the environment.
        """

        obs, reward, done, truncated, info = self.env.step(action)

        obs_wrapped = self.wrap_obs(obs)

        self.prev_action = action
        self.prev_reward = reward

        return obs_wrapped, reward, done, truncated, info
    

    def reset(self, seed = None, options = {}):
        """
        Reset the environment.
        """

        obs, info = self.env.reset()

        self.init_prev_variables()

        obs_wrapped = self.wrap_obs(obs)

        return obs_wrapped, info
    

    def init_prev_variables(self):
        """
        Reset previous variables.
        """

        self.prev_action = None
        self.prev_reward = 0.


    def wrap_obs(self, obs):
        """
        Wrap observation with previous variables.
        """

        obs_wrapped = np.hstack([
            obs,
            self.one_hot_coding(num_classes = self.env.action_space.n, labels = self.prev_action),
            self.prev_reward
        ])
        return obs_wrapped
    
if __name__ == "__main__":
    env = PartialFeedbackEnv(num_trials= 10)
    env = MetaLearningWrapper(env)
    for i in range(10):
        obs, info = env.reset()
        print(info)
    total_reward = 0
    done = False