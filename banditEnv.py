import numpy as np
import gym
from gym.spaces import Discrete, Box
import random
import gymnasium as gym
from gymnasium import Wrapper 
from gymnasium.spaces import Box, Discrete

class IndependentBanditEnv(gym.Env):
    """
    Bandits with independent arms.
    """
    def __init__(self, num_trials=100, num_arms=2, seed=None):
        """
        Initialize the bandit environment.

        Args:
            num_trials (int): Number of trials in each episode.
            num_arms (int): Number of arms in the bandit task.
            seed (int): Random seed for reproducibility.
        """
        self.num_trials = num_trials
        self.num_arms = num_arms
        self.current_trial = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Action space: number of arms
        self.action_space = Discrete(num_arms)
        # Observation space: dummy (not used in bandit problems)
        self.observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        """
        Reset the environment for a new episode.
        """
        self.current_trial = 0
        self.arm_probs = self.rng.uniform(0, 1, size=self.num_arms)  # Sample probabilities for arms
        obs = -1  # Dummy observation
        info = {"arm_probs": self.arm_probs, 'mask': self.action_mask()}
        return obs, info

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): The arm selected by the agent.

        Returns:
            obs (np.array): Dummy observation.
            reward (float): Reward for the chosen arm.
            done (bool): Whether the episode is finished.
            truncated (bool): Gym compatibility placeholder.
            info (dict): Additional information.
        """

        reward = 1.0 if self.rng.random() < self.arm_probs[action] else 0.0
        self.current_trial += 1
        done = self.current_trial >= self.num_trials
        obs = action  # arm selected
        info = {"arm_probs": self.arm_probs, 'mask': self.action_mask()}
        return obs, reward, done, False, info

    def render(self, mode="human"):
        """
        Render the current state.
        """
        print(f"Trial {self.current_trial}/{self.num_trials}")
        print(f"Arm probabilities: {self.arm_probs}")
    
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

class DependentBanditEnv(gym.Env):
    """
    Bandits with dependent arms.
    """
    def __init__(self, num_trials=100, difficulty="independent", seed=None):
        """
        Initialize the bandit environment.

        Args:
            num_trials (int): Number of trials in each episode.
            difficulty (str): Difficulty level ("easy", "medium", "hard", or "independent").
            seed (int): Random seed for reproducibility.
        """
        self.num_trials = num_trials
        self.difficulty = difficulty
        self.current_trial = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Action space: number of arms (2 arms)
        self.action_space = Discrete(2)
        # Observation space: dummy (not used in bandit problems)
        self.observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        self.reset()

    def reset(self):
        """
        Reset the environment for a new episode.
        """
        self.current_trial = 0

        if self.difficulty == "independent":
            self.p1 = self.rng.uniform(0, 1)
            self.p2 = self.rng.uniform(0, 1)
        elif self.difficulty == "easy":
            self.p1, self.p2 = 0.1, 0.9
        elif self.difficulty == "medium":
            self.p1, self.p2 = 0.25, 0.75
        elif self.difficulty == "hard":
            self.p1, self.p2 = 0.4, 0.6
        else:
            raise ValueError("Invalid difficulty level. Choose 'easy', 'medium', 'hard', or 'independent'.")

        self.arm_probs = [self.p1, self.p2]
        obs = np.zeros(1)  # Dummy observation
        return obs, {"arm_probs": self.arm_probs}

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): The arm selected by the agent.

        Returns:
            obs (np.array): Dummy observation.
            reward (float): Reward for the chosen arm.
            done (bool): Whether the episode is finished.
            truncated (bool): Gym compatibility placeholder.
            info (dict): Additional information.
        """
        assert self.action_space.contains(action), "Invalid action."

        reward = 1.0 if self.rng.random() < self.arm_probs[action] else 0.0
        self.current_trial += 1
        done = self.current_trial >= self.num_trials
        obs = np.zeros(1)  # Dummy observation
        return obs, reward, done, False, {}

    def render(self, mode="human"):
        """
        Render the current state.
        """
        print(f"Trial {self.current_trial}/{self.num_trials}")
        print(f"Arm probabilities: {self.arm_probs}")

class IowaGamblingTask(gym.Env):
    """
    Iowa Gambling Task environment with modified reward and penalty distributions.
    """
    def __init__(self, num_trials=100, seed=None):
        """
        Initialize the Iowa Gambling Task environment.

        Args:
            num_trials (int): Number of trials in the task.
            seed (int): Random seed for reproducibility.
        """
        self.num_trials = num_trials
        self.current_trial = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Action space: selecting one of four decks
        self.action_space = Discrete(4)
        # Observation space: Dummy observation
        self.observation_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        """
        Reset the environment for a new episode.
        """
        self.current_trial = 0
        obs = np.array([0.0], dtype=np.float32)  # Dummy observation

         # Deck parameters
        self.deck_rewards = self.rng.uniform(0, 150, size=4)  # Positive rewards sampled per deck
        self.deck_penalties_mean = self.rng.uniform(0, 150, size=4)  # Mean of penalties
        self.deck_penalty_probs = self.rng.uniform(0.05, 0.95, size=4)  # Penalty probabilities
        
        info = {
            'deck_rewards': self.deck_rewards,
            'deck_penalties_mean': self.deck_penalties_mean,
            'deck_penalty_probs': self.deck_penalty_probs,
            'mask': self.action_mask()
        }

        return obs, info

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): The selected deck (0 to 3).

        Returns:
            obs (np.array): Dummy observation.
            reward (float): Net reward for the chosen deck.
            done (bool): Whether the episode has finished.
            truncated (bool): Placeholder for Gym compatibility.
            info (dict): Additional information.
        """
        if action < 0 or action >= len(self.deck_rewards):
            raise ValueError("Invalid action. Must be 0, 1, 2, or 3.")
        
        # Base reward
        reward = self.deck_rewards[action]
        
        # Apply penalty if it occurs
        if self.rng.random() < self.deck_penalty_probs[action]:
            penalty = self.rng.normal(self.deck_penalties_mean[action], 10)  # Add noise to penalties
            reward -= penalty
        
        self.current_trial += 1
        done = self.current_trial >= self.num_trials
        obs = np.array([0.0], dtype=np.float32)  # Dummy observation
        info = {
            'current_trial': self.current_trial,
            'last_action': action,
            'last_reward': reward,
            'deck_rewards': self.deck_rewards,
            'deck_penalties_mean': self.deck_penalties_mean,
            'deck_penalty_probs': self.deck_penalty_probs,
            'mask': self.action_mask()
        }
        
        return obs, reward, done, False, info

    def render(self, mode="human"):
        """
        Render the current state of the task.
        """
        print(f"Trial {self.current_trial}/{self.num_trials}")
        print(f"Deck rewards: {self.deck_rewards}")
        print(f"Deck penalties mean: {self.deck_penalties_mean}")
        print(f"Deck penalty probabilities: {self.deck_penalty_probs}")

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