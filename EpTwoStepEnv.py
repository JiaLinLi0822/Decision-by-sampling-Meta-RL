import gym
from gym.spaces import Discrete, Box
import numpy as np
import random
from gymnasium import Wrapper 

S_0 = 0
S_1 = 1
S_2 = 2


class EpTwoStepEnv(gym.Env):
    """
    A two-step task environment with binary context.
      - First half of trials: uncued (new random context + new reward)
      - Second half: cued (context from memory; if we land on same S1/S2 as before, reuse old reward)
      - One "step()" call covers both stage1 (S0->S1/S2) and stage2 (reward).
      - Then we return to S0 for the next trial.

    Observations:
      - A binary vector 'context' of length ctx_len.
        + In uncued trials, a new random context is generated each time.
        + In cued trials, we pick from memory any previously stored context 
          (if empty, fallback to new random).
    
    Actions:
      - 2 discrete actions: 0 or 1.
        + If action=0 => common_prob => S1, else S2
        + If action=1 => common_prob => S2, else S1

    Rewards:
      - If uncued => S1 => Bernoulli(0.9), S2 => Bernoulli(0.1), then store context->(Sx, reward)
      - If cued => if Sx matches memory[context], reuse old reward; otherwise draw new from {0.9, 0.1}
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self,
        num_trials=100,
        common_prob=0.9,
        ctx_len=4,
        seed=None
    ):
        """
        Parameters
        ----------
        num_trials : int
            Number of trials in one episode.
        common_prob : float
            Probability of 'common' transition from S0 to S1 or S2.
        ctx_len : int
            Length of binary context vector.
        seed : int
            Random seed (optional).
        """
        super().__init__()

        self.num_trials = num_trials
        self.common_prob = common_prob
        self.ctx_len = ctx_len

        # Number of completed trials in this episode
        self.num_completed = 0
        self.done = False

        # Action space: 2 actions
        self.action_space = Discrete(2)
        # Observation space: binary vector in [0,1] of length ctx_len
        self.observation_space = Box(low=0, high=1, shape=(self.ctx_len,), dtype=np.float32)

        # A memory: context_key -> (second_stage_state, reward)
        # If we revisit that context in cued phase, we can reuse reward if we land on the same state
        self.memory = {}

        # For randomization
        self.set_random_seed(seed)

        # Reset at init
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset environment for a new episode.
        """
        if seed is not None:
            self.set_random_seed(seed)

        self.num_completed = 0
        self.done = False
        self.memory.clear()

        # Return an initial context. Typically for the first step we do uncued 
        # or just a zero context. But let's unify by calling _get_context() here.
        obs = self._get_context(uncued=True)
        info = {
            'mask': self.get_action_mask()
        }
        return obs, info

    def step(self, action):
        """
        Execute a single step:
          1. We observe a context vector (from the last reset or last step).
          2. The agent picks an action => we transition to S1 or S2 with prob = common_prob.
          3. We get a reward, either "new" or "reused" from memory if in cued phase and matching.
          4. We return the next context in obs (for next trial).
        """
        if self.done:
            # If already done, no more steps
            obs = np.zeros((self.ctx_len,), dtype=np.float32)
            return obs, 0.0, True, False, {'mask': self.get_action_mask()}

        # Decide if current trial is uncued or cued
        is_uncued = (self.num_completed < self.num_trials // 2)

        # figure out second-stage state
        if action == 0:
            if np.random.rand() < self.common_prob:
                state2 = S_1
            else:
                state2 = S_2
        else:
            if np.random.rand() < self.common_prob:
                state2 = S_2
            else:
                state2 = S_1

        # get old obs (the context we had this step)
        # Because in the last 'step' call, we returned obs = self._context
        # We can store it. Or store at the beginning. 
        # For simplicity, we do:
        context = self._context  # The context from the last observation

        # compute reward
        if is_uncued:
            # new random reward from {0.9, 0.1} depending on state2
            reward = self._uncued_reward(state2)
            # store in memory: context->(state2, reward)
            # turn context into a tuple key for dictionary
            ctx_key = tuple(context.tolist())
            self.memory[ctx_key] = (state2, reward)

        else:
            # cued => if memory has context, check if same state => reuse reward
            ctx_key = tuple(context.tolist())
            if ctx_key in self.memory:
                old_state2, old_reward = self.memory[ctx_key]
                if old_state2 == state2:
                    reward = old_reward
                else:
                    reward = self._uncued_reward(state2)
            else:
                # not found => fallback to new random
                reward = self._uncued_reward(state2)

        self.num_completed += 1
        if self.num_completed >= self.num_trials:
            self.done = True

        # Next obs => new context (for next trial)
        next_obs = self._get_context(uncued=(self.num_completed < self.num_trials // 2))
        info = {
            'mask': self.get_action_mask()
        }

        # store next_obs in internal so that next step can reference it
        self._context = next_obs
        return next_obs, reward, self.done, False, info

    def _uncued_reward(self, state2):
        """
        Return Bernoulli reward for state2: s1 => p=0.9, s2 => p=0.1
        """
        if state2 == S_1:
            return float(np.random.rand() < 0.9)
        else:
            return float(np.random.rand() < 0.1)

    def _get_context(self, uncued=True):
        """
        Return a binary context vector.
          - If uncued => randomly generate a new binary vector
          - If cued => pick from memory if possible, else fallback to random
        """
        if uncued:
            # new random binary vector
            ctx = np.random.randint(0, 2, size=self.ctx_len).astype(np.float32)
        else:
            # cued => randomly pick from memory if not empty
            if len(self.memory) > 0:
                ctx_key = random.choice(list(self.memory.keys()))
                ctx = np.array(ctx_key, dtype=np.float32)
            else:
                # fallback to new random
                ctx = np.random.randint(0, 2, size=self.ctx_len).astype(np.float32)
        # store it in self._context so step() can reference
        self._context = ctx
        return ctx

    def get_action_mask(self):
        """
        Return an action mask. 
        Here we allow both actions => [True, True].
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
    
    def one_hot_coding(self, num_classes, labels=None):
        """
        Optional: one-hot coding for some usage
        """
        if labels is None:
            return np.zeros((num_classes,))
        else:
            return np.eye(num_classes)[labels]
    
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
        # self.one_hot_coding = env.get_wrapper_attr('one_hot_coding')
        self.one_hot_coding = env.one_hot_coding

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