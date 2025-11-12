import gymnasium as gym
import numpy as np

class DiscretizedPendulumWrapper(gym.Wrapper):
    def __init__(self, env, num_bins=11):
        super().__init__(env)
        self.num_bins = num_bins
        self.low = env.action_space.low[0]
        self.high = env.action_space.high[0]
        # Define discrete action space
        self.action_space = gym.spaces.Discrete(num_bins)
        # Precompute all possible torque values
        self.bin_values = np.linspace(self.low, self.high, num_bins)

    def step(self, action):
        # Ensure 'action' is an integer index
        if isinstance(action, (np.ndarray, list)):
            action = int(action[0])
        # Convert discrete index to continuous torque
        torque = float(self.bin_values[action])
        obs, reward, terminated, truncated, info = self.env.step(np.array([torque], dtype=np.float32))
        return obs, reward, terminated, truncated, info


    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
