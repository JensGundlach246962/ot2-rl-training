"""Minimal Gymnasium wrapper"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_minimal import MinimalSimulation

class MinimalOT2Env(gym.Env):
    def __init__(self, max_steps=1000):
        super().__init__()
        
        self.max_steps = max_steps
        self.sim = MinimalSimulation(render=False)
        
        # Workspace bounds
        self.bounds = {
            'x': (-0.187, 0.253),
            'y': (-0.171, 0.220),
            'z': (0.170, 0.290)
        }
        
        # Observation: [current_pos(3), target_pos(3), error(3)]
        self.observation_space = spaces.Box(
            low=np.array([-0.3, -0.3, 0.1, -0.3, -0.3, 0.1, -1.0, -1.0, -1.0]),
            high=np.array([0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action: [vx, vy, vz]
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(3,), dtype=np.float32
        )
        
        self.current_step = 0
        self.target = None
        self.prev_dist = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # New simulation
        self.sim.close()
        self.sim = MinimalSimulation(render=False)
        
        # Random target
        self.target = np.array([
            np.random.uniform(*self.bounds['x']),
            np.random.uniform(*self.bounds['y']),
            np.random.uniform(*self.bounds['z'])
        ], dtype=np.float32)
        
        current_pos = np.array(self.sim.get_pipette_position(), dtype=np.float32)
        self.prev_dist = np.linalg.norm(self.target - current_pos)
        self.current_step = 0
        
        return self._get_obs(current_pos), {}
    
    def step(self, action):
        self.current_step += 1
        
        # Execute action
        current_pos = np.array(self.sim.run(action), dtype=np.float32)
        
        # Calculate reward
        dist = np.linalg.norm(self.target - current_pos)
        reward = self._calculate_reward(dist)
        self.prev_dist = dist
        
        # Check termination
        terminated = dist < 0.001  # 1mm success
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(current_pos), reward, terminated, truncated, {'distance': dist}
    
    def _get_obs(self, current_pos):
        error = self.target - current_pos
        return np.concatenate([current_pos, self.target, error], dtype=np.float32)
    
    def _calculate_reward(self, dist):
        # Simple reward function
        distance_penalty = -dist * 1000
        progress = (self.prev_dist - dist) * 10
        time_penalty = -0.1
        
        success_bonus = 0
        if dist < 0.005:  # 5mm
            success_bonus = 50
        if dist < 0.001:  # 1mm
            success_bonus = 100
        
        return distance_penalty + progress + time_penalty + success_bonus
    
    def close(self):
        self.sim.close()
