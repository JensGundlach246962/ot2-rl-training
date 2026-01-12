"""
Gymnasium Environment Wrapper for Opentrons OT-2 Robot
Improved reward function for better learning
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
sys.path.append('..')
from sim_class_jens import Simulation


class OT2GymEnv(gym.Env):
    """
    Gymnasium environment for training RL agents to control OT-2 pipette positioning.
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # Initialize simulation
        self.sim = Simulation(num_agents=1, render=(render_mode == 'human'))
        
        # Workspace bounds
        self.workspace_bounds = {
            'x': (-0.187, 0.253),
            'y': (-0.171, 0.220),
            'z': (0.170, 0.290)
        }
        
        # Observation space: [current_pos(3), target_pos(3), error(3)]
        self.observation_space = spaces.Box(
            low=np.array([-0.3, -0.3, 0.1, -0.3, -0.3, 0.1, -1.0, -1.0, -1.0]),
            high=np.array([0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action space: velocity commands [vx, vy, vz]
        self.action_space = spaces.Box(
            low=-0.5,
            high=0.5,
            shape=(3,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.target_position = None
        self.previous_distance = None
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state with new random target."""
        super().reset(seed=seed)
        
        # Reset simulation
        self.sim = Simulation(num_agents=1, render=(self.render_mode == 'human'))
        
        # Sample random target within workspace
        self.target_position = np.array([
            np.random.uniform(*self.workspace_bounds['x']),
            np.random.uniform(*self.workspace_bounds['y']),
            np.random.uniform(*self.workspace_bounds['z'])
        ], dtype=np.float32)
        
        # Get initial position
        state = self.sim.run([[0, 0, 0, 0]])
        robot_key = list(state.keys())[0]
        current_position = np.array(state[robot_key]['pipette_position'], dtype=np.float32)
        
        # Calculate initial distance
        self.previous_distance = np.linalg.norm(self.target_position - current_position)
        
        # Reset step counter
        self.current_step = 0
        
        # Return observation and info
        observation = self._get_observation(current_position)
        info = {'target': self.target_position, 'distance': self.previous_distance}
        
        return observation, info
    
    def step(self, action):
        """Execute one timestep with given action."""
        self.current_step += 1
        
        # Apply action (velocity command)
        velocities = np.clip(action, -0.5, 0.5)
        state = self.sim.run([np.append(velocities, 0).tolist()])
        
        # Get new position
        robot_key = list(state.keys())[0]
        current_position = np.array(state[robot_key]['pipette_position'], dtype=np.float32)
        
        # Calculate distance to target
        distance = np.linalg.norm(self.target_position - current_position)
        
        # Calculate reward
        reward = self._calculate_reward(distance)
        
        # Check termination conditions
        terminated = distance < 0.001  # Success: within 1mm
        truncated = self.current_step >= self.max_steps  # Timeout
        
        # Update previous distance
        self.previous_distance = distance
        
        # Get observation
        observation = self._get_observation(current_position)
        
        # Info dictionary
        info = {
            'distance': distance,
            'target': self.target_position,
            'success': terminated,
            'steps': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self, current_position):
        """Construct observation vector."""
        error = self.target_position - current_position
        observation = np.concatenate([
            current_position,
            self.target_position,
            error
        ], dtype=np.float32)
        return observation
    
    def _calculate_reward(self, current_distance):
        """
        Improved reward function with better scaling.
        
        Components:
        - Distance penalty: Main signal, proportional to error
        - Progress reward: Bonus for getting closer
        - Time penalty: Encourage efficiency
        - Success bonus: Tiered rewards for precision
        """
        # Main signal: negative distance (scaled)
        distance_penalty = -current_distance * 1000  # -1 for 1mm, -100 for 10cm
        
        # Progress reward: getting closer is good
        progress = (self.previous_distance - current_distance) * 10
        
        # Time penalty: encourage efficiency
        time_penalty = -0.1
        
        # Tiered success bonuses
        success_bonus = 0
        if current_distance < 0.005:  # Within 5mm
            success_bonus = 50
        if current_distance < 0.001:  # Within 1mm
            success_bonus = 100
        
        total_reward = distance_penalty + progress + time_penalty + success_bonus
        return total_reward
    
    def render(self):
        """Render environment (handled by simulation)."""
        pass
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'sim'):
            self.sim.close()
