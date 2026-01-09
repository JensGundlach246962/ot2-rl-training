"""
Gymnasium Environment Wrapper for Opentrons OT-2 Robot

Wraps the OT-2 simulation to be compatible with Stable Baselines3 RL training.
Defines observation space, action space, reward function, and termination conditions.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
sys.path.append('..')
from sim_class import Simulation


class OT2GymEnv(gym.Env):
    """
    Gymnasium environment for training RL agents to control OT-2 pipette positioning.
    
    Observation Space:
        - Current pipette position [x, y, z]
        - Target position [x, y, z]
        - Error vector [dx, dy, dz]
        Total: 9 continuous values
    
    Action Space:
        - Velocity commands [vx, vy, vz] in range [-0.5, 0.5] m/s
    
    Reward Function:
        - Dense reward based on distance reduction
        - Penalty for time steps
        - Bonus for reaching target
    """
    
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, render_mode=None, max_steps=2000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # Initialize simulation
        self.sim = Simulation(num_agents=1, render=(render_mode == 'human'))
        
        # Workspace bounds (from Task 10)
        self.workspace_bounds = {
            'x': (-0.187, 0.253),
            'y': (-0.171, 0.220),
            'z': (0.170, 0.290)
        }
        
        # Define observation space: [current_pos(3), target_pos(3), error(3)]
        self.observation_space = spaces.Box(
            low=np.array([-0.3, -0.3, 0.1, -0.3, -0.3, 0.1, -1.0, -1.0, -1.0]),
            high=np.array([0.3, 0.3, 0.4, 0.3, 0.3, 0.4, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Define action space: velocity commands [vx, vy, vz]
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
        Calculate reward based on distance to target.
        
        Reward components:
        - Progress reward: Positive if getting closer, negative if moving away
        - Time penalty: Small penalty per step to encourage efficiency
        - Success bonus: Large bonus for reaching target
        """
        # Progress reward (dense reward based on distance reduction)
        progress_reward = (self.previous_distance - current_distance) * 100
        
        # Time penalty (encourage faster convergence)
        time_penalty = -0.01
        
        # Success bonus
        success_bonus = 100 if current_distance < 0.001 else 0
        
        total_reward = progress_reward + time_penalty + success_bonus
        
        return total_reward
    
    def render(self):
        """Render environment (handled by simulation)."""
        pass
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'sim'):
            self.sim.close()


# Test the environment
if __name__ == "__main__":
    print("Testing OT2 Gymnasium Environment...")
    
    env = OT2GymEnv(render_mode='human')
    
    # Test reset
    observation, info = env.reset()
    print(f"\nInitial observation shape: {observation.shape}")
    print(f"Target position: {info['target']}")
    print(f"Initial distance: {info['distance']:.4f}m")
    
    # Test random actions for 100 steps
    for step in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Distance: {info['distance']*1000:.2f}mm")
            print(f"  Reward: {reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"  Success: {info['success']}")
            print(f"  Final distance: {info['distance']*1000:.2f}mm")
            break
    
    env.close()
    print("\nEnvironment test complete!")
