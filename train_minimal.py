"""
Minimal memory training - reduced batch size
"""

from ot2_gym_wrapper import OT2GymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def train():
    print("Training with minimal memory config...")
    
    # Single environment (not vectorized)
    env = OT2GymEnv(render_mode=None)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/",
        name_prefix="ot2_minimal"
    )
    
    # Reduced parameters for lower memory usage
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,        # Reduced from 2048
        batch_size=32,      # Reduced from 64
        verbose=1,
        device='cpu'        # Force CPU to avoid GPU memory
    )
    
    print("Training 100k steps (test run)...")
    model.learn(
        total_timesteps=100_000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    model.save("ot2_minimal_final")
    print("Complete!")
    
    env.close()

if __name__ == "__main__":
    train()
