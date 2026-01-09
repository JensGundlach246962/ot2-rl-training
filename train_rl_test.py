"""
Quick test of RL training - 10k steps only
"""

from ot2_gym_wrapper import OT2GymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return OT2GymEnv(render_mode=None)

def train_test():
    print("Quick training test - 10k steps...")
    
    env = DummyVecEnv([make_env])
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        verbose=1,
    )
    
    model.learn(total_timesteps=10_000, progress_bar=True)
    
    model.save("test_model")
    print("\nTest complete! Model saved as 'test_model.zip'")
    
    env.close()

if __name__ == "__main__":
    train_test()
