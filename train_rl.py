from clearml import Task

task = Task.init(
    project_name='OT2-RL-Control/Jens',
    task_name='OT2-PPO-no-logging-test'
)

task.set_packages(['numpy==1.26.4', 'clearml', 'tensorboard'])
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default", exit_process=True)

import numpy as np
print(f"NumPy version: {np.__version__}")

from ot2_gym_wrapper import OT2GymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

CONFIG = {
    "algorithm": "PPO",
    "policy_type": "MlpPolicy",
    "total_timesteps": 100_000,  # Just 100k for testing
    "learning_rate": 1e-3,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

task.connect(CONFIG)

def make_env():
    return OT2GymEnv(render_mode=None, max_steps=1000)

env = DummyVecEnv([make_env])

model = PPO(
    CONFIG["policy_type"],
    env,
    learning_rate=CONFIG["learning_rate"],
    n_steps=CONFIG["n_steps"],
    batch_size=CONFIG["batch_size"],
    n_epochs=CONFIG["n_epochs"],
    gamma=CONFIG["gamma"],
    gae_lambda=CONFIG["gae_lambda"],
    clip_range=CONFIG["clip_range"],
    ent_coef=CONFIG["ent_coef"],
    verbose=0,  # Disable verbose output
)

print("="*60)
print("STARTING PPO TRAINING - MINIMAL CONFIG")
print(f"Total timesteps: {CONFIG['total_timesteps']:,}")
print("="*60)

model.learn(
    total_timesteps=CONFIG["total_timesteps"],
    progress_bar=False  # Disable progress bar
)

print("Training complete!")
env.close()
