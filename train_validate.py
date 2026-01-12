"""Quick validation - 100k steps to test improvements"""
from clearml import Task

task = Task.init(
    project_name='OT2-RL-Control/Jens',
    task_name='PPO-100k-validation-improved-reward'
)

task.set_packages(['numpy==1.26.4', 'clearml'])
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default", exit_process=True)

import numpy as np
from wrapper_improved import ImprovedOT2Env
from stable_baselines3 import PPO

print(f"NumPy: {np.__version__}")

env = ImprovedOT2Env(max_steps=500)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1
)

print("="*60)
print("VALIDATION RUN - 100k steps")
print("Testing improved reward function")
print("="*60)

model.learn(
    total_timesteps=100_000,
    progress_bar=True
)

print("\nVALIDATION COMPLETE!")
env.close()
