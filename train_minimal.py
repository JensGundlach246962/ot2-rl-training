"""Minimal training - no frills"""
from clearml import Task

task = Task.init(
    project_name='OT2-RL-Control/Jens',
    task_name='PPO-minimal-rewrite'
)

task.set_packages(['numpy==1.26.4', 'clearml'])
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default", exit_process=True)

import numpy as np
from wrapper_minimal import MinimalOT2Env
from stable_baselines3 import PPO

print(f"NumPy: {np.__version__}")
print("Creating environment...")

env = MinimalOT2Env(max_steps=1000)

print("Creating PPO model...")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    n_steps=2048,
    batch_size=128,
    verbose=0
)

print("Starting training for 50k steps...")

try:
    model.learn(total_timesteps=50_000, progress_bar=False)
    print("SUCCESS: Completed 50k steps!")
    model.save("model_50k")
except Exception as e:
    print(f"FAILED: {e}")

env.close()
