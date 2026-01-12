"""Improved RL training - better reward function, 5M steps"""
from clearml import Task

task = Task.init(
    project_name='OT2-RL-Control/Jens',
    task_name='PPO-5M-improved-reward-v2'
)

task.set_packages(['numpy==1.26.4', 'clearml'])
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default", exit_process=True)

import numpy as np
from wrapper_improved import ImprovedOT2Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

print(f"NumPy: {np.__version__}")

# Environment with shorter episodes
env = ImprovedOT2Env(max_steps=500)  # Reduced from 1000

# Checkpoints every 250k steps
checkpoint_callback = CheckpointCallback(
    save_freq=250_000,
    save_path="./checkpoints/",
    name_prefix="ot2_ppo_improved"
)

# PPO with adjusted hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,      # Standard (was 1e-3, back to default)
    n_steps=2048,
    batch_size=64,           # Smaller batches (was 128)
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1
)

print("="*60)
print("IMPROVED TRAINING - 5M STEPS")
print("Max episode length: 500 (reduced)")
print("Batch size: 64 (reduced for more updates)")
print("Better reward scaling in wrapper")
print("="*60)

model.learn(
    total_timesteps=5_000_000,
    callback=[checkpoint_callback],
    progress_bar=True
)

model.save("ot2_ppo_improved_final")
task.upload_artifact("final_model", artifact_object="ot2_ppo_improved_final.zip")

print("Training complete!")
env.close()
