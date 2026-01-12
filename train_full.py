"""Full 3M training - minimal version that works"""
from clearml import Task

task = Task.init(
    project_name='OT2-RL-Control/Jens',
    task_name='PPO-3M-minimal-FINAL'
)

task.set_packages(['numpy==1.26.4', 'clearml'])
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default", exit_process=True)

import numpy as np
from wrapper_minimal import MinimalOT2Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

print(f"NumPy: {np.__version__}")

env = MinimalOT2Env(max_steps=1000)

# Add checkpoints now that we know it works
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="ot2_ppo"
)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1  # Can enable now
)

print("="*60)
print("STARTING 3M STEP TRAINING")
print("="*60)

model.learn(
    total_timesteps=3_000_000,
    callback=[checkpoint_callback],
    progress_bar=True  # Can enable now
)

model.save("ot2_ppo_final")
task.upload_artifact("final_model", artifact_object="ot2_ppo_final.zip")

print("Training complete!")
env.close()
