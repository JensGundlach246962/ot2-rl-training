"""
SAC Training Script for OT-2 Controller
Using Soft Actor-Critic for better sample efficiency
"""

from clearml import Task

task = Task.init(
    project_name='OT2-RL-Control/Jens',
    task_name='OT2-SAC-2M-improved-reward'
)

task.set_packages(['numpy==1.26.4', 'clearml', 'tensorboard'])
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default", exit_process=True)

import numpy as np
print(f"NumPy version: {np.__version__}")

from ot2_gym_wrapper import OT2GymEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

CONFIG = {
    "algorithm": "SAC",
    "policy_type": "MlpPolicy",
    "total_timesteps": 2_000_000,
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "learning_starts": 10_000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
}

task.connect(CONFIG)

def make_env():
    return OT2GymEnv(render_mode=None, max_steps=1000)

env = DummyVecEnv([make_env])

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="ot2_sac"
)

model = SAC(
    CONFIG["policy_type"],
    env,
    learning_rate=CONFIG["learning_rate"],
    buffer_size=CONFIG["buffer_size"],
    learning_starts=CONFIG["learning_starts"],
    batch_size=CONFIG["batch_size"],
    tau=CONFIG["tau"],
    gamma=CONFIG["gamma"],
    train_freq=CONFIG["train_freq"],
    gradient_steps=CONFIG["gradient_steps"],
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
)

print("="*60)
print("STARTING SAC TRAINING")
print(f"Algorithm: {CONFIG['algorithm']}")
print(f"Total timesteps: {CONFIG['total_timesteps']:,}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Buffer size: {CONFIG['buffer_size']:,}")
print("="*60)

model.learn(
    total_timesteps=CONFIG["total_timesteps"],
    callback=[checkpoint_callback],
    progress_bar=True
)

model.save("ot2_sac_final")
task.upload_artifact("final_model", artifact_object="ot2_sac_final.zip")
print("\nTraining complete! Model saved.")

env.close()
