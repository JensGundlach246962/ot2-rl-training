from clearml import Task

task = Task.init(
    project_name='OT2-RL-Control/Jens',
    task_name='OT2-PPO-3M-improved-reward'
)

task.set_packages(['numpy==1.26.4', 'clearml', 'tensorboard'])
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default", exit_process=True)

import numpy as np
print(f"NumPy version: {np.__version__}")

from ot2_gym_wrapper import OT2GymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

CONFIG = {
    "algorithm": "PPO",
    "policy_type": "MlpPolicy",
    "total_timesteps": 3_000_000,
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

checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path="./checkpoints/",
    name_prefix="ot2_ppo"
)

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
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
)

print("="*60)
print("STARTING PPO TRAINING")
print(f"Total timesteps: {CONFIG['total_timesteps']:,}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print("="*60)

model.learn(
    total_timesteps=CONFIG["total_timesteps"],
    callback=[checkpoint_callback],
    progress_bar=True
)

model.save("ot2_ppo_final")
task.upload_artifact("final_model", artifact_object="ot2_ppo_final.zip")
print("Training complete!")

env.close()
