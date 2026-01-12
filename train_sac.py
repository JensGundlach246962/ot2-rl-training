from clearml import Task

task = Task.init(
    project_name='OT2-RL-Control/Jens',
    task_name='SAC-5M-improved-reward'
)

task.set_packages(['numpy==1.26.4', 'clearml'])
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default", exit_process=True)

import numpy as np
from wrapper_improved import ImprovedOT2Env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

env = ImprovedOT2Env(max_steps=500)

checkpoint_callback = CheckpointCallback(
    save_freq=250_000,
    save_path="./checkpoints/",
    name_prefix="ot2_sac"
)

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    verbose=1
)

print("="*60)
print("SAC TRAINING - 5M STEPS")
print("Entropy-regularized, should explore better")
print("="*60)

model.learn(
    total_timesteps=5_000_000,
    callback=[checkpoint_callback],
    progress_bar=True
)

model.save("ot2_sac_final")
task.upload_artifact("final_model", artifact_object="ot2_sac_final.zip")

env.close()
