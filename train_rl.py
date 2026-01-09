"""
RL Training Script for OT-2 Controller - ClearML Remote Execution
"""

from clearml import Task
from ot2_gym_wrapper import OT2GymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback

task = Task.init(
    project_name='OT2-RL-Control/Jens',
    task_name='OT2-PPO-1M-steps-overnight'
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

CONFIG = {
    "total_timesteps": 1_000_000,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
}

def make_env():
    return OT2GymEnv(render_mode=None)

def train():
    run = wandb.init(project="ot2-rl-control", config=CONFIG, sync_tensorboard=True)
    
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path="./checkpoints/", name_prefix="ot2_ppo")
    eval_callback = EvalCallback(eval_env, best_model_save_path="./best_model/", log_path="./eval_logs/", eval_freq=10_000)
    wandb_callback = WandbCallback(model_save_path=f"models/{run.id}", verbose=2)
    
    model = PPO("MlpPolicy", env, learning_rate=CONFIG["learning_rate"], n_steps=CONFIG["n_steps"], 
                batch_size=CONFIG["batch_size"], n_epochs=CONFIG["n_epochs"], gamma=CONFIG["gamma"],
                verbose=1, tensorboard_log=f"runs/{run.id}")
    
    print("STARTING TRAINING")
    model.learn(total_timesteps=CONFIG["total_timesteps"], callback=[checkpoint_callback, eval_callback, wandb_callback], progress_bar=True)
    
    model.save("ot2_ppo_final")
    task.upload_artifact("final_model", artifact_object="ot2_ppo_final.zip")
    
    env.close()
    eval_env.close()
    run.finish()

if __name__ == "__main__":
    train()
