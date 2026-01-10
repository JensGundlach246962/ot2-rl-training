"""
RL Training Script for OT2 Controller - ClearML Remote Execution
"""

from ot2_gym_wrapper import OT2GymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from clearml import Task

# Initialize ClearML task FIRST
task = Task.init(
    project_name='OT2-RL-Control/Jens',
    task_name='OT2-PPO-1M-steps-v2'
)

# Set docker image
task.set_base_docker('deanis/2023y2b-rl:latest')

# Execute remotely on default queue
task.execute_remotely(queue_name="default")

# Configuration
CONFIG = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1_000_000,
    "env_name": "OT2-v0",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

def make_env():
    """Create environment instance."""
    return OT2GymEnv(render_mode=None)

def train():
    """Train RL agent."""
    
    # Initialize Weights & Biases
    run = wandb.init(
        project="ot2-rl-control",
        config=CONFIG,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    
    # Create environment
    env = DummyVecEnv([make_env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints/",
        name_prefix="ot2_ppo"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )
    
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
    
    # Create PPO agent
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
        tensorboard_log=f"runs/{run.id}",
    )
    
    print("="*60)
    print("STARTING REMOTE TRAINING")
    print("="*60)
    print(f"Total timesteps: {CONFIG['total_timesteps']:,}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    
    # Train
    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=[checkpoint_callback, eval_callback, wandb_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save("ot2_ppo_final")
    print("\nTraining complete! Model saved as 'ot2_ppo_final.zip'")
    
    # Upload model to ClearML
    task.upload_artifact("final_model", artifact_object="ot2_ppo_final.zip")
    print("Model uploaded to ClearML!")
    
    # Close
    env.close()
    eval_env.close()
    run.finish()

if __name__ == "__main__":
    train()
