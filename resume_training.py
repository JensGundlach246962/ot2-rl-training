"""
Resume training from checkpoint
"""

from ot2_gym_wrapper import OT2GymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return OT2GymEnv(render_mode=None)

def resume_train():
    print("Loading checkpoint from 25k steps...")
    
    # Load the checkpoint
    model = PPO.load("checkpoints/ot2_local_25000_steps.zip")
    
    # Create new environment
    env = DummyVecEnv([make_env])
    model.set_env(env)
    
    eval_env = DummyVecEnv([make_env])
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path="./checkpoints/",
        name_prefix="ot2_resumed"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=10_000
    )
    
    print("Resuming training for 475k more steps (total 500k)...")
    print("This will take 6-8 hours")
    print("Keep terminal open or use 'screen'")
    
    model.learn(
        total_timesteps=475_000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        reset_num_timesteps=False  # Continue from 25k
    )
    
    model.save("ot2_resumed_final")
    print("Training complete! Model saved.")
    
    env.close()
    eval_env.close()

if __name__ == "__main__":
    resume_train()
