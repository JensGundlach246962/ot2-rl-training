"""
Test the trained RL model - No rendering
"""

from ot2_gym_wrapper import OT2GymEnv
from stable_baselines3 import PPO
import numpy as np

# Load best model
model = PPO.load("best_model/best_model.zip")

# Test positions (same as PID test)
TEST_POSITIONS = [
    (0.05, 0.05, 0.23),
    (-0.187, -0.171, 0.170),
    (0.253, 0.22, 0.29),
    (0.0, 0.0, 0.23),
    (-0.1, 0.1, 0.25),
]

env = OT2GymEnv(render_mode=None)

print("="*60)
print("TESTING TRAINED RL MODEL")
print("="*60)

successes = 0
errors = []

for i, target in enumerate(TEST_POSITIONS, 1):
    print(f"\n[{i}/{len(TEST_POSITIONS)}] Target: {target}")
    
    obs, info = env.reset()
    env.target_position = np.array(target, dtype=np.float32)
    
    for step in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 400 == 0:
            print(f"  Step {step}: {info['distance']*1000:.2f}mm")
        
        if terminated:
            print(f"  ✓ SUCCESS in {step} steps - {info['distance']*1000:.2f}mm")
            successes += 1
            errors.append(info['distance'] * 1000)
            break
    
    if not terminated:
        print(f"  ✗ TIMEOUT - {info['distance']*1000:.2f}mm")
        errors.append(info['distance'] * 1000)

env.close()

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Success: {successes}/{len(TEST_POSITIONS)}")
if errors:
    print(f"Mean error: {np.mean(errors):.2f}mm")
    print(f"Max error: {np.max(errors):.2f}mm")
    print(f"Min error: {np.min(errors):.2f}mm")
