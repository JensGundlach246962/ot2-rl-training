"""
Simple test script for OT2 Gymnasium environment
Runs 1000 steps with random actions to verify environment works
"""

from ot2_gym_wrapper import OT2GymEnv
import numpy as np

def test_environment(n_steps=1000, n_episodes=3):
    """Test environment with random actions."""
    print("="*60)
    print("OT2 GYMNASIUM ENVIRONMENT TEST")
    print("="*60)
    
    env = OT2GymEnv(render_mode=None)  # No render for speed
    
    total_steps = 0
    successes = 0
    
    for episode in range(n_episodes):
        print(f"\n[Episode {episode + 1}/{n_episodes}]")
        
        observation, info = env.reset()
        print(f"Target: [{info['target'][0]:.3f}, {info['target'][1]:.3f}, {info['target'][2]:.3f}]")
        print(f"Initial distance: {info['distance']*1000:.2f}mm")
        
        episode_reward = 0
        episode_steps = 0
        
        for step in range(n_steps):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # Print progress every 200 steps
            if step % 200 == 0 and step > 0:
                print(f"  Step {step}: distance = {info['distance']*1000:.2f}mm")
            
            if terminated or truncated:
                if terminated:
                    print(f"  ✓ SUCCESS in {episode_steps} steps")
                    print(f"  Final distance: {info['distance']*1000:.2f}mm")
                    successes += 1
                else:
                    print(f"  ✗ TIMEOUT at {episode_steps} steps")
                    print(f"  Final distance: {info['distance']*1000:.2f}mm")
                
                print(f"  Total reward: {episode_reward:.2f}")
                break
    
    env.close()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total steps: {total_steps}")
    print(f"Success rate: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
    print("\nEnvironment is working correctly!")

if __name__ == "__main__":
    test_environment()
