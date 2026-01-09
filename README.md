# Task 11: RL Controller

## Status: Ready for Training

### Files Created
- `ot2_gym_wrapper.py` - Gymnasium environment wrapper
- `test_wrapper.py` - Environment test (passes)
- `train_rl.py` - Full training script (1M steps)
- `train_rl_test.py` - Quick test (10k steps, works)
- `test_model.zip` - Proof-of-concept trained model

### Setup Complete
- PyBullet simulation files uploaded
- W&B configured and logged in
- Checkpoint directories created

### Next Steps
1. Get ClearML submission command from Kade
2. Submit `train_rl.py` to GPU queue
3. Monitor at https://wandb.ai/
4. Check results Saturday morning

### Training Config
- Algorithm: PPO
- Total steps: 1,000,000
- Learning rate: 3e-4
- Checkpoint: Every 50k steps
- Evaluation: Every 10k steps

## TODO Saturday
- [ ] Load trained model
- [ ] Test on 18 positions (like PID test)
- [ ] Compare RL vs PID performance
- [ ] Create GIF demo
- [ ] Write final README
- [ ] Coordinate with group on best model
