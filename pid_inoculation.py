"""
Step 2: Load targets and execute PID control
"""

import sys
sys.path.append('..')
from sim_class import Simulation
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, dt, output_limits=(-0.5, 0.5), integral_limits=(-0.5, 0.5), integration_threshold=0.05):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.output_limits, self.integral_limits = output_limits, integral_limits
        self.integration_threshold = integration_threshold
        self.integral, self.prev_error = 0.0, None
        
    def compute(self, error):
        p_term = self.kp * error
        if abs(error) < self.integration_threshold:
            self.integral += error * self.dt
            self.integral = max(self.integral_limits[0], min(self.integral_limits[1], self.integral))
        i_term = self.ki * self.integral
        
        if self.prev_error is None:
            d_term, self.prev_error = 0.0, error
        else:
            d_term = self.kd * (error - self.prev_error) / self.dt
            self.prev_error = error
        
        output = max(self.output_limits[0], min(self.output_limits[1], p_term + i_term + d_term))
        return output
    
    def reset(self):
        self.integral, self.prev_error = 0.0, None


class ThreeAxisPIDController:
    def __init__(self, gains, dt, output_limits=(-0.5, 0.5), integral_limits=(-0.5, 0.5), integration_threshold=0.05):
        self.pid_x = PIDController(gains['x']['kp'], gains['x']['ki'], gains['x']['kd'], dt, output_limits, integral_limits, integration_threshold)
        self.pid_y = PIDController(gains['y']['kp'], gains['y']['ki'], gains['y']['kd'], dt, output_limits, integral_limits, integration_threshold)
        self.pid_z = PIDController(gains['z']['kp'], gains['z']['ki'], gains['z']['kd'], dt, output_limits, integral_limits, integration_threshold)
    
    def compute(self, target, current):
        return [self.pid_x.compute(target[0] - current[0]),
                self.pid_y.compute(target[1] - current[1]),
                self.pid_z.compute(target[2] - current[2])]
    
    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
    
    def get_euclidean_error(self, target, current):
        d = np.array(target) - np.array(current)
        return np.linalg.norm(d)


def execute_inoculation():
    # Load saved targets
    targets = np.load("target_positions.npy")
    print(f"Loaded {len(targets)} target positions")
    
    # Initialize simulation and PID
    sim = Simulation(num_agents=1, render=False)
    
    gains = {
        'x': {'kp': 2.0, 'ki': 0.1, 'kd': 0.1},
        'y': {'kp': 2.0, 'ki': 0.1, 'kd': 0.1},
        'z': {'kp': 1.5, 'ki': 0.075, 'kd': 0.075}
    }
    
    pid = ThreeAxisPIDController(gains=gains, dt=1/240, output_limits=(-0.5, 0.5),
                                  integral_limits=(-0.5, 0.5), integration_threshold=0.05)
    
    results = []
    
    for i, target in enumerate(targets, 1):
        print(f"\nTip {i}/{len(targets)}: [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]")
        
        pid.reset()
        for _ in range(10):
            sim.run([[0, 0, 0, 0]])
        
        step = 0
        while step < 2000:
            state = sim.run([[0, 0, 0, 0]])
            current = state[list(state.keys())[0]]['pipette_position']
            error = pid.get_euclidean_error(target, current)
            
            if error < 0.001:
                print(f"  ✓ Success: {step} steps, {error*1000:.2f}mm")
                results.append({'success': True, 'error_mm': error*1000, 'steps': step})
                break
            
            velocities = pid.compute(target, current)
            sim.run([velocities + [0]])
            step += 1
        else:
            print(f"  ✗ Timeout: {error*1000:.2f}mm")
            results.append({'success': False, 'error_mm': error*1000, 'steps': 2000})
    
    sim.close()
    
    # Summary
    print("\n" + "="*60)
    successes = sum(1 for r in results if r['success'])
    print(f"Success: {successes}/{len(results)}")
    if results:
        errors = [r['error_mm'] for r in results]
        print(f"Mean error: {np.mean(errors):.2f}mm")

if __name__ == "__main__":
    execute_inoculation()
