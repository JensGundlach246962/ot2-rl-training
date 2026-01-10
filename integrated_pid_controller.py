"""
Integrated PID Controller System
Combines CV detection, coordinate transformation, and PID control
"""

import sys
sys.path.append('..')
from sim_class import Simulation
import numpy as np
import cv2
from coordinate_transformer import CoordinateTransformer
from cv_detection import RootTipDetector

# Import PID from Task 10
class PIDController:
    def __init__(self, kp, ki, kd, dt, output_limits=(-0.5, 0.5), integral_limits=(-0.5, 0.5), integration_threshold=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.integration_threshold = integration_threshold
        self.integral = 0.0
        self.prev_error = None
        
    def compute(self, error):
        p_term = self.kp * error
        
        integration_active = abs(error) < self.integration_threshold
        if integration_active:
            self.integral += error * self.dt
            self.integral = max(self.integral_limits[0], min(self.integral_limits[1], self.integral))
        
        i_term = self.ki * self.integral
        
        if self.prev_error is None:
            d_term = 0.0
            self.prev_error = error
        else:
            derivative = (error - self.prev_error) / self.dt
            d_term = self.kd * derivative
            self.prev_error = error
        
        output = p_term + i_term + d_term
        output = max(self.output_limits[0], min(self.output_limits[1], output))
        
        return output
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = None


class ThreeAxisPIDController:
    def __init__(self, gains, dt, output_limits=(-0.5, 0.5), integral_limits=(-0.5, 0.5), integration_threshold=0.05):
        self.dt = dt
        self.pid_x = PIDController(gains['x']['kp'], gains['x']['ki'], gains['x']['kd'], dt, output_limits, integral_limits, integration_threshold)
        self.pid_y = PIDController(gains['y']['kp'], gains['y']['ki'], gains['y']['kd'], dt, output_limits, integral_limits, integration_threshold)
        self.pid_z = PIDController(gains['z']['kp'], gains['z']['ki'], gains['z']['kd'], dt, output_limits, integral_limits, integration_threshold)
    
    def compute(self, target, current):
        error_x = target[0] - current[0]
        error_y = target[1] - current[1]
        error_z = target[2] - current[2]
        
        vel_x = self.pid_x.compute(error_x)
        vel_y = self.pid_y.compute(error_y)
        vel_z = self.pid_z.compute(error_z)
        
        return [vel_x, vel_y, vel_z]
    
    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
    
    def get_euclidean_error(self, target, current):
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        dz = target[2] - current[2]
        return (dx**2 + dy**2 + dz**2)**0.5


def autonomous_inoculation_pid(image_path, model_path):
    """
    Complete autonomous pipeline: CV detection → coordinate transform → PID control
    """
    print("="*60)
    print("AUTONOMOUS INOCULATION SYSTEM - PID CONTROLLER")
    print("="*60)
    
    # Step 1: Detect root tips
    print("\n[1/4] Detecting root tips...")
    detector = RootTipDetector(model_path)
    root_tips_pixels, mask = detector.detect_root_tips(image_path)
    print(f"Found {len(root_tips_pixels)} root tips")
    
    # Get image dimensions
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    print(f"Image dimensions: {img_width}×{img_height} pixels")
    
    # Step 2: Transform to robot coordinates
    print("\n[2/4] Transforming coordinates...")
    transformer = CoordinateTransformer()
    root_tips_robot = []
    for i, (px, py) in enumerate(root_tips_pixels, 1):
        robot_coords = transformer.pixel_to_robot(px, py, img_width, img_height)
        root_tips_robot.append(robot_coords)
        print(f"  Tip {i}: pixel ({px}, {py}) → robot [{robot_coords[0]:.4f}, {robot_coords[1]:.4f}, {robot_coords[2]:.4f}]")
    
    # Step 3: Initialize simulation and PID
    print("\n[3/4] Initializing robot simulation...")
    sim = Simulation(num_agents=1, render=False)
    
    # Best PID gains from Task 10
    gains = {
        'x': {'kp': 2.0, 'ki': 0.1, 'kd': 0.1},
        'y': {'kp': 2.0, 'ki': 0.1, 'kd': 0.1},
        'z': {'kp': 1.5, 'ki': 0.075, 'kd': 0.075}
    }
    
    pid = ThreeAxisPIDController(
        gains=gains, dt=1/240,
        output_limits=(-0.5, 0.5),
        integral_limits=(-0.5, 0.5),
        integration_threshold=0.05
    )
    
    # Step 4: Inoculate each root tip
    print("\n[4/4] Autonomous inoculation...")
    results = []
    
    for i, target in enumerate(root_tips_robot, 1):
        print(f"\nInoculating tip {i}/{len(root_tips_robot)}: [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]")
        
        pid.reset()
        for _ in range(10):
            sim.run([[0, 0, 0, 0]])
        
        # Move to position
        step = 0
        while step < 2000:
            state = sim.run([[0, 0, 0, 0]])
            robot_key = list(state.keys())[0]
            current = state[robot_key]['pipette_position']
            
            error = pid.get_euclidean_error(target, current)
            
            if error < 0.001:
                print(f"  ✓ Reached tip in {step} steps - {error*1000:.2f}mm error")
                results.append({'tip': i, 'success': True, 'error_mm': error*1000, 'steps': step})
                break
            
            velocities = pid.compute(target, current)
            sim.run([velocities + [0]])
            step += 1
        else:
            print(f"  ✗ Timeout - {error*1000:.2f}mm error")
            results.append({'tip': i, 'success': False, 'error_mm': error*1000, 'steps': 2000})
    
    sim.close()
    
    # Summary
    print("\n" + "="*60)
    print("INOCULATION SUMMARY")
    print("="*60)
    successes = sum(1 for r in results if r['success'])
    print(f"Success rate: {successes}/{len(results)}")
    if results:
        errors = [r['error_mm'] for r in results]
        print(f"Mean error: {np.mean(errors):.2f}mm")
        print(f"Max error: {np.max(errors):.2f}mm")
    
    return results


if __name__ == "__main__":
    results = autonomous_inoculation_pid(
        image_path="test_plate.png",
        model_path="jens_246962_root_model_256px.h5"
    )
