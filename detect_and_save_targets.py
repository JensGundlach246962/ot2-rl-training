"""
Step 1: Run CV detection and save target coordinates
"""

import numpy as np
import cv2
from coordinate_transformer import CoordinateTransformer
from cv_detection import RootTipDetector

def detect_targets(image_path, model_path):
    print("Detecting root tips...")
    
    detector = RootTipDetector(model_path)
    root_tips_pixels, mask = detector.detect_root_tips(image_path)
    print(f"Found {len(root_tips_pixels)} root tips")
    
    # Get image dimensions
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    # Transform to robot coordinates
    transformer = CoordinateTransformer()
    root_tips_robot = []
    for px, py in root_tips_pixels:
        robot_coords = transformer.pixel_to_robot(px, py, img_width, img_height)
        root_tips_robot.append(robot_coords)
        print(f"  Pixel ({px}, {py}) → Robot [{robot_coords[0]:.4f}, {robot_coords[1]:.4f}, {robot_coords[2]:.4f}]")
    
    # Save targets
    np.save("target_positions.npy", np.array(root_tips_robot))
    print(f"\nSaved {len(root_tips_robot)} targets to target_positions.npy")
    
    return root_tips_robot

if __name__ == "__main__":
    detect_targets("test_plate.png", "jens_246962_root_model_256px.h5")
