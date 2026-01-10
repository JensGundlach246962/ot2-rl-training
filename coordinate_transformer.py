"""
Coordinate Transformation: Pixel Space → Robot World Space
"""

import numpy as np
import cv2

class CoordinateTransformer:
    """Transforms between image pixel coordinates and robot world coordinates."""
    
    def __init__(self, plate_width_m=0.08, plate_height_m=0.08,
                 plate_center_robot=[0.0, 0.0], plate_z=0.23):  # CHANGED from 0.11 to 0.23
        """
        Initialize coordinate transformer.
        
        plate_z: Height of plate surface - must be within robot workspace (0.170-0.290m)
        """
        self.plate_width_m = plate_width_m
        self.plate_height_m = plate_height_m
        self.plate_center_robot = np.array(plate_center_robot)
        self.plate_z = plate_z
        
    def pixel_to_robot(self, pixel_x, pixel_y, image_width, image_height):
        """Convert pixel coordinates to robot world coordinates."""
        scale_x = self.plate_width_m / image_width
        scale_y = self.plate_height_m / image_height
        
        centered_x = pixel_x - (image_width / 2)
        centered_y = pixel_y - (image_height / 2)
        
        robot_x = centered_x * scale_x + self.plate_center_robot[0]
        robot_y = -centered_y * scale_y + self.plate_center_robot[1]
        robot_z = self.plate_z
        
        return [robot_x, robot_y, robot_z]


if __name__ == "__main__":
    print("Testing Coordinate Transformer...")
    
    transformer = CoordinateTransformer()
    
    robot_coords = transformer.pixel_to_robot(512, 512, 1024, 1024)
    print(f"Image center (512, 512) in 1024x1024 → Robot: {robot_coords}")
    
    robot_coords = transformer.pixel_to_robot(1592, 1391, 3184, 2782)
    print(f"Image center (1592, 1391) in 3184x2782 → Robot: {robot_coords}")
