"""
CV Detection Pipeline
Loads trained model and detects root tips in specimen images
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model

class RootTipDetector:
    """Detects root tips using trained segmentation model."""
    
    def __init__(self, model_path, image_size=256):
        """
        Load trained root segmentation model.
        
        Args:
            model_path: Path to saved .h5 model
            image_size: Model input size (256px)
        """
        self.model = load_model(model_path)
        self.image_size = image_size
        
    def detect_root_tips(self, image_path, threshold=0.5):
        """
        Detect root tips in image.
        
        Args:
            image_path: Path to specimen plate image
            threshold: Segmentation threshold
            
        Returns:
            List of (x, y) pixel coordinates of detected root tips
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        original_height, original_width = img.shape[:2]
        
        # Resize to model input size
        img_resized = cv2.resize(img, (self.image_size, self.image_size))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_normalized = img_gray / 255.0
        img_normalized = np.expand_dims(img_normalized, axis=-1)  # Add channel dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Run segmentation
        prediction = self.model.predict(img_batch, verbose=0)[0]
        mask = (prediction > threshold).astype(np.uint8) * 255
        
        # Find contours (connected regions)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        root_tips = []
        for contour in contours:
            # Get bottommost point of each contour (root tip)
            contour = contour.squeeze()
            if len(contour.shape) == 2:  # Valid contour
                # Find point with maximum Y (bottom of root)
                tip_idx = np.argmax(contour[:, 1])
                tip_x, tip_y = contour[tip_idx]
                
                # Scale back to original image size
                tip_x_orig = int(tip_x * original_width / self.image_size)
                tip_y_orig = int(tip_y * original_height / self.image_size)
                
                root_tips.append((tip_x_orig, tip_y_orig))
        
        return root_tips, mask


if __name__ == "__main__":
    print("Testing Root Tip Detector...")
    
    detector = RootTipDetector("jens_246962_root_model_256px.h5")
    
    # Test on sample image
    root_tips, mask = detector.detect_root_tips("test_plate.png")
    
    print(f"\nDetected {len(root_tips)} root tips:")
    for i, (x, y) in enumerate(root_tips[:5], 1):
        print(f"  Tip {i}: pixel ({x}, {y})")
    
    # Save visualization
    cv2.imwrite("debug_mask.png", mask)
    print(f"\nSegmentation mask saved to: debug_mask.png")
