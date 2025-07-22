#!/usr/bin/env python3
"""
Create test images with various quality issues for testing the image quality assessment system.
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFilter

def create_test_images():
    """Create a set of test images with different quality issues"""
    
    # Create test_images directory
    os.makedirs('test_images', exist_ok=True)
    
    # Image dimensions
    width, height = 800, 600
    
    print("Creating test images...")
    
    # 1. Good quality image
    print("- Creating good quality image...")
    good_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a nice gradient with some patterns
    for y in range(height):
        for x in range(width):
            good_img[y, x] = [
                int(128 + 100 * np.sin(x * 0.01)),
                int(128 + 100 * np.cos(y * 0.01)), 
                int(128 + 50 * np.sin((x + y) * 0.005))
            ]
    
    # Add some geometric shapes for detail
    cv2.rectangle(good_img, (100, 100), (200, 200), (255, 255, 255), 2)
    cv2.circle(good_img, (400, 300), 50, (255, 0, 0), 3)
    cv2.line(good_img, (0, 0), (width, height), (0, 255, 0), 2)
    
    cv2.imwrite('test_images/good_quality.jpg', good_img)
    
    # 2. Blurry image
    print("- Creating blurry image...")
    blurry_img = good_img.copy()
    # Apply strong Gaussian blur
    blurry_img = cv2.GaussianBlur(blurry_img, (21, 21), 10)
    cv2.imwrite('test_images/blurry_image.jpg', blurry_img)
    
    # 3. Overexposed image
    print("- Creating overexposed image...")
    overexposed_img = np.clip(good_img.astype(np.float32) * 1.8 + 80, 0, 255).astype(np.uint8)
    cv2.imwrite('test_images/overexposed_image.jpg', overexposed_img)
    
    # 4. Underexposed image
    print("- Creating underexposed image...")
    underexposed_img = np.clip(good_img.astype(np.float32) * 0.3 - 30, 0, 255).astype(np.uint8)
    cv2.imwrite('test_images/underexposed_image.jpg', underexposed_img)
    
    # 5. Streaky image
    print("- Creating streaky image...")
    streaky_img = good_img.copy()
    
    # Add horizontal streaks
    for y in range(0, height, 20):
        cv2.line(streaky_img, (0, y), (width, y), (255, 255, 255), 3)
    
    # Add some vertical streaks
    for x in range(0, width, 40):
        cv2.line(streaky_img, (x, 0), (x, height), (200, 200, 200), 2)
        
    cv2.imwrite('test_images/streaky_image.jpg', streaky_img)
    
    # 6. Multiple issues image (blurry + overexposed)
    print("- Creating image with multiple issues...")
    multi_issue_img = np.clip(good_img.astype(np.float32) * 1.6 + 60, 0, 255).astype(np.uint8)
    multi_issue_img = cv2.GaussianBlur(multi_issue_img, (15, 15), 5)
    cv2.imwrite('test_images/multiple_issues.jpg', multi_issue_img)
    
    # 7. Create a colorful pattern image
    print("- Creating colorful pattern image...")
    pattern_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create checkerboard pattern
    square_size = 50
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if (x // square_size + y // square_size) % 2 == 0:
                color = (255, 100, 50)
            else:
                color = (50, 150, 255)
            
            cv2.rectangle(pattern_img, (x, y), 
                         (min(x + square_size, width), min(y + square_size, height)), 
                         color, -1)
    
    cv2.imwrite('test_images/pattern_image.jpg', pattern_img)
    
    # 8. Very dark image (extreme underexposure)
    print("- Creating very dark image...")
    very_dark_img = np.clip(good_img.astype(np.float32) * 0.1, 0, 255).astype(np.uint8)
    cv2.imwrite('test_images/very_dark.jpg', very_dark_img)
    
    print("\n✅ Test images created successfully!")
    print("📁 Check the 'test_images' folder for 8 sample images with different quality issues:")
    print("   1. good_quality.jpg - High quality reference image")
    print("   2. blurry_image.jpg - Blurred image")
    print("   3. overexposed_image.jpg - Too bright/overexposed")
    print("   4. underexposed_image.jpg - Too dark/underexposed")
    print("   5. streaky_image.jpg - Contains streak artifacts")
    print("   6. multiple_issues.jpg - Blurry and overexposed")
    print("   7. pattern_image.jpg - Colorful test pattern")
    print("   8. very_dark.jpg - Extremely underexposed")
    print("\n🧪 Upload these images to test your quality assessment system!")

if __name__ == "__main__":
    create_test_images()