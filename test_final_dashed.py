#!/usr/bin/env python3
"""
Final test of dashed line detection with manual function call
"""
import sys
sys.path.append('.')

import cv2
import numpy as np
import math
from png2svg.detect_primitives import LineSeg, is_dashed_line
from png2svg.config import load_config

def test_manual_dashed():
    # Load configuration 
    cfg = load_config('config.example.yaml')
    
    # Load and preprocess image
    img = cv2.imread('pngs/test.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load image")
        return
        
    # Preprocess to binary
    img_bw = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
    )
    
    print("Testing with new configuration values:")
    print(f"  on_ratio_min: {cfg.algorithms.dashed_line.on_ratio_min}")
    print(f"  on_ratio_max: {cfg.algorithms.dashed_line.on_ratio_max}")
    print(f"  dash_gap_ratio_min: {cfg.algorithms.dashed_line.dash_gap_ratio_min}")
    
    # Create test lines from the debug output
    test_lines = [
        # Line 2: potential dashed with ON ratio 0.215  
        LineSeg(p1=(282.2, 33.1), p2=(282.2, 230.6)),
        # Line 5: potential dashed with ON ratio 0.057
        LineSeg(p1=(28.3, 231.2), p2=(167.4, 122.5))
    ]
    
    dashed_count = 0
    for i, line in enumerate(test_lines):
        result = is_dashed_line(line, img_bw, cfg)
        print(f"Test line {i}: {'DASHED' if result else 'solid'}")
        if result:
            dashed_count += 1
    
    print(f"\nTotal dashed: {dashed_count}")

if __name__ == "__main__":
    test_manual_dashed()