#!/usr/bin/env python3
"""
Debug script to test dashed line detection directly
"""
import sys
sys.path.append('.')

import cv2
import numpy as np
from png2svg.detect_primitives import detect_lines, is_dashed_line, LineSeg
from png2svg.config import load_config

def debug_dashed_detection():
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
    
    print(f"Image loaded: {img.shape}")
    
    # Detect lines
    lines = detect_lines(img, img_bw, cfg)
    print(f"Detected {len(lines)} lines")
    
    # Test each line for dashed pattern
    dashed_count = 0
    for i, line in enumerate(lines):
        is_dashed = is_dashed_line(line, img_bw, cfg)
        if is_dashed:
            print(f"Line {i}: DASHED - {line.p1} to {line.p2}")
            dashed_count += 1
        else:
            length = np.hypot(line.p2[0] - line.p1[0], line.p2[1] - line.p1[1])
            if length > 50:  # Only show longer lines
                print(f"Line {i}: solid  - {line.p1} to {line.p2} (length: {length:.1f})")
    
    print(f"\nTotal dashed lines: {dashed_count}")
    
    # Check configuration
    print(f"\nDashed line config:")
    if hasattr(cfg.algorithms, 'dashed_line'):
        dl = cfg.algorithms.dashed_line
        print(f"  min_dash_count: {dl.min_dash_count}")
        print(f"  on_ratio_min: {dl.on_ratio_min}")
        print(f"  on_ratio_max: {dl.on_ratio_max}")
        print(f"  min_gap_length: {dl.min_gap_length}")
        print(f"  dash_gap_ratio_min: {dl.dash_gap_ratio_min}")
        print(f"  dash_gap_ratio_max: {dl.dash_gap_ratio_max}")
    else:
        print("  No dashed_line config found!")

if __name__ == "__main__":
    debug_dashed_detection()