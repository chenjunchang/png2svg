#!/usr/bin/env python3
"""
Very detailed debug script to understand why dashed lines aren't detected
"""
import sys
sys.path.append('.')

import cv2
import numpy as np
import math
from png2svg.detect_primitives import detect_lines, get_run_lengths, LineSeg
from png2svg.config import load_config

def debug_single_line_detail(line: LineSeg, img_bw: np.ndarray, line_idx: int):
    """Debug a single line in detail"""
    x1, y1 = line.p1
    x2, y2 = line.p2
    
    length = math.hypot(x2 - x1, y2 - y1)
    print(f"\n=== LINE {line_idx} DEBUG ===")
    print(f"  Coordinates: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})")
    print(f"  Length: {length:.1f}")
    
    if length < 20:
        print("  SKIP: Too short")
        return False
    
    # Adaptive sampling
    num_samples = max(int(length * 0.8), 20)
    num_samples = min(num_samples, 200)
    print(f"  Samples: {num_samples}")
    
    # Sample points
    x_samples = np.linspace(x1, x2, num_samples)
    y_samples = np.linspace(y1, y2, num_samples)
    
    # Clamp coordinates
    h, w = img_bw.shape
    x_samples = np.clip(x_samples, 0, w - 1).astype(int)
    y_samples = np.clip(y_samples, 0, h - 1).astype(int)
    
    # Sample pixel values
    values = []
    for x, y in zip(x_samples, y_samples):
        values.append(img_bw[y, x] > 0)
    
    values = np.array(values)
    
    # Show first 20 values for insight
    sample_str = ''.join(['█' if v else '·' for v in values[:20]])
    print(f"  First 20 pixels: {sample_str}")
    
    # Apply light smoothing
    if len(values) >= 5:
        smoothed = np.zeros_like(values)
        for i in range(len(values)):
            start = max(0, i-1)
            end = min(len(values), i+2)
            smoothed[i] = np.median(values[start:end]) > 0.5
        values = smoothed
        
        sample_str_smoothed = ''.join(['█' if v else '·' for v in values[:20]])
        print(f"  After smoothing: {sample_str_smoothed}")
    
    # Analyze runs
    runs = get_run_lengths(values)
    if not runs:
        print("  FAIL: No runs found")
        return False
    
    print(f"  Runs: {runs}")
    
    # Separate ON and OFF runs
    on_runs = [run_len for is_on, run_len in runs if is_on]
    off_runs = [run_len for is_on, run_len in runs if not is_on]
    
    print(f"  ON runs: {on_runs} (count={len(on_runs)})")
    print(f"  OFF runs: {off_runs} (count={len(off_runs)})")
    
    # Check criteria one by one
    if len(on_runs) < 1 or len(off_runs) < 1:
        print("  FAIL: Not enough ON or OFF runs")
        return False
    
    mean_on_length = np.mean(on_runs)
    if mean_on_length < 1.5:
        print(f"  FAIL: Mean ON length too small ({mean_on_length:.1f} < 1.5)")
        return False
    
    mean_off_length = np.mean(off_runs)
    if mean_off_length < 1.0:
        print(f"  FAIL: Mean OFF length too small ({mean_off_length:.1f} < 1.0)")
        return False
    
    total_on_length = sum(on_runs)
    on_ratio = total_on_length / len(values)
    print(f"  ON ratio: {on_ratio:.3f}")
    
    if on_ratio < 0.30 or on_ratio > 0.85:
        print(f"  FAIL: ON ratio out of range ({on_ratio:.3f} not in 0.30-0.85)")
        return False
    
    # Check pattern regularity
    if len(on_runs) >= 3:
        on_std = np.std(on_runs)
        on_cv = on_std / mean_on_length if mean_on_length > 0 else float('inf')
        if on_cv > 2.0:
            print(f"  FAIL: ON regularity too poor (CV={on_cv:.3f} > 2.0)")
            return False
        print(f"  ON regularity OK (CV={on_cv:.3f})")
    
    # Check gap regularity  
    if len(off_runs) >= 2:
        off_std = np.std(off_runs)
        off_cv = off_std / mean_off_length if mean_off_length > 0 else float('inf')
        if off_cv > 2.0:
            print(f"  FAIL: OFF regularity too poor (CV={off_cv:.3f} > 2.0)")
            return False
        print(f"  OFF regularity OK (CV={off_cv:.3f})")
    
    # Final check: dash/gap ratio
    dash_gap_ratio = mean_on_length / mean_off_length
    print(f"  Dash/gap ratio: {dash_gap_ratio:.3f}")
    if dash_gap_ratio < 0.5 or dash_gap_ratio > 10.0:
        print(f"  FAIL: Dash/gap ratio out of range ({dash_gap_ratio:.3f} not in 0.5-10.0)")
        return False
    
    print("  SUCCESS: All criteria passed - this is a dashed line!")
    return True

def debug_detailed():
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
    
    # Test a few promising lines in detail
    dashed_count = 0
    for i, line in enumerate(lines):
        length = np.hypot(line.p2[0] - line.p1[0], line.p2[1] - line.p1[1])
        if length > 50:  # Only debug longer lines
            if debug_single_line_detail(line, img_bw, i):
                dashed_count += 1
            if i >= 5:  # Limit detailed debug to first few lines
                break
    
    print(f"\n*** Total dashed lines found: {dashed_count} ***")

if __name__ == "__main__":
    debug_detailed()