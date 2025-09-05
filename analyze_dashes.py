import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_dashed_patterns(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Preprocess
    img_bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
    
    # Detect all lines first
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD, 0.8, 0.6, 2.0, 22.5, 0.0, 0.7, 1024)
    lines = lsd.detect(img_bw)[0]
    
    print(f"Found {len(lines) if lines is not None else 0} line segments")
    
    if lines is None:
        return
        
    # Sample each line to find potential dashes
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Skip very short lines
        if length < 30:
            continue
            
        # Sample pixels along the line
        num_samples = int(length)
        if num_samples < 10:
            continue
            
        x_coords = np.linspace(x1, x2, num_samples)
        y_coords = np.linspace(y1, y2, num_samples)
        
        # Get pixel values (foreground = True/255, background = False/0)
        values = []
        for x, y in zip(x_coords, y_coords):
            x, y = int(round(x)), int(round(y))
            if 0 <= x < img_bw.shape[1] and 0 <= y < img_bw.shape[0]:
                values.append(img_bw[y, x] > 0)
        
        if not values:
            continue
            
        # Analyze run-length encoding
        runs = []
        current_value = values[0]
        current_length = 1
        
        for val in values[1:]:
            if val == current_value:
                current_length += 1
            else:
                runs.append((current_value, current_length))
                current_value = val
                current_length = 1
        runs.append((current_value, current_length))
        
        # Separate ON and OFF runs
        on_runs = [length for is_on, length in runs if is_on]
        off_runs = [length for is_on, length in runs if not is_on]
        
        # Calculate statistics
        total_on = sum(on_runs)
        total_length = len(values)
        on_ratio = total_on / total_length if total_length > 0 else 0
        
        # Show detailed analysis for potentially dashed lines
        if len(on_runs) > 1 and len(off_runs) > 0:
            print(f"\nLine {i}: length={length:.1f}, samples={len(values)}")
            print(f"  ON runs: {on_runs} (count={len(on_runs)})")
            print(f"  OFF runs: {off_runs} (count={len(off_runs)})")
            print(f"  ON ratio: {on_ratio:.3f}")
            
            # Check if this could be a dashed line
            avg_on = np.mean(on_runs) if on_runs else 0
            avg_off = np.mean(off_runs) if off_runs else 0
            cv_on = np.std(on_runs) / avg_on if avg_on > 0 else float('inf')
            cv_off = np.std(off_runs) / avg_off if avg_off > 0 else float('inf')
            
            print(f"  Avg ON: {avg_on:.1f}, CV: {cv_on:.3f}")
            print(f"  Avg OFF: {avg_off:.1f}, CV: {cv_off:.3f}")
            print(f"  Dash/gap ratio: {avg_on/avg_off if avg_off > 0 else float('inf'):.3f}")
            
            # Could this be dashed?
            is_candidate = (
                len(on_runs) >= 2 and 
                len(off_runs) >= 1 and
                0.2 <= on_ratio <= 0.8 and
                cv_on <= 2.0 and cv_off <= 2.0
            )
            print(f"  Potential dash: {is_candidate}")

if __name__ == "__main__":
    analyze_dashed_patterns(r"D:\Projects\png2svg\pngs\test.png")