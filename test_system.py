#!/usr/bin/env python3
"""
System test script for PNG2SVG.
Creates test images and validates the complete pipeline.
"""

import numpy as np
import cv2
import os
import sys
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def create_test_image_simple_geometry():
    """Create a simple geometric test image."""
    # Create white background
    img = Image.new('RGB', (400, 300), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a triangle
    triangle = [(100, 50), (50, 150), (150, 150)]
    draw.polygon(triangle, outline='black', width=2)
    
    # Draw a circle
    draw.ellipse([200, 50, 300, 150], outline='black', width=2)
    
    # Draw some lines
    draw.line([50, 200, 150, 200], fill='black', width=2)  # horizontal line
    draw.line([200, 180, 300, 250], fill='black', width=2)  # diagonal line
    
    # Add labels
    try:
        font = ImageFont.load_default()
        draw.text((75, 160), "A", fill='black', font=font)
        draw.text((25, 160), "B", fill='black', font=font)
        draw.text((125, 160), "C", fill='black', font=font)
        draw.text((240, 100), "O", fill='black', font=font)
    except:
        # Fallback if font fails
        pass
    
    return img


def create_test_image_with_dashes():
    """Create test image with dashed lines."""
    img = Image.new('RGB', (400, 300), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw solid line
    draw.line([50, 50, 350, 50], fill='black', width=2)
    
    # Simulate dashed line by drawing segments
    for x in range(50, 350, 20):
        if (x - 50) // 20 % 2 == 0:  # Every other segment
            draw.line([x, 100, min(x + 10, 350), 100], fill='black', width=2)
    
    # Draw rectangle
    draw.rectangle([100, 150, 300, 250], outline='black', width=2)
    
    return img


def create_test_images():
    """Create all test images."""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create simple geometry test
    img1 = create_test_image_simple_geometry()
    img1.save(test_dir / "simple_geometry.png")
    
    # Create dashed line test
    img2 = create_test_image_with_dashes()
    img2.save(test_dir / "dashed_lines.png")
    
    print(f"Created test images in {test_dir}")
    return test_dir


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import png2svg
        print("[+] Main package import successful")
        
        from png2svg import Config, load_config, process_image
        print("[+] Core functions import successful")
        
        from png2svg.config import validate_dependencies
        print("[+] Config module import successful")
        
        # Test version info
        print(f"[+] PNG2SVG version: {png2svg.__version__}")
        
        return True
    except Exception as e:
        print(f"[X] Import failed: {e}")
        return False


def test_dependency_check():
    """Test dependency checking."""
    print("\nTesting dependency check...")
    
    try:
        import png2svg
        deps = png2svg.check_dependencies(verbose=True)
        
        # Check core dependencies
        core_deps = deps['core']
        missing_core = [dep for dep, available in core_deps.items() if not available]
        
        if missing_core:
            print(f"[!] Missing core dependencies: {missing_core}")
        else:
            print("[+] All core dependencies available")
        
        return len(missing_core) == 0
    except Exception as e:
        print(f"[X] Dependency check failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from png2svg import load_config
        
        # Test default config
        cfg = load_config()
        print("[+] Default config loaded")
        
        # Test with overrides
        overrides = {
            'input_dir': './test_images',
            'output_dir': './test_output',
            'jobs': 1
        }
        cfg = load_config(cli_overrides=overrides)
        print("[+] Config with overrides loaded")
        
        print(f"  Input dir: {cfg.input_dir}")
        print(f"  Output dir: {cfg.output_dir}")
        print(f"  Jobs: {cfg.jobs}")
        
        return True
    except Exception as e:
        print(f"[X] Config loading failed: {e}")
        return False


def test_image_processing():
    """Test the complete image processing pipeline."""
    print("\nTesting image processing pipeline...")
    
    try:
        # Create test images
        test_dir = create_test_images()
        
        # Load config
        from png2svg import load_config, process_image
        
        overrides = {
            'input_dir': str(test_dir),
            'output_dir': './test_output',
            'jobs': 1,
            'use_yolo_symbols': False,  # Disable to avoid model dependency
            'use_paddle_ocr': False,    # Use tesseract fallback
            'apply_constraint_solver': False,  # Disable to avoid scipy issues if missing
            'log_level': 'INFO'
        }
        
        cfg = load_config(cli_overrides=overrides)
        
        # Process test image
        test_image = str(test_dir / "simple_geometry.png")
        print(f"Processing: {test_image}")
        
        result = process_image(test_image, cfg)
        
        if result['success']:
            print("[+] Image processing successful")
            print(f"  SVG: {result['svg_path']}")
            print(f"  GeoJSON: {result['geo_path']}")
            
            # Check if files exist
            if result['svg_path'] and Path(result['svg_path']).exists():
                print("[+] SVG file created")
            else:
                print("[X] SVG file not found")
                
            if result['geo_path'] and Path(result['geo_path']).exists():
                print("[+] GeoJSON file created")
            else:
                print("[X] GeoJSON file not found")
            
            # Print stats
            stats = result.get('stats', {})
            print(f"  Processing time: {stats.get('processing_time', 0):.2f}s")
            print(f"  Nodes: {stats.get('total_nodes', 0)}")
            print(f"  Edges: {stats.get('total_edges', 0)}")
            print(f"  Relations: {stats.get('total_relations', 0)}")
            
            return True
        else:
            print(f"[X] Image processing failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"[X] Image processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_interface():
    """Test CLI interface."""
    print("\nTesting CLI interface...")
    
    try:
        # Test CLI argument parsing
        import png2svg.cli as cli
        
        # Create a minimal test
        parser = cli.create_argument_parser()
        print("[+] CLI parser created")
        
        # Test help
        try:
            args = parser.parse_args(['--help'])
        except SystemExit:
            print("[+] CLI help works")
        
        return True
    except Exception as e:
        print(f"[X] CLI test failed: {e}")
        return False


def validate_output_files():
    """Validate generated output files."""
    print("\nValidating output files...")
    
    output_dir = Path("./test_output")
    if not output_dir.exists():
        print("[X] Output directory not found")
        return False
    
    svg_files = list(output_dir.glob("*.svg"))
    geojson_files = list(output_dir.glob("*.geo.json"))
    
    print(f"Found {len(svg_files)} SVG files and {len(geojson_files)} GeoJSON files")
    
    if len(svg_files) == 0:
        print("[X] No SVG files generated")
        return False
    
    # Check SVG content
    for svg_file in svg_files:
        with open(svg_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if '<svg' not in content:
            print(f"[X] Invalid SVG file: {svg_file}")
            return False
        
        if 'data-id' not in content:
            print(f"[!] SVG file missing semantic data: {svg_file}")
        else:
            print(f"[+] SVG file has semantic attributes: {svg_file}")
    
    # Check GeoJSON content
    for geojson_file in geojson_files:
        try:
            import json
            with open(geojson_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'type' not in data or data['type'] != 'FeatureCollection':
                print(f"[X] Invalid GeoJSON file: {geojson_file}")
                return False
            
            print(f"[+] Valid GeoJSON file: {geojson_file}")
            
        except json.JSONDecodeError as e:
            print(f"[X] Invalid JSON in file {geojson_file}: {e}")
            return False
    
    print("[+] Output file validation passed")
    return True


def run_comprehensive_test():
    """Run comprehensive system test."""
    print("="*60)
    print("PNG2SVG System Test Suite")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Dependency Check", test_dependency_check),
        ("Config Loading", test_config_loading),
        ("Image Processing", test_image_processing),
        ("CLI Interface", test_cli_interface),
        ("Output Validation", validate_output_files),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"[X] {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "[+]" if result else "[X]"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! System is fully functional.")
        return True
    else:
        print("[WARNING] Some tests failed. Check the logs above.")
        return False


if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run tests
    success = run_comprehensive_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)