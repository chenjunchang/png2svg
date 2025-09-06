# PNG2SVG

Convert PNG mathematical diagrams to structured SVG with semantic metadata.

PNG2SVG is a comprehensive tool that transforms mathematical diagrams from PNG images into structured, semantic SVG files. It combines traditional computer vision techniques with optional deep learning to detect geometric primitives, mathematical symbols, and text labels, then exports both visual SVG and structured GeoJSON representations.

## Features

🎯 **Geometric Detection**
- Lines, circles, arcs with automatic dash pattern recognition
- Intersection detection and topological analysis
- Geometric constraint solving for precise shapes

🔍 **Symbol Recognition**  
- YOLO-based symbol detection (arrows, right angles, tick marks)
- Rule-based fallbacks for common mathematical symbols
- Test Time Augmentation for improved confidence

📝 **Text Recognition**
- PaddleOCR integration with Tesseract fallback
- Mathematical notation and variable recognition
- Multi-language support

🎨 **Structured Output**
- Layered SVG with semantic metadata
- GeoJSON export for programmatic analysis
- Configurable styling and coordinate systems

⚡ **Production Ready**
- Parallel batch processing
- Graceful degradation when optional dependencies unavailable
- Comprehensive error handling and logging

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/png2svg.git
cd png2svg

# Install dependencies
pip install -r requirements.txt

# Optional: Install advanced features
pip install ultralytics paddleocr  # For YOLO and advanced OCR
```

### Basic Usage

```bash
# Process all PNG files in a directory
python -m png2svg.cli --input ./images --output ./results

# Process a single file
python -m png2svg.cli --input ./images --single diagram.png

# Use custom configuration
python -m png2svg.cli --config config.yaml
```

### Configuration

Create a configuration file from the example:

```bash
# Generate example configuration
python -m png2svg.cli --create-config

# Copy and modify
cp config.example.yaml config.yaml
# Edit config.yaml with your preferences
```

## Architecture

PNG2SVG uses a multi-stage pipeline that processes images through several specialized modules:

```
PNG Input → Preprocessing → Detection → Analysis → Output
    ↓           ↓             ↓          ↓        ↓
  Original   Binarized    Primitives   Graph   SVG+JSON
   Image    + Deskewed    + Symbols   + Rels   + Metadata
```

### Processing Pipeline

1. **Preprocessing** (`preprocess.py`)
   - Grayscale conversion and denoising
   - Adaptive thresholding for binarization
   - Automatic skew detection and correction

2. **Primitive Detection** (`detect_primitives.py`)
   - Line detection using LSD/FLD with Hough fallback
   - Circle detection using HoughCircles
   - Arc detection through edge fitting
   - Dash pattern recognition

3. **Symbol Detection** (`detect_symbols.py`)
   - YOLO-based detection for mathematical symbols
   - Rule-based fallbacks for common shapes
   - Symbol-geometry association

4. **Text Recognition** (`ocr_text.py`)
   - PaddleOCR for robust multilingual OCR
   - Tesseract fallback integration
   - Text classification (labels, variables, angles)

5. **Topological Analysis** (`topology.py`)
   - Graph construction with nodes and edges
   - Intersection detection and spatial relationships
   - Semantic binding of symbols and text to geometry

6. **Constraint Solving** (`constraints.py`)
   - Geometric constraint optimization using SciPy
   - Parallel/perpendicular enforcement
   - Equal length and point-on-curve constraints

7. **Output Generation**
   - **SVG Writer** (`svg_writer.py`): Layered SVG with semantic metadata
   - **GeoJSON Writer** (`geojson_writer.py`): Structured geometric data

## Configuration

### Basic Configuration

The system uses YAML configuration files. Key settings:

```yaml
# Input/Output
input_dir: "./inputs"
output_dir: "./outputs" 
jobs: 4                    # Parallel processing threads

# Processing options
deskew: true               # Auto-correct image rotation
use_yolo_symbols: true     # Enable YOLO symbol detection
use_paddle_ocr: true       # Use PaddleOCR (fallback to Tesseract)
apply_constraint_solver: true  # Geometric optimization
confidence_tta: true       # Test Time Augmentation

# Detection parameters
min_line_len: 25           # Minimum line length (pixels)
line_merge_angle_deg: 3    # Collinear line merging threshold
line_merge_gap_px: 6       # Line gap merging threshold

# Output styling
svg:
  scale: 1.0               # SVG coordinate scaling
  stroke_main: 2           # Main element stroke width
  stroke_aux: 1            # Auxiliary element stroke width
  dash_pattern: "6,6"      # Dashed line pattern
```

### Advanced Configuration

See `config.example.yaml` for complete configuration options including:
- Fine-tuned detection parameters
- Custom styling options
- Processing pipeline controls
- Output format specifications

## Command Line Interface

### Basic Commands

```bash
# Process directory with default settings
python -m png2svg.cli --input ./images

# Specify output location
python -m png2svg.cli --input ./images --output ./results

# Process single file
python -m png2svg.cli --input ./images --single diagram.png

# Use multiple CPU cores
python -m png2svg.cli --input ./images --jobs 8
```

### Advanced Options

```bash
# Disable optional features for faster processing
python -m png2svg.cli --input ./images --no-yolo --no-constraints

# Enable verbose logging
python -m png2svg.cli --input ./images --verbose

# Dry run (show what would be processed)
python -m png2svg.cli --input ./images --dry-run

# Custom configuration
python -m png2svg.cli --config my-config.yaml
```

### Exit Codes

- `0`: Success
- `1`: General error (configuration, file access, etc.)
- `2`: Partial success (some files failed)
- `3`: Complete failure (all files failed)
- `130`: Interrupted by user (Ctrl+C)

## Output Formats

### SVG Output

Generated SVG files include:

- **Layered structure**: main geometry, auxiliary lines, symbols, text
- **Semantic metadata**: data attributes with confidence scores, relationships
- **Scalable styling**: CSS-based styling with multiple themes
- **Accessibility**: Proper titles, descriptions, and semantic markup

Example SVG structure:
```xml
<svg viewBox="0 0 400 300">
  <g id="main" class="main-layer">
    <line data-element-type="line" data-confidence="0.95" ... />
    <circle data-element-type="circle" data-radius="25.3" ... />
  </g>
  <g id="auxiliary" class="auxiliary-layer">
    <line stroke-dasharray="6,6" data-role="aux" ... />
  </g>
  <g id="text" class="text-layer">
    <text data-node-id="node1" data-semantic-role="vertex">A</text>
  </g>
</svg>
```

### GeoJSON Output

Structured data export includes:

- **Feature collection**: Points, LineStrings, Polygons for geometric elements
- **Rich properties**: Element types, confidence scores, measurements
- **Relationship data**: Spatial and semantic relationships between elements
- **Metadata**: Processing statistics, configuration used, timestamps

Example GeoJSON structure:
```json
{
  "type": "FeatureCollection",
  "metadata": {
    "version": "1.0",
    "generator": "PNG2SVG",
    "statistics": { ... }
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "LineString", "coordinates": [[10, 20], [50, 80]]},
      "properties": {
        "element_type": "line_segment",
        "confidence": 0.95,
        "length": 72.11,
        "dashed": false
      }
    }
  ]
}
```

## Optional Dependencies

PNG2SVG is designed to work with minimal dependencies but offers enhanced features when additional packages are available:

### Core Dependencies (Required)
```
opencv-python>=4.9     # Computer vision algorithms
numpy>=1.24            # Numerical computations
scipy>=1.11            # Constraint optimization  
shapely>=2.0           # Geometric operations
svgwrite>=1.4          # SVG generation
pydantic>=2.7          # Configuration validation
```

### Enhanced Features (Optional)
```
ultralytics>=8.2       # YOLO symbol detection
onnxruntime>=1.17      # ONNX model inference
paddleocr>=2.7.0       # Advanced OCR capabilities
scikit-image>=0.23     # Advanced image processing
```

### Graceful Degradation

When optional dependencies are unavailable:
- **No YOLO**: Falls back to rule-based symbol detection
- **No PaddleOCR**: Uses Tesseract for text recognition  
- **No SciPy**: Skips geometric constraint optimization
- **No scikit-image**: Uses basic OpenCV image processing

## Troubleshooting

### Common Issues

**1. Poor OCR Results**
```bash
# Try PaddleOCR instead of Tesseract
pip install paddleocr
python -m png2svg.cli --input ./images  # Will auto-detect and use PaddleOCR
```

**2. Missing Geometric Elements**
```yaml
# Adjust detection thresholds in config.yaml
min_line_len: 15        # Lower threshold for short lines
line_merge_gap_px: 10   # Higher tolerance for broken lines
```

**3. Symbol Detection Issues**
```bash
# Check if YOLO model is available
ls models/symbols_yolo.onnx

# Use rule-based detection only
python -m png2svg.cli --input ./images --no-yolo
```

**4. Performance Issues**
```bash
# Reduce parallel jobs
python -m png2svg.cli --input ./images --jobs 2

# Disable expensive features
python -m png2svg.cli --input ./images --no-constraints --no-tta
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python -m png2svg.cli --input ./images --verbose
```

This will show:
- Processing time for each stage
- Detection statistics  
- Warning messages for degraded features
- Error details for failed files

### Log Files

Check log output for detailed error information:
- Processing statistics per image
- Failed detection attempts
- Configuration validation issues
- Dependency availability status

## Development

### Project Structure

```
png2svg/
├── __init__.py              # Package initialization
├── cli.py                   # Command-line interface  
├── config.py                # Configuration management
├── pipeline.py              # Main processing orchestrator
├── io_utils.py              # File I/O and parallel processing
├── preprocess.py            # Image preprocessing
├── detect_primitives.py     # Geometric primitive detection
├── detect_symbols.py        # Symbol detection (YOLO + rules)
├── ocr_text.py              # Text recognition
├── topology.py              # Spatial relationship analysis
├── constraints.py           # Geometric constraint solving
├── svg_writer.py            # SVG output generation
├── geojson_writer.py        # GeoJSON export
├── models/                  # Optional model weights
├── styles/                  # SVG styling templates
└── examples/                # Sample files
```

### Testing

```bash
# Run with example files
python -m png2svg.cli --input examples/ --output test_output/

# Validate output
python -c "
from png2svg.svg_writer import validate_svg_output
from png2svg.geojson_writer import validate_geojson_output
print('SVG valid:', validate_svg_output('test_output/demo.svg'))
print('GeoJSON valid:', validate_geojson_output('test_output/demo.geo.json'))
"
```

### Extending

The modular architecture makes it easy to extend:

1. **Add new symbol types**: Extend `detect_symbols.py` with additional rules
2. **Custom output formats**: Create new writer modules following the existing patterns
3. **Alternative OCR engines**: Implement new OCR backends in `ocr_text.py`
4. **Advanced constraints**: Add geometric relationships in `constraints.py`

## Performance

### Benchmarks

Typical processing times on modern hardware:

| Image Size | Elements | Time (CPU) | Time (GPU) |
|------------|----------|------------|------------|
| 800×600    | 10-20    | 0.8s       | 0.5s       |
| 1200×900   | 20-40    | 1.5s       | 0.8s       |
| 1920×1080  | 40-80    | 3.2s       | 1.5s       |

### Optimization Tips

1. **Batch Processing**: Use `--jobs N` for parallel processing
2. **Selective Features**: Disable unused features with `--no-yolo`, `--no-constraints`
3. **Image Preprocessing**: Ensure high-contrast, clean input images
4. **Hardware**: GPU acceleration available with CUDA-enabled dependencies

## Support

### Documentation
- Technical specification: `png2svg技术方案.md`
- Implementation details: `png2svg实现方案.md`
- Configuration reference: `config.example.yaml`

### Community
- GitHub Issues: Report bugs and request features
- Discussions: Share examples and use cases
- Wiki: Additional examples and tutorials

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV team for computer vision algorithms
- Ultralytics for YOLO implementation
- PaddlePaddle team for PaddleOCR
- SciPy community for optimization algorithms
- Contributors to Shapely, NetworkX, and other dependencies

---

**PNG2SVG** - Transforming mathematical diagrams into structured, semantic vector graphics.