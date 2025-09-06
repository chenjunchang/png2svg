# YOLO Symbol Detection Training Pipeline
.PHONY: help install build train-small train-full export check stats clean

# Default target
help:
	@echo "YOLO Symbol Detection Training Pipeline"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@echo "  install      - Install Python dependencies"
	@echo "  build        - Build dataset from eval sources"
	@echo "  train-small  - Quick training (60 epochs, 960px)"
	@echo "  train-full   - Full training (160 epochs, 1280px)"
	@echo "  export       - Export trained model to ONNX"
	@echo "  check        - Verify model with sample predictions"
	@echo "  stats        - Generate dataset statistics and plots"
	@echo "  clean        - Clean generated files"
	@echo ""
	@echo "Full workflow:"
	@echo "  make install && make build && make train-full && make export && make check"

# Install dependencies
install:
	pip install -r requirements.txt

# Build dataset
build:
	python scripts/00_build_dataset.py

# Quick training for testing
train-small:
	python scripts/01_train.py --cfg configs/train.small.yaml --name symbols_small

# Full training for production
train-full:
	python scripts/01_train.py --cfg configs/train.full.yaml --name symbols_full

# Resume training
resume:
	python scripts/01_train.py --resume --name symbols_full

# Export to ONNX
export:
	python scripts/02_export_onnx.py

# Sanity check with predictions
check:
	python scripts/03_sanity_check.py

# Generate statistics
stats:
	python scripts/04_stats.py

# Build with limited samples (for testing)
build-small:
	python scripts/00_build_dataset.py --max-per-dataset 100

# Full pipeline (build + train + export + check)
all: build train-full export check
	@echo "Full pipeline completed successfully!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	-rm -rf ../datasets/pg_symbols
	-rm -rf ../runs
	-rm -rf ../outputs
	-rm -rf __pycache__ converters/__pycache__ scripts/__pycache__
	@echo "Cleanup completed!"

# Development helpers
dev-check:
	@echo "Checking project structure..."
	@test -d configs || echo "❌ configs/ directory missing"
	@test -d converters || echo "❌ converters/ directory missing"  
	@test -d scripts || echo "❌ scripts/ directory missing"
	@test -f configs/classes.yaml || echo "❌ classes.yaml missing"
	@test -f requirements.txt || echo "❌ requirements.txt missing"
	@echo "✅ Project structure check complete"

# Show project info
info:
	@echo "Project Information"
	@echo "==================="
	@echo "Dataset sources: PGDP5K, PGPS9K, Geometry3K"
	@echo "Target classes: 12 (geometric symbols)"
	@echo "Model format: YOLO11 → ONNX"
	@echo "Output: png2svg/models/symbols_yolo.onnx"