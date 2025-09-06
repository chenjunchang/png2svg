"""
Command-line interface for PNG2SVG.

Provides a comprehensive CLI for converting PNG mathematical diagrams to SVG
with flexible configuration options and batch processing capabilities.
"""

import sys
import argparse
import traceback
from pathlib import Path
from typing import List, Optional

from .config import load_config, create_default_config_file, Config
from .io_utils import (
    setup_logging, list_png_files, ensure_output_dir, run_parallel,
    validate_input_files, print_processing_summary, ProcessingResult
)
from .pipeline import process_image


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for PNG2SVG CLI.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="png2svg",
        description="Convert PNG mathematical diagrams to structured SVG with semantic metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PNG files in inputs/ directory using config.yaml
  python -m png2svg.cli --config config.yaml

  # Process with custom input/output directories
  python -m png2svg.cli --input ./pngs --output ./results

  # Process single file with 8 parallel jobs
  python -m png2svg.cli --input ./pngs --single demo.png --jobs 8

  # Disable optional features for faster processing
  python -m png2svg.cli --input ./pngs --no-yolo --no-constraints

  # Create example configuration file
  python -m png2svg.cli --create-config
        """
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create example configuration file and exit"
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input directory containing PNG files (overrides config)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for SVG and GeoJSON files (overrides config)"
    )
    
    parser.add_argument(
        "--single", "-s",
        type=str,
        help="Process only a specific PNG file (by filename, not full path)"
    )
    
    # Processing options
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        help="Number of parallel processing jobs (overrides config)"
    )
    
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Disable YOLO symbol detection (use rule-based fallback)"
    )
    
    parser.add_argument(
        "--no-paddleocr",
        action="store_true",
        help="Disable PaddleOCR (use Tesseract fallback)"
    )
    
    parser.add_argument(
        "--no-constraints",
        action="store_true",
        help="Disable geometric constraint solver"
    )
    
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable Test Time Augmentation for confidence improvement"
    )
    
    # Debugging and output
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="PNG2SVG 0.1.0"
    )
    
    return parser


def handle_create_config(args: argparse.Namespace) -> int:
    """
    Handle --create-config command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        int: Exit code (0 for success)
    """
    output_path = "config.example.yaml"
    if hasattr(args, 'output') and args.output:
        output_path = str(Path(args.output) / "config.example.yaml")
    
    try:
        create_default_config_file(output_path)
        print(f"Created example configuration file: {output_path}")
        print("Copy this file to config.yaml and modify as needed.")
        return 0
    except Exception as e:
        print(f"Error creating configuration file: {e}", file=sys.stderr)
        return 1


def setup_logging_from_args(args: argparse.Namespace) -> None:
    """
    Set up logging based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    """
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "ERROR"
    else:
        log_level = "INFO"
    
    setup_logging(log_level)


def validate_and_prepare_files(config: Config, single_file: Optional[str]) -> List[str]:
    """
    Validate input directory and prepare list of files to process.
    
    Args:
        config: Configuration object
        single_file: Optional single file to process
        
    Returns:
        List[str]: List of valid file paths to process
        
    Raises:
        SystemExit: If no valid files found or input validation fails
    """
    logger = setup_logging(config.log_level)
    
    try:
        # Get list of PNG files
        all_files = list_png_files(config.input_dir)
        
        if not all_files:
            logger.error(f"No PNG files found in: {config.input_dir}")
            sys.exit(1)
        
        # Filter for single file if specified
        if single_file:
            matching_files = [f for f in all_files if Path(f).name == single_file]
            if not matching_files:
                logger.error(f"File '{single_file}' not found in {config.input_dir}")
                logger.info(f"Available files: {[Path(f).name for f in all_files[:10]]}")
                sys.exit(1)
            files_to_process = matching_files
        else:
            files_to_process = all_files
        
        # Validate files
        valid_files, invalid_files = validate_input_files(files_to_process)
        
        if invalid_files:
            logger.warning(f"Found {len(invalid_files)} invalid files:")
            for invalid in invalid_files[:5]:  # Show first 5
                logger.warning(f"  - {invalid}")
            if len(invalid_files) > 5:
                logger.warning(f"  ... and {len(invalid_files) - 5} more")
        
        if not valid_files:
            logger.error("No valid PNG files to process")
            sys.exit(1)
        
        logger.info(f"Found {len(valid_files)} valid PNG files to process")
        return valid_files
        
    except Exception as e:
        logger.error(f"Error preparing files: {e}")
        sys.exit(1)


def run_dry_run(files: List[str], config: Config) -> int:
    """
    Perform a dry run showing what would be processed.
    
    Args:
        files: List of files that would be processed
        config: Configuration object
        
    Returns:
        int: Exit code (always 0 for dry run)
    """
    print(f"=== DRY RUN ===")
    print(f"Configuration:")
    print(f"  Input directory: {config.input_dir}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Parallel jobs: {config.jobs}")
    print(f"  YOLO symbols: {config.use_yolo_symbols}")
    print(f"  PaddleOCR: {config.use_paddle_ocr}")
    print(f"  Constraint solver: {config.apply_constraint_solver}")
    print(f"  TTA: {config.confidence_tta}")
    print()
    print(f"Files to process ({len(files)}):")
    for i, file_path in enumerate(files[:10], 1):
        print(f"  {i}. {Path(file_path).name}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")
    print()
    print("Run without --dry-run to actually process these files.")
    return 0


def main() -> int:
    """
    Main entry point for PNG2SVG CLI.
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        # Parse arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Handle special commands
        if args.create_config:
            return handle_create_config(args)
        
        # Set up logging
        setup_logging_from_args(args)
        logger = setup_logging("INFO")
        
        # Load configuration
        try:
            config = load_config(args)
        except Exception as e:
            logger.error(f"Configuration error: {e}")
            return 1
        
        # Apply additional CLI options
        if hasattr(args, 'no_tta') and args.no_tta:
            config.confidence_tta = False
        
        # Prepare output directory
        ensure_output_dir(config.output_dir)
        
        # Validate and prepare files
        files_to_process = validate_and_prepare_files(config, args.single)
        
        # Handle dry run
        if args.dry_run:
            return run_dry_run(files_to_process, config)
        
        # Process files
        logger.info(f"Starting PNG2SVG processing...")
        logger.info(f"Input: {config.input_dir}")
        logger.info(f"Output: {config.output_dir}")
        logger.info(f"Files: {len(files_to_process)}")
        
        results = run_parallel(
            files_to_process,
            process_image,
            config,
            desc="Converting PNG to SVG"
        )
        
        # Print summary
        print_processing_summary(results)
        
        # Determine exit code
        failed_count = sum(1 for r in results if not r.success)
        if failed_count == 0:
            logger.info("All files processed successfully!")
            return 0
        elif failed_count < len(results):
            logger.warning(f"Processing completed with {failed_count} failures")
            return 2  # Partial success
        else:
            logger.error("All files failed to process")
            return 3  # Complete failure
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user", file=sys.stderr)
        return 130
    
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        print(f"Traceback: {traceback.format_exc()}", file=sys.stderr)
        return 1


def cli_entry_point():
    """Entry point for console script."""
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())