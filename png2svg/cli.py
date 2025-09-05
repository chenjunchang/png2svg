"""
Command Line Interface for PNG2SVG system.
Provides command-line tool for batch processing PNG images to SVG.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import load_config, Config
from .io_utils import setup_logging, list_png_files, run_parallel_processing, run_sequential_processing, print_processing_summary
from .pipeline import run_pipeline_with_recovery, get_processing_summary


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='PNG2SVG - Convert PNG mathematical diagrams to semantic SVG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config file
  python -m png2svg.cli --config config.yaml

  # Specify input/output and override config
  python -m png2svg.cli --input ./pngs --output ./out --jobs 8

  # Process single file
  python -m png2svg.cli --input ./pngs --single demo.png

  # Disable optional features
  python -m png2svg.cli --input ./pngs --no-yolo --no-constraints

  # Debug mode with detailed logging
  python -m png2svg.cli --input ./pngs --verbose --log-file debug.log
        """
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.example.yaml',
        help='Path to YAML configuration file (default: config.example.yaml)'
    )
    
    # Input/Output
    parser.add_argument(
        '--input', 
        type=str,
        help='Input directory containing PNG files'
    )
    parser.add_argument(
        '--output', 
        type=str,
        help='Output directory for SVG and GeoJSON files'
    )
    
    # Processing options
    parser.add_argument(
        '--jobs', 
        type=int,
        help='Number of parallel processing jobs'
    )
    parser.add_argument(
        '--single', 
        type=str,
        help='Process only single file (filename in input directory)'
    )
    
    # Feature toggles
    parser.add_argument(
        '--no-yolo',
        action='store_true',
        help='Disable YOLO symbol detection (use rule-based fallback)'
    )
    parser.add_argument(
        '--no-paddleocr',
        action='store_true', 
        help='Disable PaddleOCR (use Tesseract fallback)'
    )
    parser.add_argument(
        '--no-constraints',
        action='store_true',
        help='Disable constraint solver optimization'
    )
    parser.add_argument(
        '--no-tta',
        action='store_true',
        help='Disable Test-Time Augmentation for improved confidence'
    )
    parser.add_argument(
        '--no-deskew',
        action='store_true',
        help='Disable automatic image deskewing'
    )
    
    # Output options
    parser.add_argument(
        '--svg-only',
        action='store_true',
        help='Generate SVG output only (skip GeoJSON)'
    )
    parser.add_argument(
        '--geojson-only', 
        action='store_true',
        help='Generate GeoJSON output only (skip SVG)'
    )
    
    # Logging and debugging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Write logs to file'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    # Information
    parser.add_argument(
        '--version',
        action='version',
        version='PNG2SVG v1.0 - Mathematical diagram vectorization system'
    )
    
    return parser


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.svg_only and args.geojson_only:
        parser.error("--svg-only and --geojson-only are mutually exclusive")
    
    if args.verbose and args.quiet:
        parser.error("--verbose and --quiet are mutually exclusive")
    
    return args


def build_config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build configuration overrides from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary of configuration overrides
    """
    overrides = {}
    
    # Basic settings
    if args.input:
        overrides['input_dir'] = args.input
    if args.output:
        overrides['output_dir'] = args.output
    if args.jobs:
        overrides['jobs'] = args.jobs
    
    # Feature toggles
    if args.no_yolo:
        overrides['use_yolo_symbols'] = False
    if args.no_paddleocr:
        overrides['use_paddle_ocr'] = False
    if args.no_constraints:
        overrides['apply_constraint_solver'] = False
    if args.no_tta:
        overrides['confidence_tta'] = False
    if args.no_deskew:
        overrides['deskew'] = False
    
    # Output options
    if args.svg_only:
        overrides['export'] = {'write_svg': True, 'write_geojson': False}
    elif args.geojson_only:
        overrides['export'] = {'write_svg': False, 'write_geojson': True}
    
    # Logging
    if args.verbose:
        overrides['log_level'] = 'DEBUG'
    elif args.quiet:
        overrides['log_level'] = 'WARNING'
    
    return overrides


def validate_inputs(args: argparse.Namespace, cfg: Config) -> bool:
    """
    Validate input arguments and configuration.
    
    Args:
        args: Parsed arguments
        cfg: Configuration object
        
    Returns:
        True if inputs are valid
    """
    # Check input directory exists
    input_path = Path(cfg.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        return False
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_path}")
        return False
    
    # Check if single file exists
    if args.single:
        single_file_path = input_path / args.single
        if not single_file_path.exists():
            print(f"Error: Single file not found: {single_file_path}")
            return False
    
    # Check output directory can be created
    output_path = Path(cfg.output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error: Cannot create output directory {output_path}: {e}")
        return False
    
    return True


def process_files(image_files: List[str], cfg: Config, show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Process list of image files.
    
    Args:
        image_files: List of image file paths
        cfg: Configuration object
        show_progress: Whether to show progress bar
        
    Returns:
        List of processing results
    """
    if cfg.jobs > 1:
        # Use parallel processing
        from functools import partial
        process_func = partial(run_pipeline_with_recovery, cfg=cfg)
        
        # Create argument tuples for parallel processing
        args_list = [(img_path, process_func, cfg) for img_path in image_files]
        
        return run_parallel_processing(
            image_files,
            run_pipeline_with_recovery,
            cfg,
            max_workers=cfg.jobs,
            progress_bar=show_progress
        )
    else:
        # Use sequential processing
        return run_sequential_processing(
            image_files,
            run_pipeline_with_recovery,
            cfg,
            progress_bar=show_progress
        )


def print_results_summary(results: List[Dict[str, Any]], args: argparse.Namespace):
    """
    Print summary of processing results.
    
    Args:
        results: List of processing results
        args: Command line arguments
    """
    if args.quiet:
        return
    
    summary = get_processing_summary(results)
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files processed: {summary['total_processed']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']}%")
    
    if summary.get('timing'):
        timing = summary['timing']
        print(f"Total processing time: {timing['total_time']:.1f}s")
        print(f"Average time per image: {timing['average_time']:.2f}s")
    
    if summary.get('geometry_totals'):
        geo = summary['geometry_totals']
        print(f"\nGeometry detected:")
        print(f"  Lines: {geo['lines']}")
        print(f"  Circles: {geo['circles']}")
        print(f"  Relations: {geo['relations']}")
    
    if summary['failed'] > 0 and summary.get('failure_analysis'):
        print(f"\nFailure analysis:")
        for step, count in summary['failure_analysis'].items():
            print(f"  {step}: {count} failures")
    
    print("="*60)
    
    # Print failed files if any
    failed_files = [r for r in results if not r.success]
    if failed_files and not args.quiet:
        print(f"\nFailed files ({len(failed_files)}):")
        for result in failed_files[:10]:  # Show first 10 failures
            error = result.error or 'Unknown error'
            # Extract filename from path
            filename = Path(result.path).name if result.path else "Unknown file"
            print(f"  {filename}: {error}")
        
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")


def main():
    """
    Main entry point for CLI application.
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration with overrides
        config_overrides = build_config_overrides(args)
        
        try:
            cfg = load_config(args.config, config_overrides)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
        
        # Setup logging
        log_file = args.log_file
        setup_logging(cfg.log_level, log_file)
        
        # Validate inputs
        if not validate_inputs(args, cfg):
            sys.exit(1)
        
        # Find files to process
        if args.single:
            image_files = [str(Path(cfg.input_dir) / args.single)]
        else:
            image_files = list_png_files(cfg.input_dir)
        
        if not image_files:
            print("No PNG files found in input directory")
            sys.exit(1)
        
        if not args.quiet:
            print(f"Found {len(image_files)} image files to process")
            print(f"Input directory: {cfg.input_dir}")
            print(f"Output directory: {cfg.output_dir}")
            print(f"Processing with {cfg.jobs} {'job' if cfg.jobs == 1 else 'jobs'}")
            
            # Show enabled features
            features = []
            if cfg.use_yolo_symbols:
                features.append("YOLO symbols")
            if cfg.use_paddle_ocr:
                features.append("PaddleOCR")
            if cfg.apply_constraint_solver:
                features.append("Constraint solver")
            if cfg.confidence_tta:
                features.append("TTA")
            if cfg.deskew:
                features.append("Deskewing")
                
            if features:
                print(f"Enabled features: {', '.join(features)}")
            print()
        
        # Process files
        results = process_files(image_files, cfg, show_progress=not args.quiet)
        
        # Print summary
        print_results_summary(results, args)
        
        # Determine exit code
        failed_count = sum(1 for r in results if not r.success)
        if failed_count == 0:
            if not args.quiet:
                print("All files processed successfully!")
            sys.exit(0)
        else:
            if not args.quiet:
                print(f"Processing completed with {failed_count} failures")
            sys.exit(1 if failed_count == len(results) else 2)  # 1 for total failure, 2 for partial failure
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Critical error: {e}")
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()