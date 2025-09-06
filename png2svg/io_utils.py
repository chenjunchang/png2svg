"""
I/O utilities, parallel processing, and logging for PNG2SVG.

Provides functions for file handling, parallel batch processing,
logging setup, and robust error handling.
"""

import os
import logging
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import List, Callable, Any, Optional, Dict, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class ProcessingResult:
    """Result of processing a single image."""
    path: str
    success: bool
    svg_path: Optional[str] = None
    geo_path: Optional[str] = None
    error_msg: Optional[str] = None
    processing_time: float = 0.0
    stats: Dict[str, Any] = None


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration for PNG2SVG.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("png2svg")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid adding multiple handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def list_png_files(input_dir: str) -> List[str]:
    """
    Find all PNG files in the input directory.
    
    Args:
        input_dir: Directory to search for PNG files
        
    Returns:
        List[str]: List of PNG file paths
        
    Raises:
        FileNotFoundError: If input directory doesn't exist
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Find PNG files (case-insensitive)
    png_files = []
    for pattern in ['*.png', '*.PNG']:
        png_files.extend(input_path.glob(pattern))
    
    # Sort for consistent ordering
    png_files.sort()
    
    return [str(f) for f in png_files]


def ensure_output_dir(output_dir: str) -> None:
    """
    Ensure output directory exists, create if necessary.
    
    Args:
        output_dir: Directory path to create
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def read_image(path: str) -> np.ndarray:
    """
    Read an image file with error handling and Unicode path support.
    
    Args:
        path: Path to image file
        
    Returns:
        np.ndarray: Image as BGR numpy array
        
    Raises:
        ValueError: If image cannot be read
    """
    try:
        # Use np.fromfile + cv2.imdecode for Unicode path support
        image_buffer = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"Cannot decode image: {path}")
        return img
        
    except Exception as e:
        raise ValueError(f"Cannot read image: {path}. Error: {str(e)}")


def get_output_paths(input_path: str, output_dir: str) -> Tuple[str, str]:
    """
    Generate output file paths for SVG and GeoJSON.
    
    Args:
        input_path: Path to input PNG file
        output_dir: Output directory
        
    Returns:
        Tuple[str, str]: (svg_path, geo_path)
    """
    base_name = Path(input_path).stem
    svg_path = Path(output_dir) / f"{base_name}.svg"
    geo_path = Path(output_dir) / f"{base_name}.geo.json"
    return str(svg_path), str(geo_path)


def _process_single_with_timeout(args: Tuple) -> ProcessingResult:
    """
    Process a single image with timeout and error handling.
    
    Args:
        args: Tuple of (image_path, process_func, config, timeout)
        
    Returns:
        ProcessingResult: Processing result with success/error info
    """
    image_path, process_func, config, timeout = args
    
    logger = logging.getLogger("png2svg")
    start_time = time.time()
    
    try:
        # Process the image
        result = process_func(image_path, config)
        
        processing_time = time.time() - start_time
        
        if result is None:
            return ProcessingResult(
                path=image_path,
                success=False,
                error_msg="Process function returned None",
                processing_time=processing_time
            )
        
        return ProcessingResult(
            path=image_path,
            success=True,
            svg_path=getattr(result, 'svg_path', None),
            geo_path=getattr(result, 'geo_path', None),
            processing_time=processing_time,
            stats=getattr(result, 'stats', {})
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        
        logger.error(f"Error processing {image_path}: {error_msg}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return ProcessingResult(
            path=image_path,
            success=False,
            error_msg=error_msg,
            processing_time=processing_time
        )


def run_parallel(
    file_paths: List[str], 
    process_func: Callable, 
    config: Any,
    timeout: int = 60,
    desc: str = "Processing images"
) -> List[ProcessingResult]:
    """
    Run processing function on multiple files in parallel.
    
    Args:
        file_paths: List of file paths to process
        process_func: Function to process each file (should accept file_path, config)
        config: Configuration object to pass to process_func
        timeout: Timeout in seconds per file
        desc: Description for progress bar
        
    Returns:
        List[ProcessingResult]: Results for each processed file
    """
    logger = logging.getLogger("png2svg")
    
    if not file_paths:
        logger.warning("No files to process")
        return []
    
    logger.info(f"Processing {len(file_paths)} files with {config.jobs} parallel jobs")
    
    # Prepare arguments for multiprocessing
    args_list = [(path, process_func, config, timeout) for path in file_paths]
    
    results = []
    
    if config.jobs == 1:
        # Single-threaded processing with progress bar
        for args in tqdm(args_list, desc=desc):
            results.append(_process_single_with_timeout(args))
    else:
        # Multi-threaded processing
        try:
            with mp.Pool(processes=config.jobs) as pool:
                # Use imap for progress tracking
                result_iter = pool.imap(_process_single_with_timeout, args_list)
                
                # Collect results with progress bar
                results = list(tqdm(result_iter, total=len(args_list), desc=desc))
                
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user")
            pool.terminate()
            pool.join()
            raise
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            raise
    
    # Log summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_time = sum(r.processing_time for r in results)
    avg_time = total_time / len(results) if results else 0
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    logger.info(f"Total time: {total_time:.2f}s, Average: {avg_time:.2f}s per file")
    
    if failed > 0:
        logger.warning("Failed files:")
        for result in results:
            if not result.success:
                logger.warning(f"  - {result.path}: {result.error_msg}")
    
    return results


def validate_input_files(file_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate input files, separating valid and invalid ones.
    
    Args:
        file_paths: List of file paths to validate
        
    Returns:
        Tuple[List[str], List[str]]: (valid_files, invalid_files)
    """
    valid_files = []
    invalid_files = []
    
    for path in file_paths:
        try:
            # Check if file exists and can be read as image
            if not Path(path).exists():
                invalid_files.append(f"{path}: File not found")
                continue
            
            # Try to read image to validate it's a valid image file
            # Use the Unicode-compatible read_image function
            read_image(path)
            valid_files.append(path)
            
        except Exception as e:
            invalid_files.append(f"{path}: {str(e)}")
    
    return valid_files, invalid_files


def print_processing_summary(results: List[ProcessingResult]) -> None:
    """
    Print a detailed summary of processing results.
    
    Args:
        results: List of processing results
    """
    if not results:
        print("No files were processed.")
        return
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\n=== Processing Summary ===")
    print(f"Total files: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_time = sum(r.processing_time for r in successful) / len(successful)
        print(f"Average processing time: {avg_time:.2f}s")
    
    if failed:
        print(f"\nFailed files:")
        for result in failed[:10]:  # Show only first 10 failed files
            print(f"  - {Path(result.path).name}: {result.error_msg}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


class ProgressTracker:
    """Simple progress tracker for batch processing."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.completed = 0
        self.desc = desc
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress by n items."""
        self.completed += n
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed) / rate if rate > 0 else 0
        
        print(f"\r{self.desc}: {self.completed}/{self.total} "
              f"({100*self.completed/self.total:.1f}%) "
              f"[{rate:.1f}it/s, ETA: {eta:.1f}s]", end="")
    
    def close(self):
        """Finish progress tracking."""
        print()  # New line