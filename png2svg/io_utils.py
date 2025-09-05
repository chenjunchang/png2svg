"""
I/O utilities for PNG2SVG system.
Handles file operations, parallel processing, logging, and error management.
"""

import os
import logging
import multiprocessing
from pathlib import Path
from typing import List, Callable, Any, Optional, Iterator, Tuple
from dataclasses import dataclass
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import glob

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class ProcessResult:
    """Result of processing a single file."""
    path: str
    success: bool
    output_svg: Optional[str] = None
    output_geojson: Optional[str] = None
    error: Optional[str] = None
    processing_time: float = 0.0


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('png2svg')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def list_png_files(input_dir: str) -> List[str]:
    """
    List all PNG files in the input directory.
    
    Args:
        input_dir: Directory to search for PNG files
        
    Returns:
        List of PNG file paths
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Supported image formats
    extensions = ['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.bmp', '*.BMP']
    
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        # Also search in subdirectories
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    return image_files


def validate_image_file(image_path: str) -> bool:
    """
    Validate if an image file is readable and valid.
    Handles Chinese/Unicode filenames on Windows properly.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        img_buffer = np.fromfile(image_path, dtype=np.uint8)
        if img_buffer.size == 0:
            return False
        
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img is not None and img.size > 0
    except Exception:
        return False


def ensure_output_directory(output_dir: str) -> None:
    """
    Ensure output directory exists, create if necessary.
    
    Args:
        output_dir: Output directory path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    """
    Convert a string to a safe filename by removing problematic characters.
    
    Args:
        name: Original filename
        
    Returns:
        Safe filename
    """
    # Replace problematic characters
    replacements = {
        '<': '_', '>': '_', ':': '_', '"': '_', 
        '/': '_', '\\': '_', '|': '_', '?': '_', 
        '*': '_', ' ': '_'
    }
    
    safe_name = name
    for old, new in replacements.items():
        safe_name = safe_name.replace(old, new)
    
    # Remove consecutive underscores
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    
    return safe_name.strip('_')


def read_image(image_path: str) -> np.ndarray:
    """
    Read image file with error handling and validation.
    Handles Chinese/Unicode filenames on Windows properly.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image array in BGR format
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If image is invalid or corrupted
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        # Use np.fromfile + cv2.imdecode to handle Chinese/Unicode filenames on Windows
        # This is more robust than cv2.imread for non-ASCII paths
        img_buffer = np.fromfile(image_path, dtype=np.uint8)
        if img_buffer.size == 0:
            raise ValueError(f"Empty file: {image_path}")
        
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image (corrupted or unsupported format): {image_path}")
        
        if img.size == 0:
            raise ValueError(f"Empty image after decoding: {image_path}")
        
        return img
    except Exception as e:
        raise ValueError(f"Error reading image {image_path}: {str(e)}")


def process_single_image_wrapper(args: Tuple[str, Callable, Any]) -> ProcessResult:
    """
    Wrapper function for processing a single image in multiprocessing context.
    
    Args:
        args: Tuple of (image_path, process_function, config)
        
    Returns:
        ProcessResult with processing outcome
    """
    image_path, process_func, config = args
    start_time = time.time()
    
    try:
        # Process the image
        result = process_func(image_path, config)
        processing_time = time.time() - start_time
        
        return ProcessResult(
            path=image_path,
            success=True,
            output_svg=result.get('svg_path') if result else None,
            output_geojson=result.get('geo_path') if result else None,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        return ProcessResult(
            path=image_path,
            success=False,
            error=error_msg,
            processing_time=processing_time
        )


def run_parallel_processing(
    image_files: List[str],
    process_function: Callable,
    config: Any,
    max_workers: int = 4,
    timeout: float = 60.0,
    progress_bar: bool = True
) -> List[ProcessResult]:
    """
    Run parallel processing on a list of image files.
    
    Args:
        image_files: List of image file paths
        process_function: Function to process each image
        config: Configuration object
        max_workers: Maximum number of parallel workers
        timeout: Timeout for each image processing (seconds)
        progress_bar: Whether to show progress bar
        
    Returns:
        List of ProcessResult objects
    """
    logger = logging.getLogger('png2svg')
    logger.info(f"Starting parallel processing of {len(image_files)} images with {max_workers} workers")
    
    results = []
    failed_count = 0
    
    # Prepare arguments for multiprocessing
    args_list = [(img_path, process_function, config) for img_path in image_files]
    
    # Setup progress bar
    if progress_bar:
        pbar = tqdm(total=len(image_files), desc="Processing images", unit="img")
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_path = {
                executor.submit(process_single_image_wrapper, args): args[0] 
                for args in args_list
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path, timeout=None):
                image_path = future_to_path[future]
                
                try:
                    # Get result with timeout
                    result = future.result(timeout=timeout)
                    results.append(result)
                    
                    if result.success:
                        logger.debug(f"Successfully processed: {image_path} ({result.processing_time:.2f}s)")
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to process: {image_path} - {result.error}")
                
                except TimeoutError:
                    failed_count += 1
                    result = ProcessResult(
                        path=image_path,
                        success=False,
                        error=f"Processing timeout after {timeout}s",
                        processing_time=timeout
                    )
                    results.append(result)
                    logger.error(f"Timeout processing: {image_path}")
                    
                except Exception as e:
                    failed_count += 1
                    result = ProcessResult(
                        path=image_path,
                        success=False,
                        error=f"Unexpected error: {str(e)}",
                        processing_time=0.0
                    )
                    results.append(result)
                    logger.error(f"Unexpected error processing {image_path}: {str(e)}")
                
                # Update progress bar
                if progress_bar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'success': len(results) - failed_count,
                        'failed': failed_count
                    })
    
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Critical error in parallel processing: {str(e)}")
    finally:
        if progress_bar:
            pbar.close()
    
    # Log summary
    success_count = len(results) - failed_count
    total_time = sum(r.processing_time for r in results)
    avg_time = total_time / len(results) if results else 0
    
    logger.info(f"Processing complete: {success_count} success, {failed_count} failed")
    logger.info(f"Total time: {total_time:.2f}s, Average: {avg_time:.2f}s per image")
    
    return results


def run_sequential_processing(
    image_files: List[str],
    process_function: Callable,
    config: Any,
    progress_bar: bool = True
) -> List[ProcessResult]:
    """
    Run sequential processing on a list of image files.
    
    Args:
        image_files: List of image file paths
        process_function: Function to process each image
        config: Configuration object
        progress_bar: Whether to show progress bar
        
    Returns:
        List of ProcessResult objects
    """
    logger = logging.getLogger('png2svg')
    logger.info(f"Starting sequential processing of {len(image_files)} images")
    
    results = []
    failed_count = 0
    
    # Setup progress bar
    if progress_bar:
        pbar = tqdm(image_files, desc="Processing images", unit="img")
        iterator = pbar
    else:
        iterator = image_files
    
    for image_path in iterator:
        start_time = time.time()
        
        try:
            result = process_function(image_path, config)
            processing_time = time.time() - start_time
            
            results.append(ProcessResult(
                path=image_path,
                success=True,
                output_svg=result.get('svg_path') if result else None,
                output_geojson=result.get('geo_path') if result else None,
                processing_time=processing_time
            ))
            
            logger.debug(f"Successfully processed: {image_path} ({processing_time:.2f}s)")
            
        except Exception as e:
            processing_time = time.time() - start_time
            failed_count += 1
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            results.append(ProcessResult(
                path=image_path,
                success=False,
                error=error_msg,
                processing_time=processing_time
            ))
            
            logger.warning(f"Failed to process: {image_path} - {error_msg}")
    
    if progress_bar:
        pbar.close()
    
    # Log summary
    success_count = len(results) - failed_count
    total_time = sum(r.processing_time for r in results)
    avg_time = total_time / len(results) if results else 0
    
    logger.info(f"Sequential processing complete: {success_count} success, {failed_count} failed")
    logger.info(f"Total time: {total_time:.2f}s, Average: {avg_time:.2f}s per image")
    
    return results


def print_processing_summary(results: List[ProcessResult]) -> None:
    """
    Print a summary of processing results.
    
    Args:
        results: List of ProcessResult objects
    """
    if not results:
        print("No results to summarize.")
        return
    
    success_count = sum(1 for r in results if r.success)
    failed_count = len(results) - success_count
    total_time = sum(r.processing_time for r in results)
    avg_time = total_time / len(results)
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total files:     {len(results)}")
    print(f"Successful:      {success_count}")
    print(f"Failed:          {failed_count}")
    print(f"Success rate:    {success_count/len(results)*100:.1f}%")
    print(f"Total time:      {total_time:.2f} seconds")
    print(f"Average time:    {avg_time:.2f} seconds per image")
    
    if failed_count > 0:
        print("\nFAILED FILES:")
        print("-" * 40)
        for result in results:
            if not result.success:
                print(f"  {Path(result.path).name}: {result.error}")
    
    print("="*60)