"""
Atomistic model management utilities.

This module handles downloading, locating, and managing the DCNN models
used for atomic-resolution microscopy analysis.
"""

import os
import glob
import logging


def get_or_download_atomistic_model(settings: dict, logger: logging.Logger = None) -> str | None:
    """
    Manages finding or downloading the DCNN models for atomistic analysis.
    
    This function will:
    1. Check if user provided a model path and validate it
    2. Check if default model directory exists
    3. If not, download the model from Google Drive
    4. Unzip and locate the model files
    
    Args:
        settings: Dictionary containing model configuration:
            - model_dir_path (str, optional): User-provided path to models
            - dcnn_model_gdrive_id (str, optional): Google Drive file ID for download
            - default_model_dir (str, optional): Default directory name
        logger: Optional logger instance for status messages
        
    Returns:
        str: Path to the directory containing model files, or None if unavailable
        
    Example:
        >>> settings = {
        ...     'model_dir_path': '/path/to/models',  # Optional
        ...     'dcnn_model_gdrive_id': '16LFMIEADO3XI8uNqiUoKKlrzWlc1_Q-p',
        ...     'default_model_dir': 'dcnn_trained'
        ... }
        >>> model_path = get_or_download_atomistic_model(settings)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Default values
    DCNN_MODEL_GDRIVE_ID = settings.get('dcnn_model_gdrive_id', '16LFMIEADO3XI8uNqiUoKKlrzWlc1_Q-p')
    DEFAULT_MODEL_DIR = settings.get('default_model_dir', "dcnn_trained")
    
    # 1. Check user-provided path
    user_provided_path = settings.get('model_dir_path')
    if user_provided_path:
        if not os.path.isdir(user_provided_path):
            logger.error(f"Provided 'model_dir_path' ('{user_provided_path}') does not exist.")
            return None
        logger.info(f"Using user-provided model path: {user_provided_path}")
        return user_provided_path
    
    # 2. Check default path
    default_path = DEFAULT_MODEL_DIR
    
    if not os.path.isdir(default_path):
        logger.warning(f"Default model directory '{default_path}' not found. Downloading...")
        
        # Download the model
        success = _download_and_extract_model(
            gdrive_id=DCNN_MODEL_GDRIVE_ID,
            output_dir=default_path,
            logger=logger
        )
        
        if not success:
            logger.error("Failed to download and extract the model.")
            return None
    
    # 3. Locate model files
    model_path = _locate_model_files(default_path, logger)
    
    if model_path:
        logger.info(f"Using atomistic models from: {model_path}")
    else:
        logger.error(f"Could not find model files in '{default_path}' or subdirectories.")
    
    return model_path


def _download_and_extract_model(gdrive_id: str, output_dir: str, logger: logging.Logger) -> bool:
    """
    Download model from Google Drive and extract it.
    
    Args:
        gdrive_id: Google Drive file ID
        output_dir: Directory to extract model files to
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    from . import atomistic_tools  # Import from current package
    
    zip_filename = f"{output_dir}.zip"
    
    # Download
    logger.info(f"Downloading model from Google Drive (ID: {gdrive_id})...")
    downloaded_zip_path = atomistic_tools.download_file_with_gdown(
        gdrive_id, zip_filename, logger
    )
    
    if not downloaded_zip_path or not os.path.exists(downloaded_zip_path):
        logger.error("Failed to download the model.")
        return False
    
    # Extract
    logger.info(f"Extracting model to {output_dir}...")
    unzip_success = atomistic_tools.unzip_file(downloaded_zip_path, output_dir, logger)
    
    # Cleanup zip file
    try:
        os.remove(downloaded_zip_path)
        logger.info(f"Cleaned up downloaded zip file: {downloaded_zip_path}")
    except OSError as e:
        logger.warning(f"Could not remove zip file {downloaded_zip_path}: {e}")
    
    return unzip_success


def _locate_model_files(search_dir: str, logger: logging.Logger) -> str | None:
    """
    Locate atomnet3*.tar files in the given directory or subdirectories.
    
    Args:
        search_dir: Directory to search in
        logger: Logger instance
        
    Returns:
        str: Path to directory containing model files, or None if not found
    """
    try:
        # Check if models are directly in the search directory
        if glob.glob(os.path.join(search_dir, 'atomnet3*.tar')):
            return search_dir
        
        # Check subdirectories
        for item in os.listdir(search_dir):
            sub_path = os.path.join(search_dir, item)
            if os.path.isdir(sub_path) and glob.glob(os.path.join(sub_path, 'atomnet3*.tar')):
                logger.info(f"Found models in nested directory: {sub_path}")
                return sub_path
                
    except FileNotFoundError:
        logger.error(f"The model directory '{search_dir}' does not exist.")
    
    return None


def validate_model_directory(model_dir: str) -> bool:
    """
    Validate that a directory contains the required atomistic model files.
    
    Args:
        model_dir: Path to directory to validate
        
    Returns:
        bool: True if directory contains valid model files
        
    Example:
        >>> if validate_model_directory('/path/to/models'):
        ...     print("Models found!")
    """
    if not os.path.isdir(model_dir):
        return False
    
    # Look for atomnet3*.tar files
    model_files = glob.glob(os.path.join(model_dir, 'atomnet3*.tar'))
    
    return len(model_files) > 0


def get_model_info(model_dir: str) -> dict:
    """
    Get information about the models in a directory.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        dict: Information about the models including:
            - model_files: List of model file paths
            - num_models: Number of model files found
            - total_size_mb: Total size of model files in MB
            
    Example:
        >>> info = get_model_info('/path/to/models')
        >>> print(f"Found {info['num_models']} models")
    """
    if not os.path.isdir(model_dir):
        return {
            'model_files': [],
            'num_models': 0,
            'total_size_mb': 0.0,
            'error': 'Directory not found'
        }
    
    model_files = glob.glob(os.path.join(model_dir, 'atomnet3*.tar'))
    
    total_size = 0
    for file_path in model_files:
        try:
            total_size += os.path.getsize(file_path)
        except OSError:
            pass
    
    return {
        'model_files': model_files,
        'num_models': len(model_files),
        'total_size_mb': total_size / (1024 * 1024),
        'directory': model_dir
    }