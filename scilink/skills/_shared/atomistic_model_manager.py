"""
Atomistic model management utilities.

This module handles downloading, locating, and managing the DCNN models
used for atomic-resolution microscopy analysis.
"""

import os
import glob
import logging
from typing import Optional


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
    
    # Default values. The weights are a release asset of the SciLink
    # repository (first choice: a plain HTTPS download that CI builds and
    # containers can rely on); the Google Drive copy is the fallback.
    DCNN_MODEL_URL = (settings.get('dcnn_model_url')
                      or os.environ.get('SCILINK_DCNN_MODEL_URL')
                      or DEFAULT_DCNN_MODEL_URL)
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
    
    # 2. Check default path. The default is a PERSISTENT per-user cache, not
    # the current directory: generated analysis scripts run in a fresh per-item
    # working directory, so a relative default downloaded the ~770 MB ensemble
    # again into EVERY image's folder (observed live: 11 frames of a live loop
    # took 36 s each and filled a disk with 12 GB of identical weights). A
    # ``dcnn_trained`` folder already in the working directory still wins, so
    # an existing setup keeps working.
    default_path = DEFAULT_MODEL_DIR
    if not os.path.isabs(default_path) and not os.path.isdir(default_path):
        default_path = _persistent_model_dir(default_path)

    if not os.path.isdir(default_path) or not _locate_model_files(default_path, logging.getLogger("quiet")):
        logger.warning(f"Default model directory '{default_path}' not found. Downloading...")
        
        # Download the model
        success = _download_and_extract_model(
            gdrive_id=DCNN_MODEL_GDRIVE_ID,
            output_dir=default_path,
            logger=logger,
            url=DCNN_MODEL_URL,
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


def _persistent_model_dir(name: str) -> str:
    """``<SCILINK_HOME or ~/.scilink>/models/<name>``: downloaded once, shared by
    every analysis on this machine."""
    from pathlib import Path
    home = Path(os.environ.get("SCILINK_MODELS") or "").expanduser() if os.environ.get("SCILINK_MODELS") \
        else Path(os.environ.get("SCILINK_HOME") or (Path.home() / ".scilink")).expanduser() / "models"
    home.mkdir(parents=True, exist_ok=True)
    return str(home / name)


DEFAULT_DCNN_MODEL_URL = ("https://github.com/ziatdinovmax/SciLink/releases/download/"
                          "models-dcnn-v1/dcnn_trained.zip")


def _download_url(url: str, dest: str, logger: logging.Logger) -> Optional[str]:
    """Stream ``url`` to ``dest``; the path on success, ``None`` on any failure
    (a partial file is removed). Plain HTTPS, no third-party client."""
    import shutil
    import urllib.request
    tmp = dest + ".part"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh, length=1 << 20)
        os.replace(tmp, dest)
        logger.info(f"Downloaded {os.path.getsize(dest) / 1e6:.0f} MB from {url}")
        return dest
    except Exception as exc:  # noqa: BLE001 - the caller falls back
        logger.warning(f"Download from {url} failed: {exc}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None


def _download_and_extract_model(gdrive_id: str, output_dir: str, logger: logging.Logger,
                                url: Optional[str] = None) -> bool:
    """
    Download the model archive and extract it.

    ``url`` (a plain HTTPS location, normally the SciLink release asset) is
    tried first; the Google Drive file ``gdrive_id`` is the fallback.

    Args:
        gdrive_id: Google Drive file ID (fallback source)
        output_dir: Directory to extract model files to
        logger: Logger instance
        url: HTTPS location of the same zip (first choice), or None

    Returns:
        bool: True if successful, False otherwise
    """
    # atomistic_tools.py was renamed to atomic_stem.py and moved into the
    # atomic_stem skill bundle. Imported with the legacy name to keep the
    # call sites below stable.
    from scilink.skills.image_analysis.atomic_stem import atomic_stem as atomistic_tools
    
    zip_filename = f"{output_dir}.zip"
    
    # Download: the release asset first, Google Drive as the fallback
    downloaded_zip_path = None
    if url:
        logger.info(f"Downloading model from {url} ...")
        downloaded_zip_path = _download_url(url, zip_filename, logger)
    if not downloaded_zip_path:
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