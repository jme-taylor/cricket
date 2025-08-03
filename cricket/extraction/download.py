"""Download module for fetching and extracting cricket data from Cricsheet."""

import logging
import requests
import zipfile
from pathlib import Path
from typing import Optional

from ..constants import RAW_DATA_URL, INPUT_DATA_FOLDER, DATA_FOLDER

logger = logging.getLogger(__name__)


def download_file(
    url: str, 
    destination: Path,
    chunk_size: int = 8192
) -> None:
    """
    Download a file from a URL to a destination path.

    Doesn't return anything, just downloads the file to the destination path.
    
    Parameters
    ----------
    url : str
        The URL to download from
    destination : Path, optional
        Path where the file should be saved, by default INPUT_DATA_FOLDER
    chunk_size : int, optional
        Size of chunks to download at a time (bytes), by default 8192
    """
    logger.info(f"Starting download from {url}")
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                downloaded_size += len(chunk)
                
                if total_size > 0 and downloaded_size % (10 * 1024 * 1024) == 0:
                    progress = (downloaded_size / total_size) * 100
                    logger.info(f"Download progress: {progress:.1f}%")
    
    logger.info(f"Download completed: {destination}")



def extract_zip(
    zip_path: Path,
    extract_to: Path = DATA_FOLDER, 
    remove_zip: bool = True
) -> None:
    """
    Extract a zip file to a destination directory.

    Parameters
    ----------
    zip_path : Path
        Path to the zip file
    extract_to : Path, optional
        Directory to extract files to, by default DATA_FOLDER
    remove_zip : bool, optional
        Whether to remove the zip file after extraction, by default True
    """
    logger.info(f"Extracting {zip_path} to {extract_to}")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    
        file_list = zip_ref.namelist()
        logger.info(f"Zip contains {len(file_list)} files")
        
        zip_ref.extractall(extract_to)
        
    logger.info(f"Extraction completed to {extract_to}")
    
    if remove_zip:
        zip_path.unlink()
        logger.info(f"Removed zip file: {zip_path}")
        


def download_and_extract_zipped_cricket_data() -> None:
    """
    Download and extract cricket data from Cricsheet.
    
    Parameters
    ----------
    url : str, optional
        The URL to download from, by default RAW_DATA_URL
    destination_folder : Path, optional
        Directory to extract files to, by default DATA_FOLDER
    remove_zip : bool, optional
        Whether to remove the zip file after extraction, by default True
    """
    logger.info("Starting cricket data download and extraction")
    
    zip_filename = RAW_DATA_URL.split('/')[-1]
    zip_file_path = INPUT_DATA_FOLDER / zip_filename

    download_file(RAW_DATA_URL, zip_file_path)
    extract_zip(zip_file_path)
    logger.info("Cricket data download and extraction completed successfully")
        

