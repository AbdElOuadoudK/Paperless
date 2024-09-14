from .utils import *
from .algorithm import ResumeParser
from .globals import RESUMES_PATH, LOGS_PATH, resume

__all__ = [
    # Global path constants
    'RESUMES_PATH',       # Resumes directory
    'LOGS_PATH',          # Main logs directory
    'resume',             # Upload resume from path
    
    # Utility functions for image processing
    'display',            # Display an image
    'writeDisplay',       # Write and display an image with optional saving
    'greyscale',          # Convert an image to greyscale
    'thinning',           # Apply thinning algorithm to an image
    'noise_removal',      # Remove noise from an image
    'medianCanny',        # Apply Canny edge detection with a median filter

    # Visualization and box processing
    'display_',           # Display image with detected bounding boxes
    'overlap_1',          # Check if one box overlaps another in one dimension
    'overlap_2',          # Check if one box overlaps another in two dimensions
    'getAllOverlaps',      # Get all overlapping boxes
    'iter_boxes',         # Iterate through and merge overlapping boxes

    # Line and box-related functions
    'attach_lines_',      # Attach lines to bounding boxes
    'get_lines',          # Extract lines from processed image
    'merge_clusters',     # Merge clusters of boxes
    'get_boxes',          # Get boxes from an image after processing
    'scan_img',           # Scan an image to detect boxes
    'retrieve_cropped',   # Retrieve cropped image based on bounding box

    # Box manipulation
    'unpack',             # Unpack a box into coordinates
    'pack',               # Pack coordinates into a box

    # ResumeParser class
    'ResumeParser',       # Main class for processing resumes
]
