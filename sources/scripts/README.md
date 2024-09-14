# Scripts Overview

The **scripts** directory serves as the core hub for the image processing and document parsing functionalities used in this project. This directory includes an organized structure to guide users through the various steps involved in extracting and processing content from image-based documents like resumes.

_Khouri A. Ouadoud - k.abd_elouadoud@univ-boumerdes.dz_

## Contents

1. **imagproc_pkg**:  
   This is a Python package that contains all the core image processing utilities, methods, and functions used throughout this project. It handles tasks such as image enhancement, line detection, bounding box extraction, and optical character recognition (OCR). The package provides a comprehensive set of tools that streamline document image processing, making it ideal for use in parsing complex document formats like resumes.

   The `imagproc_pkg` package offers a modular and extensible architecture for users looking to integrate image processing steps in their projects. All functionalities within this package are designed to work cohesively, providing a robust foundation for detecting and extracting text from image-based files.

2. **main_process.ipynb**:  
   A Jupyter notebook that demonstrates the entire image processing pipeline in action. This notebook:
   - Loads a sample image (in this case, a resume).
   - Applies the `ResumeParser` class to detect and crop important areas of the document.
   - Uses OCR (`pytesseract`) to extract text from the cropped images.
   - Displays the text and images for each detected section of the document.

   Each step in the notebook is well-commented and provides insights into how each part of the `imagproc_pkg` package works together to complete the task of image parsing and text extraction.

**DISCLAIMER**

   It should be noted that the current directory is a primary version of the project, thus may contains some issues regarding the developpement/programming phase. Therefore, if you encounter any bug, logical or syntaxical error, please be aware of the challenges and difficulties in handling certain image distortions, where text detection might fail due to overlapping or skewed text.


**NOTE -** If in any cirmuctances, you want to get a copy of the experiment into a particular plateform, please make sure to contact the **authors** of the project for further authority.

### How to Use the Package
For more detailed information, refer to the `documentation` directory included in the project repository, where you will find a deeper explanation of the package's functionalities and use cases.

To clone the repository to your local machine
   ```bash
   git clone https://github.com/AbdElOuadoudK/Paperless.git
