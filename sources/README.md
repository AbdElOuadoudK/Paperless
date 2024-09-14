# Project Environment Setup

This `Paperless: REsume PArsing` project is designed to run efficiently on a specific environment setup, ensuring compatibility between the various tools, libraries, and frameworks used. Below is a detailed description of the required environment, operating system, and Python settings.

_Khouri A. Ouadoud - k.abd_elouadoud@univ-boumerdes.dz_


## Sub-directories mapping

the sources directory contains three main sub-directories named; `scripts`, `data` and `logs`. Defining separate directories rather than using a single directory offers several advantages. One and foremost, having dedicated directories prevents clutter and makes it easier to expand each component without losing track of files as the project grows. Additionally,eEach directory has a specific purpose, making it easier to find and manage files. This organization helps developers quickly understand the project's structure.  

- **Scripts**:  This directory serves as the primary repository for all source code related to the project. It includes various scripts and Jupyter notebooks that facilitate data processing, Information Extraction, and evaluation. The organization within this directory allows for easy navigation and modification of the codebase, supporting both development and experimentation.

- **Data**:  The data directory is dedicated to storing all relevant input data for the project. It contains a variety of file types, including resumes (PNGs, PDFs, ...etc) and other document such as conextual_keyword (array, JSON ...etc). Additionally, this directory may hold temporary data files that are generated during processing. The structured organization of this directory is crucial for efficient data management and retrieval during various stages of the project.


- **Logs**:  The logs directory is designed to capture and store all logging outputs generated during the execution of the project. This includes results from various operations, debug messages, and derived files that result from data processing or model evaluation. By keeping a comprehensive log, this directory aids in tracking performance, identifying issues, and providing insights for further development and optimization of the project.

## Environment settings

- **Operating System**:  
   The recommended operating system for this project is **Ubuntu 20.04 LTS** or any Linux-based distribution. However, it is also compatible with other platforms such as macOS or Windows (with minor adjustments in some dependencies).

- **Development Tool**:  
   The project was primarily developed and tested using **Jupyter Lab**, a web-based interactive development environment perfect for running notebooks and visualizing data in real-time.


## Dependencies

- **Python**:   
  The project uses **Python 3.12.3** as the base language. It's recommended to install the same version to ensure compatibility with all libraries and frameworks.

- **Libraries and Frameworks**:  
  The following Python libraries and frameworks are crucial for the proper functioning of the project:
  1. **NumPy**: For numerical and array operations.
  2. **OpenCV**: For image processing and computer vision tasks.
  3. **Pytesseract**: A wrapper for Google's Tesseract-OCR engine, used for OCR tasks.
  4. **Scikit-learn**: A popular machine learning library that is used for the clustering process.

**DISCLAIMER**

   It should be noted that the current directory is a primary version of the project, thus may contains some issues regarding the developpement/programming phase. Therefore, if you encounter any bug, logical or syntaxical error, please be aware of the challenges and difficulties in handling certain image distortions, where text detection might fail due to overlapping or skewed text.


**NOTE -** If in any cirmuctances, you want to get a copy of the experiment into a particular plateform, please make sure to contact the **authors** of the project for further authority.

To streamline the installation process, you can install all dependencies using the provided requirements.txt file in the repository:
```bash
pip install -r requirements.txt






















