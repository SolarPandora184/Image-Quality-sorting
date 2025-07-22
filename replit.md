# Bulk Image Quality Assessment System

## Overview

This is a Streamlit-based web application for bulk image quality assessment. The system analyzes multiple uploaded images to detect quality issues such as blur, overexposure, underexposure, and streaks. It provides a user-friendly interface for uploading images (including ZIP file support), viewing analysis results, and downloading organized image sets. The project is fully configured for both Replit deployment and GitHub Pages hosting.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework
- **Layout**: Wide layout configuration for better image display
- **State Management**: Streamlit session state for maintaining analysis results and uploaded images across interactions
- **File Handling**: Multi-file upload support with format validation and ZIP file extraction
- **GitHub Pages**: Professional landing page with application redirect functionality

### Backend Architecture
- **Processing Engine**: OpenCV for image processing and analysis
- **Analysis Module**: Custom `ImageQualityAnalyzer` class for quality assessment
- **Utility Functions**: Separate utilities module for file operations and data organization

## Key Components

### Image Quality Analyzer (`image_analyzer.py`)
- **Purpose**: Core image analysis engine
- **Capabilities**: 
  - Blur detection using Laplacian variance
  - Exposure analysis (overexposure/underexposure detection)
  - Streak detection for motion artifacts
- **Thresholds**: Configurable quality assessment thresholds for different issues
- **Output**: Structured analysis results with quality scores and detected issues

### Main Application (`app.py`)
- **Purpose**: Streamlit web interface and application orchestration
- **Features**:
  - Multi-file image upload with format validation
  - Progress tracking during analysis
  - Results visualization and display
  - Download functionality for processed images

### Utilities (`utils.py`)
- **Purpose**: Helper functions for file operations and data organization
- **Functions**:
  - ZIP file creation for downloading image sets
  - ZIP file extraction for bulk image uploads
  - Image organization by quality categories
  - Safe filename handling and path management

### GitHub Pages Integration (`docs/index.html`)
- **Purpose**: Professional landing page for GitHub Pages deployment
- **Features**:
  - Responsive design with modern styling
  - Feature showcase and application description
  - Automatic redirection to live application
  - Mobile-friendly interface

## Data Flow

1. **Upload Phase**: Users upload multiple images directly or through ZIP files via tabbed interface
2. **Extraction Phase**: ZIP files are automatically extracted to find supported image formats
3. **Validation**: File format validation (PNG, JPG, JPEG, BMP, TIFF, WebP)
4. **Analysis Phase**: Each image is processed through the quality analyzer
4. **Processing Steps**:
   - Image conversion to appropriate formats (BGR, grayscale)
   - Blur detection using Laplacian variance
   - Exposure analysis on RGB channels
   - Streak detection using specialized algorithms
5. **Results Generation**: Analysis results are compiled with quality scores and issue flags
6. **Visualization**: Results are displayed in the web interface with image previews
7. **Export Phase**: Users can download organized image sets as ZIP files

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **OpenCV (cv2)**: Image processing and computer vision operations
- **NumPy**: Numerical computing for image data manipulation
- **PIL (Pillow)**: Additional image processing capabilities
- **Pandas**: Data manipulation and analysis results handling

### Python Standard Library
- **zipfile**: ZIP archive creation for download functionality
- **io**: In-memory file operations
- **os**: File system operations and path handling

## Deployment Strategy

### Local Development
- **Runtime**: Python-based Streamlit application
- **Execution**: Direct Python execution with `streamlit run app.py`
- **Dependencies**: Managed through requirements.txt or similar package management

### Production Considerations
- **Scalability**: Single-threaded processing suitable for moderate image volumes
- **Memory Management**: Images processed in memory, suitable for typical web upload sizes
- **File Storage**: Temporary in-memory storage for uploaded images and results
- **Performance**: CPU-intensive image processing operations

### Configuration
- **Quality Thresholds**: Configurable through ImageQualityAnalyzer initialization
- **File Formats**: Extensible format support through Streamlit file uploader configuration
- **UI Layout**: Responsive wide layout for optimal image viewing experience

The system is designed as a self-contained web application with no external database requirements, making it suitable for standalone deployment or integration into larger systems.

## Recent Changes (July 22, 2025)

- **Migration to Replit**: Successfully migrated from Replit Agent to standard Replit environment
- **ZIP File Support**: Added capability to upload and extract images from ZIP files
- **GitHub Pages Integration**: Created professional landing page for GitHub Pages deployment
- **Enhanced UI**: Implemented tabbed interface for different upload methods
- **Documentation**: Added comprehensive README.md and project documentation
- **Security**: Ensured proper client/server separation and secure file handling