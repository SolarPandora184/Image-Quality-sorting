# 📸 Bulk Image Quality Assessment System

A professional Streamlit-based web application for bulk image quality assessment. This system analyzes multiple uploaded images to detect quality issues such as blur, overexposure, underexposure, and streaks.

## 🌟 Features

- **Bulk Processing**: Analyze hundreds of images at once
- **ZIP File Support**: Upload ZIP files containing images for automatic extraction and processing
- **Quality Detection**: Advanced algorithms to detect:
  - Blur using Laplacian variance
  - Overexposure and underexposure
  - Streak artifacts and motion blur
- **Smart Organization**: Automatically categorize images by quality
- **Download Options**: Export organized image sets as ZIP files
- **User-Friendly Interface**: Clean, responsive web interface
- **Real-time Progress**: Track processing with progress bars and time estimates

## 🚀 Quick Start

### Running on Replit
1. Open the project in Replit
2. The Streamlit app will automatically start
3. Access the application at the provided URL

### Running Locally
1. Install dependencies:
   ```bash
   pip install streamlit opencv-python numpy pillow pandas
   ```
2. Run the application:
   ```bash
   streamlit run app.py --server.port 5000
   ```
3. Open your browser to `http://localhost:5000`

### GitHub Pages
This repository includes a GitHub Pages setup:
1. Go to repository Settings > Pages
2. Set source to "Deploy from a branch"
3. Select the `main` branch and `/docs` folder
4. The landing page will be available at your GitHub Pages URL

## 📁 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── image_analyzer.py      # Core image quality analysis engine
├── utils.py              # Utility functions for file operations
├── create_test_images.py # Tool for generating test images
├── docs/
│   └── index.html        # GitHub Pages landing page
├── test_images/          # Sample images for testing
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

## 🔧 Configuration

The application uses the following configuration in `.streamlit/config.toml`:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

## 📊 How It Works

1. **Upload Phase**: Users can upload multiple images or ZIP files
2. **Extraction**: ZIP files are automatically extracted to find image files
3. **Analysis**: Each image is processed through quality assessment algorithms
4. **Categorization**: Images are sorted into quality categories
5. **Results**: Visual display of results with download options

## 🎯 Quality Assessment Criteria

- **Blur Detection**: Uses Laplacian variance to detect motion blur and focus issues
- **Exposure Analysis**: Analyzes RGB histogram distribution to detect over/underexposure
- **Streak Detection**: Identifies motion artifacts and streaking patterns

## 🌐 Deployment

### Replit Deployment
- Automatically configured for Replit environment
- Uses proper server configuration for external access

### GitHub Pages Integration
- Landing page provides project information
- Redirects users to the live application
- Professional presentation for portfolio use

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.