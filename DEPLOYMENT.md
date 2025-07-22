# 🚀 Deploy to Streamlit Community Cloud

This guide will help you deploy your Image Quality Assessment System to Streamlit Community Cloud for free hosting.

## Prerequisites

1. **GitHub Account**: You need a GitHub repository with your code
2. **Streamlit Community Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)

## Step 1: Prepare Files for Deployment

Your project is already prepared with the following files needed for Streamlit Community Cloud:

### Required Files (Already Created):
- `streamlit_requirements.txt` - Python dependencies (rename to `requirements.txt` when uploading to GitHub)
- `packages.txt` - System packages for OpenCV support
- `.streamlit/config.toml` - Streamlit configuration
- `.streamlit/secrets.toml` - For future secrets (currently empty)

### File Structure:
```
your-repo/
├── app.py                    # Main Streamlit app
├── image_analyzer.py         # Core analysis engine  
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies (rename from streamlit_requirements.txt)
├── packages.txt            # System packages
├── .streamlit/
│   ├── config.toml        # App configuration
│   └── secrets.toml       # Secrets (empty for now)
├── docs/
│   └── index.html         # GitHub Pages landing page
├── test_images/           # Sample images
└── README.md             # Documentation
```

## Step 2: Upload to GitHub

1. **Create a new repository** on GitHub (e.g., `image-quality-assessment`)

2. **Upload all files** to your GitHub repository:
   - Copy all project files to your GitHub repo
   - **Important**: Rename `streamlit_requirements.txt` to `requirements.txt`

3. **Commit and push** your changes

## Step 3: Deploy to Streamlit Community Cloud

1. **Go to** [share.streamlit.io](https://share.streamlit.io)

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Configure deployment**:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom URL (e.g., `image-quality-sorting`)

5. **Click "Deploy"**

6. **Wait for deployment** (usually 2-5 minutes)

7. **Your app will be live** at: `https://your-app-name.streamlit.app`

## Step 4: Update GitHub Pages (Optional)

Update your GitHub Pages landing page to point to your new Streamlit Community Cloud URL:

1. Edit `docs/index.html`
2. Replace the Replit links with your new Streamlit Community Cloud URL
3. Push changes to GitHub

## Dependencies Included:

- `streamlit>=1.47.0` - Web framework
- `opencv-python-headless>=4.11.0` - Image processing (headless version for cloud)
- `numpy>=2.3.1` - Numerical computing
- `pandas>=2.3.1` - Data handling
- `pillow>=11.3.0` - Additional image processing

## System Packages:

- `libgl1-mesa-glx` - OpenGL support for OpenCV
- `libglib2.0-0` - Required system library

## Features Supported:

✅ Multi-file image upload (up to 1GB)  
✅ ZIP file extraction and processing  
✅ Blur, exposure, and streak detection  
✅ Organized download of results  
✅ Real-time progress tracking

## Troubleshooting:

**Build Errors**: Check the logs in Streamlit Community Cloud dashboard
**OpenCV Issues**: The `packages.txt` file should resolve most OpenCV dependencies
**Memory Issues**: Large file processing may hit memory limits on free tier

## Cost: Completely Free! 🎉

Streamlit Community Cloud provides:
- Free hosting
- Auto-deployment from GitHub
- Custom subdomain
- No credit card required