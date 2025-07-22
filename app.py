import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import zipfile
import io
import os
from image_analyzer import ImageQualityAnalyzer
from utils import create_download_zip, organize_images_by_quality, extract_images_from_zip

def main():
    st.set_page_config(
        page_title="Bulk Image Quality Assessment",
        page_icon="📸",
        layout="wide"
    )
    
    st.title("📸 Bulk Image Quality Assessment System")
    st.markdown("Upload multiple images to analyze their quality and sort them based on exposure, blur, and streak detection.")
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = None
    
    # File upload section
    st.header("📁 Upload Images")
    st.info("💡 **Tip**: You can select hundreds of images at once or upload a ZIP file containing images! The system supports up to 1GB total upload size.")
    
    # Add tabs for different upload methods
    upload_tab1, upload_tab2 = st.tabs(["📸 Upload Images", "📦 Upload ZIP File"])
    
    uploaded_files = None
    
    with upload_tab1:
        uploaded_files = st.file_uploader(
            "Choose image files (select multiple images using Ctrl/Cmd + click)",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            accept_multiple_files=True,
            help="Supported formats: PNG, JPG, JPEG, BMP, TIFF, WebP. Maximum total size: 1GB. Select multiple files using Ctrl/Cmd + click or Ctrl/Cmd + A to select all."
        )
    
    with upload_tab2:
        zip_file = st.file_uploader(
            "Choose a ZIP file containing images (up to 1GB)",
            type=['zip'],
            help="Upload a ZIP file containing images. The system will extract and process all supported image formats. Maximum size: 1GB."
        )
        
        if zip_file:
            # Check ZIP file size
            zip_size_mb = len(zip_file.getvalue()) / (1024 * 1024)
            st.info(f"📦 ZIP file size: {zip_size_mb:.1f} MB")
            
            if zip_size_mb > 1024:  # 1GB limit
                st.error("❌ ZIP file is too large. Maximum size is 1GB (1024 MB).")
                uploaded_files = None
            else:
                with st.spinner("Extracting images from ZIP file..."):
                    extracted_files = extract_images_from_zip(zip_file.getvalue())
                
                if extracted_files:
                    st.success(f"✅ Successfully extracted {len(extracted_files)} images from ZIP file!")
                    
                    # Convert extracted files to format expected by the rest of the app
                    class FakeUploadedFile:
                        def __init__(self, filename, data):
                            self.name = filename
                            self._data = data
                        
                        def getvalue(self):
                            return self._data
                        
                        def read(self):
                            return self._data
                    
                    uploaded_files = [FakeUploadedFile(f['filename'], f['data']) for f in extracted_files]
                    
                    # Show file list
                    with st.expander(f"📋 View all {len(uploaded_files)} extracted files"):
                        for i, file in enumerate(uploaded_files, 1):
                            file_size_mb = len(file.getvalue()) / (1024 * 1024)
                            st.text(f"{i:3d}. {file.name} ({file_size_mb:.1f} MB)")
                else:
                    st.error("❌ No supported images found in the ZIP file. Make sure your ZIP contains PNG, JPG, JPEG, BMP, TIFF, or WebP files.")
                    uploaded_files = None
    
    if uploaded_files:
        # Calculate total file size
        total_size_mb = sum(len(file.getvalue()) for file in uploaded_files) / (1024 * 1024)
        
        st.success(f"✅ {len(uploaded_files)} images uploaded successfully! (Total size: {total_size_mb:.1f} MB)")
        
        # Show file list if there are many files
        if len(uploaded_files) > 10:
            with st.expander(f"📋 View all {len(uploaded_files)} uploaded files"):
                for i, file in enumerate(uploaded_files, 1):
                    file_size_mb = len(file.getvalue()) / (1024 * 1024)
                    st.text(f"{i:3d}. {file.name} ({file_size_mb:.1f} MB)")
        
        # Show warning for very large batches
        if len(uploaded_files) > 100:
            st.warning(f"⚠️ You've uploaded {len(uploaded_files)} images. Processing may take several minutes. Consider analyzing in smaller batches for faster results.")
        
        # Analyze button with additional warning
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔍 Analyze Images", type="primary"):
                analyze_images(uploaded_files)
        with col2:
            if len(uploaded_files) > 50:
                st.error("⚠️ Large batch detected! For best results, try smaller batches (under 50 images).")
    
    # Display results if analysis is complete
    if st.session_state.analysis_results is not None:
        display_results()
        display_download_options()

def analyze_images(uploaded_files):
    """Analyze uploaded images for quality issues"""
    try:
        analyzer = ImageQualityAnalyzer()
        results = []
        
        # Add memory management warning and batch processing
        batch_size = 10  # Smaller batches to prevent memory crashes
        if len(uploaded_files) > batch_size:
            st.info(f"📦 Processing {len(uploaded_files)} images in batches of {batch_size} to prevent memory issues.")
        
        # Limit total number of images to prevent crashes
        max_images = 100
        if len(uploaded_files) > max_images:
            st.warning(f"⚠️ Too many images ({len(uploaded_files)}). Processing only the first {max_images} images to prevent crashes.")
            uploaded_files = uploaded_files[:max_images]
        
        # Progress tracking with estimated time
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_estimate = st.empty()
        
        import time
        import gc  # Garbage collection for memory management
        start_time = time.time()
        
    except Exception as init_error:
        st.error(f"Failed to initialize image analyzer: {str(init_error)}")
        return
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress with time estimation
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
            
            # Calculate estimated time remaining
            if i > 0:
                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / i
                remaining_images = len(uploaded_files) - i
                estimated_remaining = avg_time_per_image * remaining_images
                time_estimate.text(f"⏱️ Estimated time remaining: {estimated_remaining:.1f} seconds")
            
            # Read image with better error handling
            try:
                image_bytes = uploaded_file.read()
                if len(image_bytes) == 0:
                    raise ValueError("Empty file")
                    
                image = Image.open(io.BytesIO(image_bytes))
                
                # Validate image
                if image.mode not in ['RGB', 'RGBA', 'L']:
                    image = image.convert('RGB')
                
                # Aggressive image size reduction to prevent memory issues
                max_size = 2000  # Further reduced max size
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Additional memory check - force smaller images
                image_array = np.array(image)
                memory_mb = image_array.nbytes / (1024 * 1024)
                if memory_mb > 20:  # Much stricter 20MB limit per image
                    scale = 0.5  # More aggressive scaling
                    new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Final size check
                if max(image.size) > 1500:
                    image = image.resize((1500, 1500), Image.Resampling.LANCZOS)
                
                # Convert PIL image to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Analyze image quality
                analysis = analyzer.analyze_image(opencv_image, uploaded_file.name)
                
                # Don't store large image data to save memory - only store if small
                if len(image_bytes) < 5 * 1024 * 1024:  # Only store if less than 5MB
                    analysis['image_data'] = image_bytes
                    analysis['pil_image'] = image
                else:
                    analysis['image_data'] = None
                    analysis['pil_image'] = None
                
                results.append(analysis)
                
                # Force garbage collection every 5 images to free memory
                if (i + 1) % 5 == 0:
                    gc.collect()
                    
                # Clear variables to free memory
                del image_array, opencv_image
                
            except Exception as img_error:
                st.warning(f"⚠️ Skipping corrupted or unsupported image: {uploaded_file.name}")
                results.append({
                    'filename': uploaded_file.name,
                    'overall_quality': 'Error',
                    'issues': [f"Image error: {str(img_error)}"],
                    'scores': {},
                    'image_data': None,
                    'pil_image': None
                })
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            results.append({
                'filename': uploaded_file.name,
                'overall_quality': 'Error',
                'issues': [f"Processing error: {str(e)}"],
                'scores': {},
                'image_data': None,
                'pil_image': None
            })
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    time_estimate.empty()
    
    # Final garbage collection
    gc.collect()
    
    # Store results in session state
    st.session_state.analysis_results = results
    st.session_state.uploaded_images = uploaded_files
    
    # Show completion summary
    total_time = time.time() - start_time
    successful = len([r for r in results if r['overall_quality'] != 'Error'])
    st.success(f"✅ Analysis complete! Successfully processed {successful}/{len(uploaded_files)} images in {total_time:.1f} seconds")
    st.rerun()

def display_results():
    """Display analysis results in a organized format"""
    st.header("📊 Analysis Results")
    
    results = st.session_state.analysis_results
    
    # Create summary statistics
    total_images = len(results)
    good_quality = len([r for r in results if r['overall_quality'] == 'Good'])
    poor_quality = len([r for r in results if r['overall_quality'] == 'Poor'])
    errors = len([r for r in results if r['overall_quality'] == 'Error'])
    
    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Good Quality", good_quality, delta=f"{(good_quality/total_images*100):.1f}%")
    with col3:
        st.metric("Poor Quality", poor_quality, delta=f"{(poor_quality/total_images*100):.1f}%")
    with col4:
        st.metric("Processing Errors", errors)
    
    # Create detailed results table
    df_data = []
    for result in results:
        issues_str = ", ".join(result['issues']) if result['issues'] else "None"
        scores_str = ", ".join([f"{k}: {v:.2f}" for k, v in result['scores'].items()]) if result['scores'] else "N/A"
        
        df_data.append({
            'Filename': result['filename'],
            'Quality': result['overall_quality'],
            'Issues Detected': issues_str,
            'Scores': scores_str
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Filter options
    st.subheader("🔍 Filter Results")
    filter_option = st.selectbox(
        "Filter by quality:",
        ["All Images", "Good Quality", "Poor Quality", "Overexposed", "Underexposed", "Blurry", "Streaky", "Errors"]
    )
    
    # Filter results based on selection
    filtered_results = filter_results_by_option(results, filter_option)
    
    # Display filtered images
    if filtered_results:
        st.subheader(f"📷 {filter_option} ({len(filtered_results)} images)")
        
        # Display images in grid
        cols_per_row = 3
        for i in range(0, len(filtered_results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(filtered_results):
                    result = filtered_results[i + j]
                    with col:
                        if result['pil_image'] is not None:
                            st.image(result['pil_image'], caption=result['filename'], use_column_width=True)
                            
                            # Quality badge
                            if result['overall_quality'] == 'Good':
                                st.success(f"✅ {result['overall_quality']}")
                            elif result['overall_quality'] == 'Poor':
                                st.error(f"❌ {result['overall_quality']}")
                            else:
                                st.warning(f"⚠️ {result['overall_quality']}")
                            
                            # Issues
                            if result['issues']:
                                st.caption(f"Issues: {', '.join(result['issues'])}")
                            
                            # Scores
                            if result['scores']:
                                with st.expander("📈 Detailed Scores"):
                                    for metric, score in result['scores'].items():
                                        st.metric(metric.replace('_', ' ').title(), f"{score:.3f}")
                        else:
                            st.error(f"❌ {result['filename']}")
                            st.caption("Image could not be processed")

def filter_results_by_option(results, filter_option):
    """Filter results based on the selected option"""
    if filter_option == "All Images":
        return results
    elif filter_option == "Good Quality":
        return [r for r in results if r['overall_quality'] == 'Good']
    elif filter_option == "Poor Quality":
        return [r for r in results if r['overall_quality'] == 'Poor']
    elif filter_option == "Overexposed":
        return [r for r in results if 'Overexposed' in r['issues']]
    elif filter_option == "Underexposed":
        return [r for r in results if 'Underexposed' in r['issues']]
    elif filter_option == "Blurry":
        return [r for r in results if 'Blurry' in r['issues']]
    elif filter_option == "Streaky":
        return [r for r in results if 'Streaky' in r['issues']]
    elif filter_option == "Errors":
        return [r for r in results if r['overall_quality'] == 'Error']
    return results

def display_download_options():
    """Display download options for processed images"""
    st.header("📥 Download Options")
    
    results = st.session_state.analysis_results
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Bulk Downloads")
        
        # Download all good quality images
        good_images = [r for r in results if r['overall_quality'] == 'Good' and r['image_data'] is not None]
        if good_images:
            zip_data = create_download_zip(good_images, "good_quality")
            st.download_button(
                label=f"📥 Download Good Quality Images ({len(good_images)} files)",
                data=zip_data,
                file_name="good_quality_images.zip",
                mime="application/zip"
            )
        
        # Download by issue type
        issue_types = ['Overexposed', 'Underexposed', 'Blurry', 'Streaky']
        for issue_type in issue_types:
            issue_images = [r for r in results if issue_type in r['issues'] and r['image_data'] is not None]
            if issue_images:
                zip_data = create_download_zip(issue_images, issue_type.lower())
                st.download_button(
                    label=f"📥 Download {issue_type} Images ({len(issue_images)} files)",
                    data=zip_data,
                    file_name=f"{issue_type.lower()}_images.zip",
                    mime="application/zip"
                )
    
    with col2:
        st.subheader("📋 Analysis Report")
        
        # Create comprehensive report
        report_data = create_analysis_report(results)
        st.download_button(
            label="📥 Download Analysis Report (CSV)",
            data=report_data,
            file_name="image_quality_analysis_report.csv",
            mime="text/csv"
        )
        
        # Download all images organized by quality
        if results:
            organized_zip = organize_images_by_quality(results)
            st.download_button(
                label=f"📥 Download All Images (Organized)",
                data=organized_zip,
                file_name="organized_images.zip",
                mime="application/zip"
            )

def create_analysis_report(results):
    """Create a detailed CSV report of the analysis"""
    report_data = []
    
    for result in results:
        row = {
            'Filename': result['filename'],
            'Overall_Quality': result['overall_quality'],
            'Issues': '; '.join(result['issues']) if result['issues'] else 'None'
        }
        
        # Add scores
        for metric, score in result['scores'].items():
            row[metric] = score
        
        report_data.append(row)
    
    df = pd.DataFrame(report_data)
    return df.to_csv(index=False)

if __name__ == "__main__":
    main()
