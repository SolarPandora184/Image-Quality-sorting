import zipfile
import io
from typing import List, Dict, Any
import os

def create_download_zip(images: List[Dict[str, Any]], category: str) -> bytes:
    """
    Create a ZIP file containing images from a specific category.
    
    Args:
        images: List of image analysis results with image_data
        category: Category name for the ZIP file
        
    Returns:
        ZIP file as bytes
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, image_result in enumerate(images):
            if image_result['image_data'] is not None:
                # Create a safe filename
                filename = image_result['filename']
                # Remove any path separators for safety
                filename = os.path.basename(filename)
                
                # Add image to ZIP
                zip_file.writestr(filename, image_result['image_data'])
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def organize_images_by_quality(results: List[Dict[str, Any]]) -> bytes:
    """
    Create a ZIP file with images organized into folders by quality and issues.
    
    Args:
        results: List of all analysis results
        
    Returns:
        ZIP file as bytes with organized folder structure
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create folder structure and organize images
        for result in results:
            if result['image_data'] is not None:
                filename = os.path.basename(result['filename'])
                
                # Determine primary folder based on overall quality
                if result['overall_quality'] == 'Good':
                    folder = "Good_Quality"
                elif result['overall_quality'] == 'Poor':
                    # For poor quality, create subfolders based on primary issue
                    if result['issues']:
                        primary_issue = result['issues'][0]  # Use first issue as primary
                        folder = f"Poor_Quality/{primary_issue}"
                    else:
                        folder = "Poor_Quality/Unknown_Issue"
                else:
                    folder = "Processing_Errors"
                
                # Add image to appropriate folder
                zip_path = f"{folder}/{filename}"
                zip_file.writestr(zip_path, result['image_data'])
                
                # If image has multiple issues, also add to "Multiple_Issues" folder
                if len(result['issues']) > 1:
                    multiple_issues_path = f"Multiple_Issues/{filename}"
                    zip_file.writestr(multiple_issues_path, result['image_data'])
        
        # Add a summary report to the ZIP
        summary_report = create_summary_report(results)
        zip_file.writestr("analysis_summary.txt", summary_report)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def create_summary_report(results: List[Dict[str, Any]]) -> str:
    """
    Create a text summary report of the analysis results.
    
    Args:
        results: List of analysis results
        
    Returns:
        Summary report as string
    """
    total_images = len(results)
    good_quality = len([r for r in results if r['overall_quality'] == 'Good'])
    poor_quality = len([r for r in results if r['overall_quality'] == 'Poor'])
    errors = len([r for r in results if r['overall_quality'] == 'Error'])
    
    # Count specific issues
    overexposed = len([r for r in results if 'Overexposed' in r['issues']])
    underexposed = len([r for r in results if 'Underexposed' in r['issues']])
    blurry = len([r for r in results if 'Blurry' in r['issues']])
    streaky = len([r for r in results if 'Streaky' in r['issues']])
    
    report = f"""
IMAGE QUALITY ANALYSIS SUMMARY REPORT
=====================================

Analysis Overview:
- Total Images Processed: {total_images}
- Good Quality Images: {good_quality} ({good_quality/total_images*100:.1f}%)
- Poor Quality Images: {poor_quality} ({poor_quality/total_images*100:.1f}%)
- Processing Errors: {errors} ({errors/total_images*100:.1f}%)

Issue Breakdown:
- Overexposed Images: {overexposed}
- Underexposed Images: {underexposed}  
- Blurry Images: {blurry}
- Streaky Images: {streaky}

Folder Organization:
- Good_Quality/: Contains {good_quality} high-quality images
- Poor_Quality/Overexposed/: Contains {overexposed} overexposed images
- Poor_Quality/Underexposed/: Contains {underexposed} underexposed images
- Poor_Quality/Blurry/: Contains {blurry} blurry images
- Poor_Quality/Streaky/: Contains {streaky} streaky images
- Multiple_Issues/: Contains images with multiple quality issues
- Processing_Errors/: Contains images that could not be processed

Detailed Results:
"""
    
    # Add detailed results for each image
    for result in results:
        issues_str = ", ".join(result['issues']) if result['issues'] else "None"
        report += f"\n{result['filename']}:"
        report += f"\n  Quality: {result['overall_quality']}"
        report += f"\n  Issues: {issues_str}"
        
        if result['scores']:
            report += f"\n  Scores:"
            for metric, score in result['scores'].items():
                report += f"\n    {metric.replace('_', ' ').title()}: {score:.3f}"
        report += "\n"
    
    return report

def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename.
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (lowercase)
    """
    return os.path.splitext(filename)[1].lower()

def is_supported_image_format(filename: str) -> bool:
    """
    Check if the file format is supported for image processing.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if format is supported, False otherwise
    """
    supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    return get_file_extension(filename) in supported_extensions

def calculate_processing_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive processing statistics.
    
    Args:
        results: List of analysis results
        
    Returns:
        Dictionary with processing statistics
    """
    if not results:
        return {}
    
    total_images = len(results)
    successful_analyses = len([r for r in results if r['overall_quality'] != 'Error'])
    
    # Calculate average scores for successful analyses
    if successful_analyses > 0:
        valid_results = [r for r in results if r['scores']]
        
        avg_scores = {}
        if valid_results:
            score_keys = valid_results[0]['scores'].keys()
            for key in score_keys:
                scores = [r['scores'][key] for r in valid_results if key in r['scores']]
                if scores:
                    avg_scores[f'avg_{key}'] = sum(scores) / len(scores)
    else:
        avg_scores = {}
    
    # Issue distribution
    all_issues = []
    for result in results:
        all_issues.extend(result['issues'])
    
    issue_counts = {}
    for issue in set(all_issues):
        issue_counts[issue] = all_issues.count(issue)
    
    stats = {
        'total_images': total_images,
        'successful_analyses': successful_analyses,
        'processing_success_rate': successful_analyses / total_images * 100,
        'average_scores': avg_scores,
        'issue_distribution': issue_counts,
        'quality_distribution': {
            'Good': len([r for r in results if r['overall_quality'] == 'Good']),
            'Poor': len([r for r in results if r['overall_quality'] == 'Poor']),
            'Error': len([r for r in results if r['overall_quality'] == 'Error'])
        }
    }
    
    return stats
