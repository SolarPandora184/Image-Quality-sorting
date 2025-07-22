import cv2
import numpy as np
from typing import Dict, List, Tuple, Any

class ImageQualityAnalyzer:
    """
    A comprehensive image quality analyzer that detects various image quality issues
    including exposure problems, blur, and streaks.
    """
    
    def __init__(self):
        # Thresholds for quality assessment
        self.blur_threshold = 100.0  # Laplacian variance threshold
        self.overexposure_threshold = 0.05  # Percentage of overexposed pixels
        self.underexposure_threshold = 0.05  # Percentage of underexposed pixels
        self.streak_threshold = 0.7  # Threshold for streak detection
        
    def analyze_image(self, image: np.ndarray, filename: str) -> Dict[str, Any]:
        """
        Analyze an image for various quality issues.
        
        Args:
            image: OpenCV image (BGR format)
            filename: Name of the image file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Convert to grayscale for some analyses
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Perform individual quality checks
            blur_score = self._detect_blur(gray)
            exposure_scores = self._analyze_exposure(image)
            streak_score = self._detect_streaks(gray)
            
            # Determine issues based on thresholds
            issues = []
            
            if blur_score < self.blur_threshold:
                issues.append("Blurry")
            
            if exposure_scores['overexposure_percentage'] > self.overexposure_threshold:
                issues.append("Overexposed")
            
            if exposure_scores['underexposure_percentage'] > self.underexposure_threshold:
                issues.append("Underexposed")
            
            if streak_score > self.streak_threshold:
                issues.append("Streaky")
            
            # Determine overall quality
            overall_quality = "Good" if not issues else "Poor"
            
            # Compile all scores
            scores = {
                'blur_score': blur_score,
                'overexposure_percentage': exposure_scores['overexposure_percentage'],
                'underexposure_percentage': exposure_scores['underexposure_percentage'],
                'brightness_mean': exposure_scores['brightness_mean'],
                'contrast_std': exposure_scores['contrast_std'],
                'streak_score': streak_score
            }
            
            return {
                'filename': filename,
                'overall_quality': overall_quality,
                'issues': issues,
                'scores': scores
            }
            
        except Exception as e:
            return {
                'filename': filename,
                'overall_quality': 'Error',
                'issues': [f"Analysis error: {str(e)}"],
                'scores': {}
            }
    
    def _detect_blur(self, gray_image: np.ndarray) -> float:
        """
        Detect blur using Laplacian variance method.
        Higher values indicate sharper images.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Blur score (higher = sharper)
        """
        try:
            # Apply Laplacian operator
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            
            # Calculate variance of Laplacian
            blur_score = laplacian.var()
            
            return blur_score
            
        except Exception:
            return 0.0
    
    def _analyze_exposure(self, image: np.ndarray) -> Dict[str, float]:
        """
        Analyze image exposure using histogram analysis.
        
        Args:
            image: Color image (BGR format)
            
        Returns:
            Dictionary with exposure metrics
        """
        try:
            # Convert to grayscale for exposure analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            total_pixels = gray.shape[0] * gray.shape[1]
            
            # Calculate overexposure (bright pixels near 255)
            overexposed_pixels = np.sum(hist[240:256])  # Pixels in range 240-255
            overexposure_percentage = overexposed_pixels / total_pixels
            
            # Calculate underexposure (dark pixels near 0)
            underexposed_pixels = np.sum(hist[0:16])  # Pixels in range 0-15
            underexposure_percentage = underexposed_pixels / total_pixels
            
            # Calculate overall brightness and contrast
            brightness_mean = np.mean(gray)
            contrast_std = np.std(gray)
            
            return {
                'overexposure_percentage': float(overexposure_percentage),
                'underexposure_percentage': float(underexposure_percentage),
                'brightness_mean': float(brightness_mean),
                'contrast_std': float(contrast_std)
            }
            
        except Exception:
            return {
                'overexposure_percentage': 0.0,
                'underexposure_percentage': 0.0,
                'brightness_mean': 0.0,
                'contrast_std': 0.0
            }
    
    def _detect_streaks(self, gray_image: np.ndarray) -> float:
        """
        Detect streaks and linear artifacts in the image using line detection.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Streak score (higher = more streaky)
        """
        try:
            # Apply edge detection
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            
            # Apply Hough Line Transform to detect lines
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return 0.0
            
            # Analyze detected lines
            line_count = len(lines)
            image_area = gray_image.shape[0] * gray_image.shape[1]
            
            # Calculate streak score based on line density
            streak_score = line_count / (image_area / 10000)  # Normalize by image size
            
            # Additional analysis: check for parallel lines (indicating streaks)
            if line_count > 5:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angles.append(theta)
                
                # Check for clustering of angles (parallel lines)
                angles = np.array(angles)
                angle_std = np.std(angles)
                
                # If angles are clustered (low std), it indicates parallel streaks
                if angle_std < 0.5:  # Threshold for parallel detection
                    streak_score *= 2  # Increase score for parallel streaks
            
            return min(streak_score, 2.0)  # Cap the score at 2.0
            
        except Exception:
            return 0.0
    
    def _detect_artifacts(self, gray_image: np.ndarray) -> Dict[str, float]:
        """
        Detect various image artifacts like noise, compression artifacts, etc.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Dictionary with artifact scores
        """
        try:
            # Noise detection using local variance
            kernel = np.ones((3, 3), np.float32) / 9
            smoothed = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            noise_map = np.abs(gray_image.astype(np.float32) - smoothed)
            noise_score = np.mean(noise_map)
            
            # Compression artifact detection using DCT analysis
            # Split image into 8x8 blocks and analyze DCT coefficients
            h, w = gray_image.shape
            compression_score = 0.0
            block_count = 0
            
            for i in range(0, h-8, 8):
                for j in range(0, w-8, 8):
                    block = gray_image[i:i+8, j:j+8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    
                    # Check for quantization artifacts (zeros in high frequency)
                    high_freq_count = np.sum(np.abs(dct_block[4:, 4:]) < 1.0)
                    compression_score += high_freq_count / 16  # 4x4 high freq region
                    block_count += 1
            
            if block_count > 0:
                compression_score /= block_count
            
            return {
                'noise_score': float(noise_score),
                'compression_score': float(compression_score)
            }
            
        except Exception:
            return {
                'noise_score': 0.0,
                'compression_score': 0.0
            }
