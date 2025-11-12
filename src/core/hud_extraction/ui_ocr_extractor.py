"""
UI OCR Extractor for Super People
Extracts metrics from game UI elements using OCR.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class SuperPeopleUIExtractor:
    """Extract UI metrics from Super People gameplay"""

    # UI region definitions based on actual Super People gameplay screenshots
    # Positions calibrated for 1920x1080 resolution
    # Coordinates are (x, y, width, height) from top-left corner
    # NOTE: These coordinates need to be recalibrated for your specific game HUD layout
    UI_REGIONS = {
        'compass_heading': {
            'x': 925, 'y': 15, 'width': 100, 'height': 30,
            'type': 'text',
            'description': 'Compass direction at top center (N, NE, 330, etc.)',
        },

        'game_status': {
            'x': 1650, 'y': 15, 'width': 250, 'height': 60,
            'type': 'text',
            'description': 'Game status at top right (Kills, Assists, Teams, Soldiers)',
        },

        'player_rank': {
            'x': 1650, 'y': 400, 'width': 250, 'height': 40,
            'type': 'text',
            'description': 'Player rank and class at right side (e.g., "Lv 32 Firearms Expert")',
        },

        'equipped_accessories': {
            'x': 1650, 'y': 850, 'width': 250, 'height': 200,
            'type': 'text',
            'description': 'Equipment accessories in 3x3 box at lower right',
        },

        'weapon_and_ammo': {
            'x': 800, 'y': 950, 'width': 300, 'height': 100,
            'type': 'text',
            'description': 'Current weapon and ammo at lower center right',
        },

        'player_health': {
            'x': 860, 'y': 1000, 'width': 200, 'height': 50,
            'type': 'number',
            'description': 'Player health at lower center (current/max format)',
        },

        'team_health': {
            'x': 20, 'y': 300, 'width': 250, 'height': 300,
            'type': 'text',
            'description': 'Team member names and health at lower left',
        },
    }

    def __init__(self, resolution_scale: float = 1.0, custom_regions: Optional[Dict] = None,
                 regions_file: Optional[str] = None, enhanced_preprocessing: bool = False,
                 debug_dir: Optional[str] = None):
        """
        Initialize UI extractor.

        Args:
            resolution_scale: Scale factor for UI coordinates (e.g., 0.5 for 720p if regions defined for 1440p)
            custom_regions: Custom UI region definitions (overrides defaults)
            regions_file: Path to auto-calibration JSON file (highest priority)
            enhanced_preprocessing: Use enhanced preprocessing for low-contrast HUDs (default: False)
            debug_dir: If provided, save preprocessing debug images to this directory
        """
        self.resolution_scale = resolution_scale
        self.enhanced_preprocessing = enhanced_preprocessing
        self.debug_dir = debug_dir

        # Create debug directory if specified
        if debug_dir:
            from pathlib import Path
            Path(debug_dir).mkdir(parents=True, exist_ok=True)

        if regions_file:
            # Load from auto-calibration file (highest priority)
            self.ui_regions = self.load_regions_from_file(regions_file)
        elif custom_regions:
            self.ui_regions = custom_regions
        else:
            # Scale default regions if needed
            self.ui_regions = self._scale_regions(self.UI_REGIONS, resolution_scale)

    def _scale_regions(self, regions: Dict, scale: float) -> Dict:
        """Scale UI region coordinates"""
        if scale == 1.0:
            return regions

        scaled = {}
        for name, region in regions.items():
            scaled[name] = {
                'x': int(region['x'] * scale),
                'y': int(region['y'] * scale),
                'width': int(region['width'] * scale),
                'height': int(region['height'] * scale),
                'type': region['type'],
                'description': region['description']
            }
        return scaled

    @staticmethod
    def load_regions_from_file(regions_file: str) -> Dict:
        """
        Load UI regions from auto-calibration JSON file.

        Args:
            regions_file: Path to JSON file with region definitions

        Returns:
            Dictionary of UI regions with padding added for better OCR
        """
        with open(regions_file, 'r') as f:
            regions = json.load(f)

        # Add padding to bounding boxes for better OCR accuracy
        return SuperPeopleUIExtractor._add_padding_to_regions(regions)

    @staticmethod
    def _add_padding_to_regions(regions: Dict, padding: int = 10) -> Dict:
        """
        Add padding to bounding boxes for better OCR accuracy.

        Vision API returns tight boxes, OCR needs breathing room.

        Args:
            regions: Dictionary of UI region definitions
            padding: Pixels to add around each region (default: 10)

        Returns:
            Dictionary with padded regions
        """
        padded = {}
        for name, region in regions.items():
            padded[name] = {
                'x': max(0, region['x'] - padding),
                'y': max(0, region['y'] - padding),
                'width': region['width'] + 2 * padding,
                'height': region['height'] + 2 * padding,
                'type': region['type'],
                'description': region.get('description', '')
            }
            # Preserve pattern if it exists
            if 'pattern' in region:
                padded[name]['pattern'] = region['pattern']

        return padded

    def extract_roi(self, frame: np.ndarray, region_name: str) -> np.ndarray:
        """
        Extract region of interest from frame.

        Args:
            frame: Input frame (RGB format)
            region_name: Name of UI region to extract

        Returns:
            Cropped region as numpy array
        """
        region = self.ui_regions[region_name]
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        return frame[y:y+h, x:x+w]

    def preprocess_for_ocr(self, roi: np.ndarray, region_type: str) -> np.ndarray:
        """
        Preprocess ROI for better OCR accuracy.

        Args:
            roi: Region of interest
            region_type: Type of region ('number', 'text', 'currency')

        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Apply thresholding (inverted for light text on dark backgrounds)
        if region_type == 'number':
            # For numbers, use adaptive threshold
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
        else:
            # For text, use Otsu's thresholding
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Optional: denoise
        processed = cv2.fastNlMeansDenoising(processed)

        # Optional: resize for better OCR (upscale small text)
        if roi.shape[0] < 40 or roi.shape[1] < 40:
            processed = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        return processed

    def preprocess_for_ocr_enhanced(self, roi: np.ndarray, region_type: str) -> np.ndarray:
        """
        Enhanced preprocessing for low-contrast gaming HUDs.

        Designed for challenging conditions:
        - Light text on bright backgrounds (daylight gameplay)
        - Semi-transparent HUD overlays
        - Low contrast scenarios

        Args:
            roi: Region of interest (RGB)
            region_type: Type of region ('number', 'text')

        Returns:
            Preprocessed binary image optimized for OCR
        """
        # Step 1: Find best color channel
        # Gaming HUDs with white/yellow text often have better contrast in specific channels
        b, g, r = cv2.split(roi)

        # Calculate variance for each channel (higher = more contrast)
        variances = {
            'b': np.var(b),
            'g': np.var(g),
            'r': np.var(r),
            'gray': np.var(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY))
        }

        # Use channel with highest variance (best contrast)
        best_channel = max(variances, key=variances.get)
        if best_channel == 'b':
            gray = b
        elif best_channel == 'g':
            gray = g
        elif best_channel == 'r':
            gray = r
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Step 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This dramatically improves local contrast in washed-out images
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Step 3: Aggressive upscaling BEFORE thresholding
        # Upscale 3x for small text to improve edge detection
        scale_factor = 3
        upscaled = cv2.resize(enhanced, None, fx=scale_factor, fy=scale_factor,
                             interpolation=cv2.INTER_LANCZOS4)

        # Step 4: Multi-threshold approach
        # Try multiple threshold values and pick the best result
        thresholds = []

        # Method 1: Otsu's (automatic threshold selection)
        _, otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholds.append(otsu)

        # Method 2: Inverted Otsu's (for light text on dark bg)
        _, otsu_inv = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresholds.append(otsu_inv)

        # Method 3: Adaptive threshold (handles varying backgrounds)
        adaptive = cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, 2)
        thresholds.append(adaptive)

        # Method 4: Inverted adaptive
        adaptive_inv = cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 15, 2)
        thresholds.append(adaptive_inv)

        # Pick threshold with most white pixels (likely to contain text)
        # For gaming HUDs, text is usually the brightest element
        white_counts = [np.sum(t == 255) for t in thresholds]
        best_threshold = thresholds[np.argmax(white_counts)]

        # Step 5: Apply morphological operations
        kernel = np.ones((2, 2), np.uint8)

        # Close small gaps in text
        processed = cv2.morphologyEx(best_threshold, cv2.MORPH_CLOSE, kernel)

        # Remove small noise
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

        # Step 6: Final denoising
        processed = cv2.fastNlMeansDenoising(processed)

        # Step 7: Try BOTH polarities - Tesseract works better with black-on-white
        # Store both versions so ocr_region can test both
        self._last_processed = processed
        self._last_processed_inv = cv2.bitwise_not(processed)

        return processed

    def ocr_region(self, frame: np.ndarray, region_name: str, preprocess: bool = True,
                   timestamp: Optional[float] = None) -> str:
        """
        Perform OCR on a specific UI region.

        Args:
            frame: Input frame
            region_name: Name of UI region
            preprocess: Whether to preprocess image
            timestamp: Optional timestamp for debug image naming

        Returns:
            Extracted text
        """
        roi = self.extract_roi(frame, region_name)
        region_type = self.ui_regions[region_name]['type']

        # Save original ROI for debug if enabled
        if self.debug_dir:
            debug_name = f"{timestamp:.1f}s_{region_name}_original.png" if timestamp else f"{region_name}_original.png"
            cv2.imwrite(f"{self.debug_dir}/{debug_name}", cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

        if preprocess:
            # Use enhanced preprocessing if enabled, otherwise use standard
            if self.enhanced_preprocessing:
                roi = self.preprocess_for_ocr_enhanced(roi, region_type)
            else:
                roi = self.preprocess_for_ocr(roi, region_type)

            # Save preprocessed ROI for debug if enabled
            if self.debug_dir:
                debug_name = f"{timestamp:.1f}s_{region_name}_processed.png" if timestamp else f"{region_name}_processed.png"
                cv2.imwrite(f"{self.debug_dir}/{debug_name}", roi)

        # Configure tesseract based on region type
        # PSM 7 = single text line, PSM 8 = single word, PSM 13 = raw line
        if region_type == 'number':
            # Digits, slash, and space (for formats like "110 / 110")
            # Try PSM 7 (single line) - more permissive than before
            config = '--psm 7 --oem 3'
        elif region_type == 'currency':
            # Digits and E for money format
            config = '--psm 7 --oem 3'
        else:
            # Default for text - use PSM 7 for single lines
            config = '--psm 7 --oem 3'

        # If enhanced preprocessing, try BOTH polarities and pick better result
        if self.enhanced_preprocessing and hasattr(self, '_last_processed_inv'):
            # Try original polarity
            pil_img_orig = Image.fromarray(roi)
            text_orig = pytesseract.image_to_string(pil_img_orig, config=config).strip()

            # Try inverted polarity
            pil_img_inv = Image.fromarray(self._last_processed_inv)
            text_inv = pytesseract.image_to_string(pil_img_inv, config=config).strip()

            # Pick the result with more alphanumeric characters (likely better OCR)
            alphanum_orig = sum(c.isalnum() for c in text_orig)
            alphanum_inv = sum(c.isalnum() for c in text_inv)

            if alphanum_inv > alphanum_orig:
                return text_inv
            else:
                return text_orig
        else:
            # Standard preprocessing - just run OCR once
            pil_img = Image.fromarray(roi)
            text = pytesseract.image_to_string(pil_img, config=config)
            return text.strip()

    def parse_value(self, text: str, region_type: str, region_name: str = '') -> Optional[any]:
        """
        Parse OCR text into structured value.

        Args:
            text: Raw OCR text
            region_type: Type of region
            region_name: Name of region (for special parsing rules)

        Returns:
            Parsed value (int, float, or dict)
        """
        if not text:
            return None

        try:
            # Get pattern if defined for this region
            pattern = self.ui_regions.get(region_name, {}).get('pattern')

            if region_type == 'number':
                # Extract first number found
                if pattern:
                    match = re.search(pattern, text)
                    return int(match.group(1) if match.groups() else match.group()) if match else None
                else:
                    match = re.search(r'\d+', text)
                    return int(match.group()) if match else None

            elif region_type == 'text':
                # Use pattern if provided
                if pattern:
                    match = re.search(pattern, text)
                    if match:
                        return match.group(1) if match.groups() else match.group()
                    return None
                else:
                    # Return cleaned text
                    return text.strip()

        except Exception:
            return None

    def extract_all_metrics(self, frame: np.ndarray) -> Dict:
        """
        Extract all UI metrics from a frame.

        Args:
            frame: Input frame (RGB format)

        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}

        for region_name, region_info in self.ui_regions.items():
            try:
                text = self.ocr_region(frame, region_name)
                value = self.parse_value(text, region_info['type'], region_name)
                metrics[region_name] = {
                    'raw_text': text,
                    'value': value,
                    'description': region_info['description']
                }
            except Exception as e:
                metrics[region_name] = {
                    'raw_text': '',
                    'value': None,
                    'error': str(e),
                    'description': region_info['description']
                }

        return metrics

    def calibrate_regions(self, frame: np.ndarray, output_dir: str = "calibration_output"):
        """
        Helper function to visualize and calibrate UI regions.

        Args:
            frame: Sample frame from video
            output_dir: Directory to save calibration images
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Draw all regions on frame
        frame_vis = frame.copy()
        for name, region in self.ui_regions.items():
            x, y, w, h = region['x'], region['y'], region['width'], region['height']

            # Draw rectangle
            cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Add label
            cv2.putText(frame_vis, name, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Save individual ROI
            roi = self.extract_roi(frame, name)
            roi_path = os.path.join(output_dir, f"{name}_roi.png")
            cv2.imwrite(roi_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

            # Save preprocessed ROI
            preprocessed = self.preprocess_for_ocr(roi, region['type'])
            prep_path = os.path.join(output_dir, f"{name}_preprocessed.png")
            cv2.imwrite(prep_path, preprocessed)

        # Save annotated frame
        frame_path = os.path.join(output_dir, "annotated_frame.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR))

        print(f"✓ Calibration images saved to: {output_dir}")
        print(f"  - annotated_frame.png: Full frame with all regions marked")
        print(f"  - *_roi.png: Original regions")
        print(f"  - *_preprocessed.png: Preprocessed regions ready for OCR")


def extract_ui_metrics_from_video(video_path: str,
                                  interval_seconds: float = 2.0,
                                  start_time: float = 0.0,
                                  max_frames: Optional[int] = None,
                                  output_file: Optional[str] = None,
                                  regions_file: Optional[str] = None,
                                  enhanced_preprocessing: bool = False,
                                  debug_dir: Optional[str] = None,
                                  high_quality: bool = False,
                                  verbose: bool = True) -> List[Dict]:
    """
    Extract UI metrics across entire video.

    Args:
        video_path: Path to video file
        interval_seconds: Time between samples
        max_frames: Maximum frames to process
        output_file: Path to save results CSV/JSON
        regions_file: Path to auto-calibration JSON file
        enhanced_preprocessing: Use enhanced preprocessing for low-contrast HUDs
        debug_dir: If provided, save preprocessing debug images
        high_quality: Use high-quality frame extraction (ffmpeg with max quality, slower)
        verbose: Print progress

    Returns:
        List of metrics dictionaries with timestamps
    """
    from .frame_extractor import FrameExtractor

    if verbose:
        print(f"\nExtracting UI metrics from: {video_path}")
        print(f"Sample interval: {interval_seconds}s")
        if regions_file:
            print(f"Using calibrated regions from: {regions_file}")
        if enhanced_preprocessing:
            print("Enhanced preprocessing: ENABLED (low-contrast mode)")
        if high_quality:
            print("High-quality extraction: ENABLED (ffmpeg with maximum quality)")
        if debug_dir:
            print(f"Debug images: {debug_dir}")
        print("=" * 60)

    # Initialize extractor with calibrated regions
    extractor_obj = SuperPeopleUIExtractor(
        regions_file=regions_file,
        enhanced_preprocessing=enhanced_preprocessing,
        debug_dir=debug_dir
    )
    results = []

    with FrameExtractor(video_path, high_quality=high_quality) as frame_ext:
        if verbose:
            info = frame_ext.get_info()
            print(f"Video duration: {info['duration_formatted']}")
            print(f"Processing frames...\n")

        for timestamp, frame in frame_ext.extract_frames(interval_seconds=interval_seconds,
                                                         start_time=start_time,
                                                         max_frames=max_frames):
            metrics = extractor_obj.extract_all_metrics(frame)

            # Add timestamp
            result = {'timestamp': timestamp}

            # Flatten metrics
            for region_name, data in metrics.items():
                result[region_name] = data['value']
                result[f"{region_name}_raw"] = data['raw_text']

            results.append(result)

            if verbose:
                # Print key metrics
                players = result.get('players_remaining')
                health = result.get('health')
                money = result.get('money')
                print(f"[{timestamp:7.1f}s] Players: {players}, Health: {health}, Money: {money}")

    # Save results
    if output_file:
        import pandas as pd

        df = pd.DataFrame(results)

        if output_file.endswith('.csv'):
            df.to_csv(output_file, index=False)
        elif output_file.endswith('.json'):
            df.to_json(output_file, orient='records', indent=2)
        else:
            # Default to CSV
            df.to_csv(output_file + '.csv', index=False)

        if verbose:
            print(f"\n✓ Results saved to: {output_file}")

    if verbose:
        print(f"\n✓ Extracted metrics from {len(results)} frames")

    return results
