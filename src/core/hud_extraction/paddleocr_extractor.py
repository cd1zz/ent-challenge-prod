"""PaddleOCR HUD Extraction Module.

This module provides high-accuracy OCR-based HUD extraction using PaddleOCR,
which offers better performance than Tesseract for gaming interfaces.

PaddleOCR advantages:
- Superior accuracy on low-contrast gaming HUDs
- Better handling of stylized game fonts
- Support for 80+ languages
- GPU acceleration support
- Faster processing than Tesseract

Typical usage example:

    from paddleocr_extractor import PaddleOCRExtractor, extract_hud_data

    # Load regions definition
    regions = load_regions_from_file('ui_regions.json')

    # Initialize extractor
    extractor = PaddleOCRExtractor(regions=regions, use_gpu=True)

    # Extract from video
    results = extract_hud_data(
        video_path='gameplay.mkv',
        extractor=extractor,
        interval_seconds=2.0
    )
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
import pandas as pd

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Error: PaddleOCR not installed", file=sys.stderr)
    print("\nInstall with:", file=sys.stderr)
    print("  pip install paddlepaddle paddleocr", file=sys.stderr)
    print("\nFor GPU support:", file=sys.stderr)
    print("  pip install paddlepaddle-gpu paddleocr", file=sys.stderr)
    raise


class PaddleOCRExtractor:
    """Extract UI metrics from gameplay videos using PaddleOCR.

    PaddleOCR provides superior accuracy compared to Tesseract for:
    - Low-contrast gaming HUDs
    - Stylized game fonts
    - Complex text layouts
    - Multiple languages

    Attributes:
        regions: Dictionary of UI region definitions with bounding boxes.
        ocr: Initialized PaddleOCR instance.
    """

    def __init__(self,
                 regions: Dict,
                 lang: str = 'en',
                 use_gpu: bool = False,
                 use_angle_cls: bool = True,
                 show_log: bool = False,
                 det_db_thresh: float = 0.3,
                 det_db_box_thresh: float = 0.5,
                 rec_batch_num: int = 6):
        """Initialize PaddleOCR extractor.

        Args:
            regions: UI region definitions from JSON file. Each region must have:
                - 'x': Left edge pixel coordinate
                - 'y': Top edge pixel coordinate
                - 'width': Region width in pixels
                - 'height': Region height in pixels
            lang: Language code (en, ch, fr, de, es, pt, ru, ar, hi, etc.).
                See https://github.com/PaddlePaddle/PaddleOCR#supported-languages
            use_gpu: Enable GPU acceleration (requires paddlepaddle-gpu).
                GPU usage is automatic if paddlepaddle-gpu is installed.
            use_angle_cls: Enable text direction classification for rotated text.
            show_log: Show PaddleOCR internal logs for debugging.
            det_db_thresh: Text detection threshold (0.0-1.0).
                Lower values = more sensitive detection.
            det_db_box_thresh: Bounding box threshold (0.0-1.0).
            rec_batch_num: Batch size for recognition.
                Higher values = faster but more memory usage.

        Example:
            >>> regions = {'health': {'x': 50, 'y': 50, 'width': 100, 'height': 30}}
            >>> extractor = PaddleOCRExtractor(regions, use_gpu=True)
        """
        self.regions = regions

        # Initialize PaddleOCR (compatible with 2.7.3+)
        print(f"Initializing PaddleOCR (lang={lang})...")
        self.ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=use_angle_cls,
            show_log=show_log,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            rec_batch_num=rec_batch_num
        )
        print("✓ PaddleOCR initialized")

    def extract_roi(self, frame: np.ndarray, region_name: str) -> np.ndarray:
        """Extract region of interest from frame.

        Args:
            frame: Input frame in BGR format (OpenCV default).
            region_name: Name of UI region to extract.

        Returns:
            Cropped region as numpy array in BGR format.

        Raises:
            KeyError: If region_name not found in regions dictionary.
        """
        region = self.regions[region_name]
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        return frame[y:y+h, x:x+w]

    def ocr_region(self, frame: np.ndarray, region_name: str) -> Tuple[str, float]:
        """Perform OCR on a specific UI region using PaddleOCR.

        Args:
            frame: Input frame in BGR format.
            region_name: Name of UI region to OCR.

        Returns:
            Tuple of (extracted_text, confidence_score).
            - extracted_text: Combined text from all detected lines, space-separated.
            - confidence_score: Average confidence across all detections (0.0-1.0).
            Returns ("", 0.0) if no text detected.

        Example:
            >>> text, conf = extractor.ocr_region(frame, 'health')
            >>> print(f"Health: {text} (confidence: {conf:.2f})")
        """
        roi = self.extract_roi(frame, region_name)

        # PaddleOCR expects BGR format (OpenCV default)
        result = self.ocr.ocr(roi, cls=True)

        if not result or not result[0]:
            return "", 0.0

        # Combine all detected text in the ROI
        texts = []
        confidences = []

        for line in result[0]:
            if line:
                # line format: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)]
                text_info = line[1]
                text = text_info[0]
                confidence = text_info[1]

                texts.append(text)
                confidences.append(confidence)

        # Join all text with spaces
        combined_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return combined_text, avg_confidence

    def extract_all_metrics(self, frame: np.ndarray,
                           timestamp: float,
                           verbose: bool = False) -> Dict:
        """Extract all UI metrics from a frame.

        Args:
            frame: Input frame in BGR format.
            timestamp: Frame timestamp in seconds.
            verbose: Print extraction details to stdout.

        Returns:
            Dictionary containing:
            - 'timestamp': Frame timestamp (float)
            - For each region:
                - '{region_name}': Extracted text (str)
                - '{region_name}_confidence': Confidence score (float)

        Example:
            >>> metrics = extractor.extract_all_metrics(frame, 120.5, verbose=True)
            >>> print(f"Health at {metrics['timestamp']}s: {metrics['health']}")
        """
        metrics = {'timestamp': timestamp}

        for region_name, region_info in self.regions.items():
            try:
                text, confidence = self.ocr_region(frame, region_name)

                metrics[region_name] = text
                metrics[f"{region_name}_confidence"] = confidence

                if verbose:
                    print(f"  [{region_name}] '{text}' (conf: {confidence:.2f})")

            except Exception as e:
                if verbose:
                    print(f"  [{region_name}] Error: {e}")
                metrics[region_name] = ""
                metrics[f"{region_name}_confidence"] = 0.0

        return metrics


def load_regions_from_file(regions_file: str) -> Dict:
    """Load UI regions from JSON file.

    Args:
        regions_file: Path to JSON file containing region definitions.

    Returns:
        Dictionary mapping region names to region definitions.
        Each region contains: x, y, width, height.

    Raises:
        FileNotFoundError: If regions_file does not exist.
        json.JSONDecodeError: If file is not valid JSON.

    Example:
        >>> regions = load_regions_from_file('ui_regions.json')
        >>> print(f"Loaded {len(regions)} regions")
    """
    with open(regions_file, 'r') as f:
        return json.load(f)


def extract_frames(video_path: str,
                  interval_seconds: float = 2.0,
                  max_frames: Optional[int] = None) -> Generator[Tuple[float, np.ndarray], None, None]:
    """Generator that yields frames from video at regular intervals.

    Args:
        video_path: Path to video file.
        interval_seconds: Time between extracted frames in seconds.
        max_frames: Maximum number of frames to extract. None = all frames.

    Yields:
        Tuple of (timestamp, frame):
        - timestamp: Frame timestamp in seconds (float)
        - frame: Frame as numpy array in BGR format

    Raises:
        ValueError: If video cannot be opened.

    Example:
        >>> for timestamp, frame in extract_frames('video.mp4', interval_seconds=5.0):
        ...     print(f"Processing frame at {timestamp}s")
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frame_interval = int(fps * interval_seconds)
    frame_count = 0
    frames_extracted = 0

    print(f"Video: {video_path}")
    print(f"  Duration: {duration:.1f}s ({total_frames} frames @ {fps:.1f} fps)")
    print(f"  Sampling every {interval_seconds}s ({frame_interval} frames)")
    print()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Extract frames at intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            yield timestamp, frame

            frames_extracted += 1

            if max_frames and frames_extracted >= max_frames:
                break

        frame_count += 1

    cap.release()
    print(f"\n✓ Extracted {frames_extracted} frames")


def extract_hud_data(video_path: str,
                    extractor: PaddleOCRExtractor,
                    interval_seconds: float = 2.0,
                    max_frames: Optional[int] = None,
                    output_file: Optional[str] = None,
                    verbose: bool = False) -> pd.DataFrame:
    """Extract HUD data from video using PaddleOCR.

    High-level convenience function that combines frame extraction
    and OCR processing.

    Args:
        video_path: Path to video file.
        extractor: Initialized PaddleOCRExtractor instance.
        interval_seconds: Frame sampling interval in seconds.
        max_frames: Maximum frames to process. None = all frames.
        output_file: Optional path to save results (.csv or .json).
        verbose: Print extraction progress.

    Returns:
        DataFrame containing extracted metrics with columns:
        - timestamp: Frame timestamp (float)
        - {region_name}: Extracted text for each region (str)
        - {region_name}_confidence: Confidence for each region (float)

    Example:
        >>> regions = load_regions_from_file('ui_regions.json')
        >>> extractor = PaddleOCRExtractor(regions, use_gpu=True)
        >>> df = extract_hud_data('gameplay.mkv', extractor, verbose=True)
        >>> print(f"Extracted {len(df)} frames")
    """
    results = []
    frame_count = 0

    print("Extracting HUD metrics...")
    print("-" * 70)

    for timestamp, frame in extract_frames(video_path, interval_seconds, max_frames):
        frame_count += 1

        if verbose:
            print(f"\n[{timestamp:7.1f}s] Frame {frame_count}")
        else:
            # Progress indicator
            if frame_count % 10 == 0:
                print(f"  Processed {frame_count} frames...", end='\r')

        metrics = extractor.extract_all_metrics(frame, timestamp, verbose)
        results.append(metrics)

    print()
    print("-" * 70)
    print(f"✓ Extracted metrics from {len(results)} frames")
    print()

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save if requested
    if output_file:
        if output_file.endswith('.json'):
            df.to_json(output_file, orient='records', indent=2)
        else:
            df.to_csv(output_file, index=False)
        print(f"✓ Results saved to: {output_file}")

    return df
