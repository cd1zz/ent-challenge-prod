"""
Frame Extraction Utility
Extracts frames from video at specified intervals for analysis.
"""

import cv2
import os
from pathlib import Path
import numpy as np
from typing import Generator, Tuple, Optional


class FrameExtractor:
    """Extract frames from video files for analysis"""

    def __init__(self, video_path: str, high_quality: bool = False, use_pyav: bool = True):
        """
        Initialize frame extractor.

        Args:
            video_path: Path to video file
            high_quality: Use high-quality ffmpeg-based extraction (default: False)
            use_pyav: If high_quality=True, try PyAV first (default: True)
        """
        self.video_path = video_path
        self.high_quality = high_quality

        # If high quality requested, delegate to HighQualityFrameExtractor
        if high_quality:
            from .high_quality_frame_extractor import HighQualityFrameExtractor
            self._hq_extractor = HighQualityFrameExtractor(video_path, use_pyav=use_pyav)

            # Copy properties from high-quality extractor
            self.fps = self._hq_extractor.fps
            self.frame_count = self._hq_extractor.frame_count
            self.duration = self._hq_extractor.duration
            self.width = self._hq_extractor.width
            self.height = self._hq_extractor.height
            self.video = None  # Not using cv2.VideoCapture in high-quality mode
            return

        # Standard quality mode with improved settings
        self._hq_extractor = None

        # Use ffmpeg backend explicitly and configure for better quality
        self.video = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

        if not self.video.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Set buffer size to 1 to reduce buffering artifacts
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_info(self) -> dict:
        """Get video information"""
        if self._hq_extractor:
            return self._hq_extractor.get_info()

        return {
            'path': self.video_path,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration_seconds': self.duration,
            'width': self.width,
            'height': self.height,
            'duration_formatted': self._format_time(self.duration),
            'extractor': 'OpenCV (improved)'
        }

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def extract_frames(self,
                      interval_seconds: float = 2.0,
                      start_time: float = 0,
                      end_time: Optional[float] = None,
                      max_frames: Optional[int] = None) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Extract frames at regular intervals.

        Args:
            interval_seconds: Time between frames (default: 2.0)
            start_time: Start time in seconds (default: 0)
            end_time: End time in seconds (default: end of video)
            max_frames: Maximum number of frames to extract (optional)

        Yields:
            Tuple of (timestamp, frame) where frame is numpy array (H, W, C)
        """
        # Delegate to high-quality extractor if enabled
        if self._hq_extractor:
            yield from self._hq_extractor.extract_frames(
                interval_seconds=interval_seconds,
                start_time=start_time,
                end_time=end_time,
                max_frames=max_frames
            )
            return

        if end_time is None:
            end_time = self.duration

        frame_interval = int(self.fps * interval_seconds)
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)

        self.video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_num = start_frame
        extracted_count = 0

        while frame_num < end_frame:
            success, frame = self.video.read()
            if not success:
                break

            if (frame_num - start_frame) % frame_interval == 0:
                timestamp = frame_num / self.fps
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield timestamp, frame_rgb

                extracted_count += 1
                if max_frames and extracted_count >= max_frames:
                    break

            frame_num += 1

    def extract_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract a single frame at specific timestamp.

        Args:
            timestamp: Time in seconds

        Returns:
            Frame as numpy array or None if failed
        """
        # Delegate to high-quality extractor if enabled
        if self._hq_extractor:
            return self._hq_extractor.extract_frame_at_time(timestamp)

        frame_num = int(timestamp * self.fps)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        success, frame = self.video.read()
        if success:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def save_frames(self,
                   output_dir: str,
                   interval_seconds: float = 2.0,
                   prefix: str = "frame",
                   max_frames: Optional[int] = None,
                   format: str = 'jpg') -> list:
        """
        Extract and save frames to disk.

        Args:
            output_dir: Directory to save frames
            interval_seconds: Time between frames
            prefix: Filename prefix
            max_frames: Maximum number of frames to save
            format: Image format ('jpg' or 'png')

        Returns:
            List of saved file paths
        """
        # Delegate to high-quality extractor if enabled
        if self._hq_extractor:
            return self._hq_extractor.save_frames(
                output_dir=output_dir,
                interval_seconds=interval_seconds,
                prefix=prefix,
                max_frames=max_frames,
                format=format
            )

        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        for timestamp, frame in self.extract_frames(interval_seconds=interval_seconds,
                                                     max_frames=max_frames):
            filename = f"{prefix}_{timestamp:08.2f}s.{format}"
            filepath = os.path.join(output_dir, filename)

            # Convert RGB back to BGR for saving with cv2
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, frame_bgr)
            saved_files.append(filepath)

        return saved_files

    def extract_roi(self, frame: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Extract region of interest from frame.

        Args:
            frame: Input frame
            x, y: Top-left corner coordinates
            width, height: Region dimensions

        Returns:
            Cropped region as numpy array
        """
        if self._hq_extractor:
            return self._hq_extractor.extract_roi(frame, x, y, width, height)

        return frame[y:y+height, x:x+width]

    def close(self):
        """Release video resources"""
        if self._hq_extractor:
            self._hq_extractor.close()
        elif self.video:
            self.video.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def extract_frames_simple(video_path: str,
                         interval_seconds: float = 2.0,
                         max_frames: Optional[int] = None,
                         verbose: bool = True) -> list:
    """
    Simple convenience function to extract frames.

    Args:
        video_path: Path to video file
        interval_seconds: Time between frames
        max_frames: Maximum frames to extract
        verbose: Print progress

    Returns:
        List of (timestamp, frame) tuples
    """
    frames = []

    with FrameExtractor(video_path) as extractor:
        if verbose:
            info = extractor.get_info()
            print(f"Video: {info['duration_formatted']} ({info['frame_count']} frames @ {info['fps']:.1f} fps)")
            print(f"Extracting frames every {interval_seconds}s...")

        for timestamp, frame in extractor.extract_frames(interval_seconds=interval_seconds,
                                                         max_frames=max_frames):
            frames.append((timestamp, frame))
            if verbose and len(frames) % 10 == 0:
                print(f"  Extracted {len(frames)} frames ({timestamp:.1f}s)...")

    if verbose:
        print(f"✓ Extracted {len(frames)} total frames")

    return frames
