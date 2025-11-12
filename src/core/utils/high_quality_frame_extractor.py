"""
High-Quality Frame Extraction Utility

Uses ffmpeg with explicit quality parameters for superior frame extraction.
Provides maximum quality for OCR and computer vision tasks.
"""

import cv2
import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict
import json


class HighQualityFrameExtractor:
    """
    Extract frames from video with maximum quality using ffmpeg.

    Uses explicit quality parameters:
    - q:v 1 (highest quality on 1-31 scale)
    - pix_fmt rgb24 (explicit 24-bit RGB)
    - vcodec png (lossless codec)
    - sws_flags lanczos+accurate_rnd+full_chroma_int (best scaling)
    """

    def __init__(self, video_path: str, use_pyav: bool = True):
        """
        Initialize high-quality frame extractor.

        Args:
            video_path: Path to video file
            use_pyav: Try to use PyAV library if available (default: True)
        """
        self.video_path = video_path
        self.use_pyav = use_pyav

        # Check if video exists
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        # Get video properties using ffprobe
        self._probe_video()

        # Try to initialize PyAV if requested
        self.pyav_container = None
        if use_pyav:
            try:
                import av
                self.pyav_container = av.open(video_path)
                self.pyav_stream = self.pyav_container.streams.video[0]
                # Configure for best quality
                self.pyav_stream.codec_context.thread_type = 'AUTO'
            except ImportError:
                # PyAV not available, will use ffmpeg subprocess
                pass
            except Exception as e:
                # PyAV failed to open, will use ffmpeg subprocess
                print(f"Warning: PyAV initialization failed, falling back to ffmpeg: {e}")
                self.pyav_container = None

    def _probe_video(self):
        """Use ffprobe to get video properties"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-show_format',
            '-select_streams', 'v:0',
            self.video_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            if not data.get('streams'):
                raise ValueError(f"No video streams found in: {self.video_path}")

            stream = data['streams'][0]
            format_info = data.get('format', {})

            # Parse video properties
            self.width = int(stream.get('width', 0))
            self.height = int(stream.get('height', 0))

            # Calculate FPS from r_frame_rate (more accurate than avg_frame_rate)
            fps_str = stream.get('r_frame_rate', '0/1')
            num, den = map(int, fps_str.split('/'))
            self.fps = num / den if den > 0 else 0

            # Get duration from multiple possible sources
            self.duration = 0.0

            # Try 1: Stream-level duration field
            if 'duration' in stream:
                self.duration = float(stream['duration'])

            # Try 2: Stream tags DURATION field (format: HH:MM:SS.mmmmmmmmm)
            if self.duration == 0.0 and 'tags' in stream and 'DURATION' in stream['tags']:
                duration_str = stream['tags']['DURATION']
                # Parse HH:MM:SS.mmmmmmmmm format
                try:
                    parts = duration_str.split(':')
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = float(parts[2])
                    self.duration = hours * 3600 + minutes * 60 + seconds
                except (ValueError, IndexError):
                    pass

            # Try 3: Format-level duration
            if self.duration == 0.0 and 'duration' in format_info:
                self.duration = float(format_info['duration'])

            # Get frame count
            self.frame_count = int(stream.get('nb_frames', 0))

            # If nb_frames not available, calculate from duration
            if self.frame_count == 0 and self.duration > 0:
                self.frame_count = int(self.duration * self.fps)

            # If duration not available, calculate from frame count
            if self.duration == 0 and self.frame_count > 0 and self.fps > 0:
                self.duration = self.frame_count / self.fps

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Failed to probe video with ffprobe: {e}")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse ffprobe output: {e}")

    def get_info(self) -> Dict:
        """Get video information"""
        return {
            'path': self.video_path,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration_seconds': self.duration,
            'width': self.width,
            'height': self.height,
            'duration_formatted': self._format_time(self.duration),
            'extractor': 'PyAV' if self.pyav_container else 'ffmpeg'
        }

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def extract_frame_at_time(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Extract a single frame at specific timestamp with maximum quality.

        Args:
            timestamp: Time in seconds

        Returns:
            Frame as numpy array (H, W, C) in RGB format, or None if failed
        """
        if self.pyav_container:
            return self._extract_frame_pyav(timestamp)
        else:
            return self._extract_frame_ffmpeg(timestamp)

    def _extract_frame_ffmpeg(self, timestamp: float) -> Optional[np.ndarray]:
        """Extract frame using ffmpeg subprocess with maximum quality"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-ss', str(timestamp),  # Seek to timestamp
                '-i', self.video_path,  # Input file
                '-vframes', '1',  # Extract 1 frame
                '-q:v', '1',  # Highest quality (1-31 scale, lower is better)
                '-pix_fmt', 'rgb24',  # Explicit 24-bit RGB pixel format
                '-vcodec', 'png',  # Lossless PNG codec
                '-sws_flags', 'lanczos+accurate_rnd+full_chroma_int',  # Best scaling quality
                '-f', 'image2',  # Image output format
                tmp_path
            ]

            result = subprocess.run(cmd, capture_output=True, check=False)

            if result.returncode != 0 or not os.path.exists(tmp_path):
                return None

            # Load the PNG file (lossless)
            frame = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
            if frame is None:
                return None

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _extract_frame_pyav(self, timestamp: float) -> Optional[np.ndarray]:
        """Extract frame using PyAV library with maximum quality"""
        try:
            import av

            # Seek to timestamp (PyAV uses microseconds internally)
            self.pyav_container.seek(int(timestamp * av.time_base))

            # Decode frames until we get the one at or after our timestamp
            for frame in self.pyav_container.decode(video=0):
                frame_time = float(frame.pts * frame.time_base)

                if frame_time >= timestamp:
                    # Convert frame to numpy array in RGB format
                    # PyAV provides high-quality decoding with no extra compression
                    img = frame.to_ndarray(format='rgb24')
                    return img

            return None

        except Exception as e:
            print(f"Warning: PyAV frame extraction failed, falling back to ffmpeg: {e}")
            return self._extract_frame_ffmpeg(timestamp)

    def extract_frames(self,
                      interval_seconds: float = 2.0,
                      start_time: float = 0,
                      end_time: Optional[float] = None,
                      max_frames: Optional[int] = None) -> Generator[Tuple[float, np.ndarray], None, None]:
        """
        Extract frames at regular intervals with maximum quality.

        Args:
            interval_seconds: Time between frames (default: 2.0)
            start_time: Start time in seconds (default: 0)
            end_time: End time in seconds (default: end of video)
            max_frames: Maximum number of frames to extract (optional)

        Yields:
            Tuple of (timestamp, frame) where frame is numpy array (H, W, C) in RGB
        """
        if end_time is None:
            end_time = self.duration

        current_time = start_time
        extracted_count = 0

        while current_time < end_time:
            frame = self.extract_frame_at_time(current_time)

            if frame is not None:
                yield current_time, frame
                extracted_count += 1

                if max_frames and extracted_count >= max_frames:
                    break

            current_time += interval_seconds

    def save_frames(self,
                   output_dir: str,
                   interval_seconds: float = 2.0,
                   prefix: str = "frame",
                   max_frames: Optional[int] = None,
                   format: str = 'png') -> list:
        """
        Extract and save frames to disk with maximum quality.

        Args:
            output_dir: Directory to save frames
            interval_seconds: Time between frames
            prefix: Filename prefix
            max_frames: Maximum number of frames to save
            format: Image format ('png' for lossless, 'jpg' for smaller files)

        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        for timestamp, frame in self.extract_frames(interval_seconds=interval_seconds,
                                                     max_frames=max_frames):
            filename = f"{prefix}_{timestamp:08.2f}s.{format}"
            filepath = os.path.join(output_dir, filename)

            if format.lower() == 'png':
                # Save as lossless PNG
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            else:
                # Save as high-quality JPEG
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])

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
        # Ensure coordinates are within frame bounds
        x = max(0, min(x, frame.shape[1]))
        y = max(0, min(y, frame.shape[0]))
        width = min(width, frame.shape[1] - x)
        height = min(height, frame.shape[0] - y)

        return frame[y:y+height, x:x+width]

    def close(self):
        """Release video resources"""
        if self.pyav_container:
            try:
                self.pyav_container.close()
            except:
                pass
            self.pyav_container = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def extract_frames_high_quality(video_path: str,
                                interval_seconds: float = 2.0,
                                max_frames: Optional[int] = None,
                                verbose: bool = True,
                                use_pyav: bool = True) -> list:
    """
    Convenience function to extract frames with maximum quality.

    Args:
        video_path: Path to video file
        interval_seconds: Time between frames
        max_frames: Maximum frames to extract
        verbose: Print progress
        use_pyav: Try to use PyAV if available

    Returns:
        List of (timestamp, frame) tuples
    """
    frames = []

    with HighQualityFrameExtractor(video_path, use_pyav=use_pyav) as extractor:
        if verbose:
            info = extractor.get_info()
            print(f"Video: {info['duration_formatted']} ({info['frame_count']} frames @ {info['fps']:.1f} fps)")
            print(f"Extractor: {info['extractor']}")
            print(f"Extracting high-quality frames every {interval_seconds}s...")

        for timestamp, frame in extractor.extract_frames(interval_seconds=interval_seconds,
                                                         max_frames=max_frames):
            frames.append((timestamp, frame))
            if verbose and len(frames) % 10 == 0:
                print(f"  Extracted {len(frames)} frames ({timestamp:.1f}s)...")

    if verbose:
        print(f"✓ Extracted {len(frames)} total frames (high quality)")

    return frames
