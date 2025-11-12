"""Game Over Screen Detection and Stats Extraction.

This module provides automated detection of game over/end screens in gameplay
videos using PaddleOCR. It identifies victory/defeat outcomes and extracts
final game statistics like placement, kills, damage, and survival time.

Key features:
- Automatic detection of game over screens via keyword analysis
- Victory/defeat/elimination classification
- Stats extraction: placement, kills, damage, survival time, assists, revives
- Confidence scoring for detections
- Support for multiple game types and UI layouts

Typical usage example:

    from game_over_detector import GameOverDetector, scan_video_for_game_over

    # Initialize detector
    detector = GameOverDetector(use_gpu=True)

    # Scan video
    events = scan_video_for_game_over(
        video_path='gameplay.mkv',
        detector=detector,
        interval_seconds=2.0
    )
"""

import sys
import json
import cv2
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from datetime import datetime

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("Error: PaddleOCR not installed", file=sys.stderr)
    print("\nInstall with:", file=sys.stderr)
    print("  pip install paddlepaddle paddleocr", file=sys.stderr)
    raise


class GameOverDetector:
    """Detect game over screens and extract stats using PaddleOCR.

    Uses keyword-based detection to identify game over screens and regex
    pattern matching to extract game statistics from the detected text.

    Attributes:
        ocr: Initialized PaddleOCR instance.
        confidence_threshold: Minimum confidence for text detection (0.0-1.0).
    """

    # Keywords that indicate game over screens
    VICTORY_KEYWORDS = [
        'winner', 'victory', 'chicken dinner', 'win', 'won',
        'first place', '#1', '1st', 'champion', 'congratulations'
    ]

    DEFEAT_KEYWORDS = [
        'eliminated', 'died', 'death', 'defeated', 'killed',
        'game over', 'you died', 'wasted', 'dead'
    ]

    PLACEMENT_KEYWORDS = [
        'place', 'rank', 'placement', 'position', 'finished',
        '#', 'st', 'nd', 'rd', 'th'
    ]

    STATS_KEYWORDS = [
        'kills', 'damage', 'time', 'survival', 'eliminations',
        'assists', 'revives', 'distance', 'accuracy'
    ]

    def __init__(self,
                 lang: str = 'en',
                 use_gpu: bool = False,
                 confidence_threshold: float = 0.6):
        """Initialize game over detector.

        Args:
            lang: Language code for OCR (en, ch, fr, de, es, etc.).
            use_gpu: Enable GPU acceleration (requires paddlepaddle-gpu).
            confidence_threshold: Minimum confidence for text detection (0.0-1.0).
                Lower values detect more text but with lower accuracy.

        Example:
            >>> detector = GameOverDetector(use_gpu=True, confidence_threshold=0.7)
        """
        self.confidence_threshold = confidence_threshold

        print(f"Initializing PaddleOCR (lang={lang}, gpu={use_gpu})...")
        self.ocr = PaddleOCR(
            lang=lang,
            use_gpu=use_gpu,
            use_angle_cls=True,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5
        )
        print("✓ PaddleOCR initialized")

    def extract_text_from_frame(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple]]:
        """Extract all text from frame with positions.

        Args:
            frame: Input frame in BGR format.

        Returns:
            List of tuples: (text, confidence, bounding_box).
            - text: Detected text string (str)
            - confidence: Detection confidence 0.0-1.0 (float)
            - bounding_box: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (list)
            Only returns text with confidence >= confidence_threshold.

        Example:
            >>> texts = detector.extract_text_from_frame(frame)
            >>> for text, conf, bbox in texts:
            ...     print(f"{text} ({conf:.2f})")
        """
        result = self.ocr.ocr(frame, cls=True)

        if not result or not result[0]:
            return []

        texts = []
        for line in result[0]:
            if line:
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]  # (text, confidence)
                text = text_info[0]
                confidence = text_info[1]

                if confidence >= self.confidence_threshold:
                    texts.append((text, confidence, bbox))

        return texts

    def detect_game_over_screen(self, texts: List[Tuple[str, float, Tuple]]) -> Dict:
        """Detect if frame is a game over screen and determine outcome.

        Uses keyword matching to identify game over screens and classify
        the outcome (victory, elimination, or generic ended).

        Args:
            texts: List of (text, confidence, bbox) tuples from extract_text_from_frame.

        Returns:
            Dictionary with detection results:
            - 'is_game_over': True if game over screen detected (bool)
            - 'outcome': 'victory', 'eliminated', 'ended', or 'unknown' (str)
            - 'confidence': 'high', 'medium', or 'low' (str)
            - 'victory_score': Number of victory keywords found (int)
            - 'defeat_score': Number of defeat keywords found (int)
            - 'stats_score': Number of stats keywords found (int)
            - 'all_text': Combined text from all detections (str)

        Example:
            >>> texts = detector.extract_text_from_frame(frame)
            >>> result = detector.detect_game_over_screen(texts)
            >>> if result['is_game_over']:
            ...     print(f"Game over: {result['outcome']}")
        """
        # Combine all text
        all_text = ' '.join([t[0].lower() for t in texts])

        # Check for victory
        victory_score = sum(1 for keyword in self.VICTORY_KEYWORDS if keyword in all_text)

        # Check for defeat/elimination
        defeat_score = sum(1 for keyword in self.DEFEAT_KEYWORDS if keyword in all_text)

        # Check for placement/stats (indicates end screen)
        stats_score = sum(1 for keyword in self.STATS_KEYWORDS if keyword in all_text)
        placement_score = sum(1 for keyword in self.PLACEMENT_KEYWORDS if keyword in all_text)

        # Determine if it's a game over screen
        is_game_over = (victory_score > 0 or defeat_score > 0 or
                       (stats_score >= 2 and placement_score > 0))

        # Determine outcome
        outcome = 'unknown'
        confidence = 'low'

        if victory_score >= 2:
            outcome = 'victory'
            confidence = 'high'
        elif victory_score == 1 and stats_score >= 2:
            outcome = 'victory'
            confidence = 'medium'
        elif defeat_score >= 2:
            outcome = 'eliminated'
            confidence = 'high'
        elif defeat_score == 1 and stats_score >= 2:
            outcome = 'eliminated'
            confidence = 'medium'
        elif stats_score >= 3 and placement_score > 0:
            outcome = 'ended'  # Game ended (victory or defeat)
            confidence = 'medium'

        return {
            'is_game_over': is_game_over,
            'outcome': outcome,
            'confidence': confidence,
            'victory_score': victory_score,
            'defeat_score': defeat_score,
            'stats_score': stats_score,
            'all_text': all_text
        }

    def extract_stats(self, texts: List[Tuple[str, float, Tuple]]) -> Dict:
        """Extract game statistics from detected text.

        Uses regex patterns to extract numeric stats like placement,
        kills, damage, and survival time from OCR text.

        Args:
            texts: List of (text, confidence, bbox) tuples.

        Returns:
            Dictionary with extracted stats:
            - 'placement': Rank/placement number (int or None)
            - 'kills': Number of kills (int or None)
            - 'damage': Damage dealt (int or None)
            - 'survival_time': Time in MM:SS format (str or None)
            - 'assists': Number of assists (int or None)
            - 'revives': Number of revives (int or None)
            - 'raw_text': List of all detected text strings

        Example:
            >>> texts = detector.extract_text_from_frame(frame)
            >>> stats = detector.extract_stats(texts)
            >>> print(f"Placement: #{stats['placement']}, Kills: {stats['kills']}")
        """
        stats = {
            'placement': None,
            'kills': None,
            'damage': None,
            'survival_time': None,
            'assists': None,
            'revives': None,
            'raw_text': []
        }

        for text, conf, bbox in texts:
            text_lower = text.lower()
            stats['raw_text'].append(text)

            # Extract placement (rank)
            # Look for patterns like: "#5", "5th", "Rank: 5", "Place: 5"
            placement_patterns = [
                r'#\s*(\d+)',
                r'(\d+)\s*(?:st|nd|rd|th)',
                r'(?:rank|place|placement|position)[\s:]*(\d+)',
                r'finished\s+(\d+)'
            ]
            for pattern in placement_patterns:
                match = re.search(pattern, text_lower)
                if match and stats['placement'] is None:
                    try:
                        stats['placement'] = int(match.group(1))
                    except (ValueError, IndexError):
                        pass

            # Extract kills
            # Look for patterns like: "Kills: 5", "5 Kills", "K: 5", "5 eliminations"
            kill_patterns = [
                r'(?:kills?|eliminations?)[\s:]*(\d+)',
                r'(\d+)\s+(?:kills?|eliminations?)',
                r'k[\s:]+(\d+)'
            ]
            for pattern in kill_patterns:
                match = re.search(pattern, text_lower)
                if match and stats['kills'] is None:
                    try:
                        stats['kills'] = int(match.group(1))
                    except (ValueError, IndexError):
                        pass

            # Extract damage
            # Look for patterns like: "Damage: 487", "487 Damage", "DMG: 487"
            damage_patterns = [
                r'(?:damage|dmg)[\s:]*(\d+)',
                r'(\d+)\s+(?:damage|dmg)'
            ]
            for pattern in damage_patterns:
                match = re.search(pattern, text_lower)
                if match and stats['damage'] is None:
                    try:
                        stats['damage'] = int(match.group(1))
                    except (ValueError, IndexError):
                        pass

            # Extract survival time
            # Look for patterns like: "15:30", "15m 30s", "Time: 15:30"
            time_patterns = [
                r'(\d+):(\d+)',
                r'(\d+)\s*m\s*(\d+)\s*s',
                r'(?:time|survival)[\s:]*(\d+):(\d+)'
            ]
            for pattern in time_patterns:
                match = re.search(pattern, text_lower)
                if match and stats['survival_time'] is None:
                    try:
                        minutes = int(match.group(1))
                        seconds = int(match.group(2))
                        stats['survival_time'] = f"{minutes:02d}:{seconds:02d}"
                    except (ValueError, IndexError):
                        pass

            # Extract assists
            assist_patterns = [
                r'(?:assists?)[\s:]*(\d+)',
                r'(\d+)\s+(?:assists?)'
            ]
            for pattern in assist_patterns:
                match = re.search(pattern, text_lower)
                if match and stats['assists'] is None:
                    try:
                        stats['assists'] = int(match.group(1))
                    except (ValueError, IndexError):
                        pass

            # Extract revives
            revive_patterns = [
                r'(?:revives?)[\s:]*(\d+)',
                r'(\d+)\s+(?:revives?)'
            ]
            for pattern in revive_patterns:
                match = re.search(pattern, text_lower)
                if match and stats['revives'] is None:
                    try:
                        stats['revives'] = int(match.group(1))
                    except (ValueError, IndexError):
                        pass

        return stats

    def analyze_frame(self, frame: np.ndarray, timestamp: float) -> Dict:
        """Analyze a single frame for game over detection and stats.

        Combines text extraction, game over detection, and stats extraction
        into a single convenient method.

        Args:
            frame: Input frame in BGR format.
            timestamp: Frame timestamp in seconds.

        Returns:
            Dictionary with complete analysis:
            - 'timestamp': Frame timestamp (float)
            - 'is_game_over': True if game over detected (bool)
            - 'outcome': Outcome classification (str)
            - 'confidence': Detection confidence level (str)
            - 'stats': Dictionary of extracted stats
            - 'detection_scores': Dictionary of keyword match scores

        Example:
            >>> result = detector.analyze_frame(frame, 450.5)
            >>> if result['is_game_over']:
            ...     print(f"Game ended at {result['timestamp']}s")
            ...     print(f"Placement: #{result['stats']['placement']}")
        """
        # Extract all text from frame
        texts = self.extract_text_from_frame(frame)

        # Detect if game over
        detection = self.detect_game_over_screen(texts)

        # If game over, extract stats
        stats = {}
        if detection['is_game_over']:
            stats = self.extract_stats(texts)

        return {
            'timestamp': timestamp,
            'is_game_over': detection['is_game_over'],
            'outcome': detection['outcome'],
            'confidence': detection['confidence'],
            'stats': stats,
            'detection_scores': {
                'victory': detection['victory_score'],
                'defeat': detection['defeat_score'],
                'stats_present': detection['stats_score']
            }
        }


def extract_frames(video_path: str,
                  interval_seconds: float = 2.0,
                  max_frames: Optional[int] = None) -> Generator[Tuple[float, np.ndarray], None, None]:
    """Generator that yields frames from video at regular intervals.

    Args:
        video_path: Path to video file.
        interval_seconds: Time between frames in seconds.
        max_frames: Maximum frames to extract. None = all frames.

    Yields:
        Tuple of (timestamp, frame):
        - timestamp: Frame timestamp in seconds (float)
        - frame: Frame as numpy array in BGR format

    Raises:
        ValueError: If video cannot be opened.

    Example:
        >>> for timestamp, frame in extract_frames('video.mp4', 5.0):
        ...     print(f"Frame at {timestamp}s")
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

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            yield timestamp, frame

            frames_extracted += 1

            if max_frames and frames_extracted >= max_frames:
                break

        frame_count += 1

    cap.release()


def scan_video_for_game_over(video_path: str,
                             detector: GameOverDetector,
                             interval_seconds: float = 2.0,
                             start_time: float = 0.0,
                             max_duration: Optional[float] = None,
                             verbose: bool = False) -> List[Dict]:
    """Scan video for game over screens and extract stats.

    High-level convenience function for detecting all game over events
    in a video.

    Args:
        video_path: Path to video file.
        detector: Initialized GameOverDetector instance.
        interval_seconds: Frame sampling interval in seconds.
        start_time: Start scanning at this timestamp in seconds.
        max_duration: Maximum duration to scan in seconds. None = entire video.
        verbose: Print detection details.

    Returns:
        List of game over event dictionaries. Each event contains:
        - timestamp, is_game_over, outcome, confidence, stats, detection_scores

    Example:
        >>> detector = GameOverDetector(use_gpu=True)
        >>> events = scan_video_for_game_over('gameplay.mkv', detector)
        >>> print(f"Found {len(events)} game over events")
    """
    game_over_events = []
    frame_count = 0

    # Calculate max frames
    max_frames = None
    if max_duration:
        max_frames = int(max_duration / interval_seconds)

    for timestamp, frame in extract_frames(video_path, interval_seconds, max_frames):
        # Skip frames before start time
        if timestamp < start_time:
            continue

        frame_count += 1

        if verbose:
            print(f"\n[{timestamp:7.1f}s] Analyzing frame {frame_count}...", end=' ')
        else:
            if frame_count % 10 == 0:
                print(f"  Processed {frame_count} frames...", end='\r')

        # Analyze frame
        result = detector.analyze_frame(frame, timestamp)

        # If game over detected
        if result['is_game_over']:
            game_over_events.append(result)

            if verbose:
                print(f"✓ GAME OVER DETECTED!")
                print(f"    Outcome: {result['outcome']}")
                print(f"    Confidence: {result['confidence']}")
                if result['stats'].get('placement'):
                    print(f"    Placement: #{result['stats']['placement']}")
                if result['stats'].get('kills') is not None:
                    print(f"    Kills: {result['stats']['kills']}")
                if result['stats'].get('damage'):
                    print(f"    Damage: {result['stats']['damage']}")
                if result['stats'].get('survival_time'):
                    print(f"    Time: {result['stats']['survival_time']}")
            else:
                print(f"\n  ✓ Game over detected at {timestamp:.1f}s ({result['outcome']})")

        elif verbose:
            print("No game over")

    return game_over_events
