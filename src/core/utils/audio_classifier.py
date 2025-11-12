"""Audio Event Classifier using CLAP for Gameplay Analysis.

This module provides zero-shot classification of gameplay audio events using LAION's
CLAP (Contrastive Language-Audio Pretraining) model. It enables temporal audio
analysis to identify sound events, game states, and audio cues.

The classifier uses pre-trained CLAP models to match audio segments against textual
descriptions of sound events without requiring task-specific training data.

Custom labels can be defined project-wide in .env file:
    CLAP_LABELS='label1|label2|label3|...'

Typical usage example:

    from audio_classifier import AudioClassifier

    # Initialize classifier (automatically loads labels from .env if present)
    classifier = AudioClassifier()

    # Classify audio segment
    predictions = classifier.classify_audio_segment("segment.wav")

    # Or process video audio at intervals
    results = classifier.classify_video_audio("gameplay.mp4", interval_seconds=2.0)

References:
    CLAP Paper: https://arxiv.org/abs/2211.06687
    LAION-CLAP: https://github.com/LAION-AI/CLAP
"""

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import laion_clap
import numpy as np
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AudioClassifier:
    """Classify gameplay audio events using CLAP zero-shot learning.

    This classifier uses LAION's CLAP model to perform zero-shot classification
    of audio segments against predefined event labels. The text features are
    precomputed for efficiency when processing multiple segments.

    Attributes:
        device (str): Computation device ('cuda' or 'cpu').
        model: Loaded CLAP model instance.
        labels (List[str]): Audio event labels for classification.
        num_labels (int): Total number of event labels.
        text_features (torch.Tensor): Precomputed CLAP embeddings for labels.
        ffmpeg_path (str): Path to ffmpeg executable.
    """

    # Default audio event labels for battle royale gameplay
    DEFAULT_LABELS = [
        # Weapon sounds
        "gunshots and weapon fire",
        "automatic rifle firing",
        "sniper rifle shot",
        "shotgun blast",
        "pistol gunfire",
        "grenade explosion",
        "rocket launcher firing",

        # Combat sounds
        "intense combat with multiple gunshots",
        "distant gunfire",
        "bullet impacts and ricochets",

        # Environment sounds
        "footsteps running",
        "footsteps walking",
        "door opening or closing",
        "vehicle engine running",
        "helicopter rotor sounds",
        "airplane or parachute sounds",

        # Player actions
        "reloading weapon",
        "healing or using medical items",
        "picking up items or looting",
        "throwing grenade",

        # Game states
        "victory music or celebration sounds",
        "defeat or elimination sounds",
        "background ambient sounds",
        "menu music or UI sounds",
        "loading screen music",

        # Voice and communication
        "voice chat or teammate communication",
        "enemy voice callouts",

        # Quiet/uncertain
        "silence or very quiet ambient noise",
    ]

    def __init__(
        self,
        device: Optional[str] = None,
        labels: Optional[List[str]] = None,
        ffmpeg_path: str = r"C:\Windows\ffmpeg\ffmpeg.exe"
    ) -> None:
        """Initialize CLAP audio classifier.

        Args:
            device: Computation device ('cuda' or 'cpu'). Auto-detects if None.
            labels: Custom audio event labels for classification. Priority order:
                1. labels parameter (if provided)
                2. CLAP_LABELS environment variable (if set in .env)
                3. DEFAULT_LABELS (fallback)
            ffmpeg_path: Path to ffmpeg executable for audio extraction.

        Raises:
            RuntimeError: If CLAP model fails to load.

        Example:
            >>> classifier = AudioClassifier(device="cuda")
            >>> print(f"Loaded {classifier.num_labels} labels")
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.ffmpeg_path = ffmpeg_path

        print(f"Loading CLAP model on {self.device}...")
        self.model = laion_clap.CLAP_Module(enable_fusion=False, device=self.device)
        self.model.load_ckpt()  # Downloads pretrained weights on first run
        print("✓ CLAP model loaded")

        # Set labels with priority: parameter > env var > default
        if labels is not None:
            self.labels = labels
            print(f"Using {len(labels)} custom labels (parameter)")
        else:
            # Try to load from environment variable
            env_labels = os.getenv('CLAP_LABELS')
            if env_labels:
                self.labels = [label.strip() for label in env_labels.split('|') if label.strip()]
                print(f"Using {len(self.labels)} custom labels (from .env)")
            else:
                self.labels = self.DEFAULT_LABELS
                print(f"Using {len(self.labels)} default labels")

        self.num_labels = len(self.labels)

        # Precompute text features for efficiency
        self._precompute_text_features()

    def _precompute_text_features(self) -> None:
        """Precompute CLAP text embeddings for all labels.

        This method encodes all audio event labels into CLAP's embedding space.
        The embeddings are normalized and cached to avoid recomputation during
        audio classification.
        """
        print(f"Encoding {self.num_labels} audio event labels...")
        self.text_features = self.model.get_text_embedding(self.labels)
        print("✓ Text features computed")

    def extract_audio_segment(
        self,
        video_path: str,
        start_time: float,
        duration: float = 2.0,
        output_path: Optional[str] = None
    ) -> str:
        """Extract audio segment from video using ffmpeg.

        Args:
            video_path: Path to input video file.
            start_time: Start time in seconds.
            duration: Duration of audio segment in seconds.
            output_path: Optional output path. If None, uses temp file.

        Returns:
            Path to extracted audio WAV file.

        Raises:
            RuntimeError: If ffmpeg extraction fails.
        """
        if output_path is None:
            # Create temp file
            temp_dir = Path(tempfile.gettempdir())
            output_path = str(temp_dir / f"audio_segment_{int(start_time*1000)}.wav")

        cmd = [
            self.ffmpeg_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # PCM 16-bit
            "-ar", "48000",  # 48kHz sample rate
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            output_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg extraction failed: {result.stderr}")

        return output_path

    def classify_audio_segment(
        self,
        audio_path: str,
        cleanup: bool = False
    ) -> Dict[str, float]:
        """Classify a single audio segment against all event labels.

        Args:
            audio_path: Path to audio WAV file.
            cleanup: If True, delete audio file after classification.

        Returns:
            Dictionary mapping each event label to its probability (0.0-1.0).
            Probabilities sum to 1.0 across all labels.

        Example:
            >>> predictions = classifier.classify_audio_segment("segment.wav")
            >>> top_event = max(predictions.items(), key=lambda x: x[1])
            >>> print(f"Top event: {top_event[0]} ({top_event[1]:.2%})")
        """
        # Get audio embedding
        audio_embed = self.model.get_audio_embedding_from_filelist([audio_path])

        # Compute similarity with text features
        similarity = audio_embed @ self.text_features.T

        # Softmax to get probabilities
        probs = torch.nn.functional.softmax(torch.tensor(similarity[0]), dim=0).numpy()

        # Cleanup temp file if requested
        if cleanup and Path(audio_path).exists():
            Path(audio_path).unlink()

        return {label: float(prob) for label, prob in zip(self.labels, probs)}

    def get_primary_event(self, predictions: Dict[str, float]) -> str:
        """Get the primary (highest probability) event from predictions.

        Args:
            predictions: Dictionary of event labels to probabilities.

        Returns:
            Label of the event with highest probability.

        Example:
            >>> predictions = classifier.classify_audio_segment("segment.wav")
            >>> primary = classifier.get_primary_event(predictions)
            >>> print(f"Primary event: {primary}")
        """
        if not predictions:
            return "uncertain"
        return max(predictions.items(), key=lambda x: x[1])[0]

    def classify_video_audio(
        self,
        video_path: str,
        interval_seconds: float = 2.0,
        segment_duration: float = 2.0,
        max_duration: Optional[float] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """Classify audio events throughout a video at regular intervals.

        Args:
            video_path: Path to input video file.
            interval_seconds: Time between audio extractions in seconds.
            segment_duration: Duration of each audio segment in seconds.
            max_duration: Optional maximum video duration to process.
            verbose: If True, print progress information.

        Returns:
            List of classification results, each containing:
                - timestamp: Time in seconds
                - primary_event: Top event label
                - top_3_predictions: List of top 3 events with probabilities
                - all_predictions: Dict of all event probabilities

        Example:
            >>> results = classifier.classify_video_audio(
            ...     "gameplay.mp4",
            ...     interval_seconds=2.0,
            ...     verbose=True
            ... )
            >>> for result in results[:5]:
            ...     print(f"{result['timestamp']}s: {result['primary_event']}")
        """
        # Get video duration
        cmd = [
            self.ffmpeg_path,
            "-i", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Parse duration from ffmpeg output
        duration_line = [line for line in result.stderr.split('\n') if 'Duration' in line]
        if duration_line:
            duration_str = duration_line[0].split('Duration:')[1].split(',')[0].strip()
            h, m, s = duration_str.split(':')
            video_duration = int(h) * 3600 + int(m) * 60 + float(s)
        else:
            raise RuntimeError("Could not determine video duration")

        if max_duration:
            video_duration = min(video_duration, max_duration)

        if verbose:
            print(f"Video duration: {video_duration:.1f}s")
            print(f"Processing audio at {interval_seconds}s intervals...\n")

        results = []
        current_time = 0.0

        while current_time < video_duration:
            # Extract audio segment
            audio_path = self.extract_audio_segment(
                video_path,
                current_time,
                segment_duration
            )

            # Classify
            predictions = self.classify_audio_segment(audio_path, cleanup=True)

            # Get top results
            top_event = self.get_primary_event(predictions)
            top_3 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
            top_confidence = top_3[0][1]

            results.append({
                'timestamp': current_time,
                'primary_event': top_event,
                'top_3_predictions': [{'event': e, 'probability': p} for e, p in top_3],
                'all_predictions': predictions
            })

            if verbose:
                print(f"[{current_time:7.1f}s] {top_event} ({top_confidence*100:.1f}%)")

            current_time += interval_seconds

        return results
