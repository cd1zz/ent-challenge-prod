"""Action Classifier using CLIP for Gameplay Analysis.

This module provides zero-shot classification of gameplay actions using OpenAI's
CLIP (Contrastive Language-Image Pre-training) model. It enables frame-by-frame
analysis of game videos to identify player activities, game states, and transitions.

The classifier uses pre-trained CLIP models to match visual frames against textual
descriptions of game actions without requiring task-specific training data.

Custom labels can be defined project-wide in .env file:
    CLIP_LABELS='label1|label2|label3|...'

Typical usage example:

    from action_classifier import ActionClassifier, classify_video_actions

    # Initialize classifier (automatically loads labels from .env if present)
    classifier = ActionClassifier(model_name="ViT-B/32")

    # Classify single frame
    predictions = classifier.classify_frame(frame_array)

    # Or process entire video
    results = classify_video_actions("gameplay.mp4", interval_seconds=2.0)

References:
    CLIP Paper: https://arxiv.org/abs/2103.00020
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ActionClassifier:
    """Classify gameplay actions using CLIP zero-shot learning.

    This classifier uses OpenAI's CLIP model to perform zero-shot classification
    of gameplay frames against predefined action labels. The text features are
    precomputed for efficiency when processing multiple frames.

    Attributes:
        device (str): Computation device ('cuda' or 'cpu').
        model: Loaded CLIP model instance.
        preprocess: CLIP preprocessing function for images.
        labels (List[str]): Action labels for classification.
        num_labels (int): Total number of action labels.
        text_features (torch.Tensor): Precomputed CLIP embeddings for labels.
    """

    # Default action labels for battle royale gameplay
    DEFAULT_LABELS = [
        # Gameplay actions
        "player looting items and equipment indoors",
        "player in active combat shooting at enemies",
        "player driving a vehicle on road",
        "player running and rotating between zones",
        "player healing and managing inventory",
        "player hiding and taking cover in building",
        "helicopters visible on screen before parachute drop",
        "player parachuting and landing",
        "player aiming down sights at enemy",
        "player in open field moving tactically",
        "player dead or spectating teammates",
        "player using inventory menu",
        "player inside vehicle as passenger",

        # Game state transitions (for segmentation)
        "main menu or lobby screen",
        "victory screen showing final placement",
        "defeat screen or player eliminated",
        "post-game statistics and summary screen",
        "loading screen or match starting",
    ]

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None,
        labels: Optional[List[str]] = None
    ) -> None:
        """Initialize CLIP action classifier.

        Args:
            model_name: CLIP model architecture to use. Available options:
                - 'ViT-B/32': Fast, good for real-time (default)
                - 'ViT-B/16': Better accuracy, slower
                - 'ViT-L/14': Best accuracy, slowest
            device: Computation device ('cuda' or 'cpu'). Auto-detects if None.
            labels: Custom action labels for classification. Priority order:
                1. labels parameter (if provided)
                2. CLIP_LABELS environment variable (if set in .env)
                3. DEFAULT_LABELS (fallback)

        Raises:
            RuntimeError: If CLIP model fails to load.

        Example:
            >>> classifier = ActionClassifier(
            ...     model_name="ViT-B/32",
            ...     device="cuda"
            ... )
            >>> print(f"Loaded {classifier.num_labels} labels")

            >>> # Or use custom labels from .env file:
            >>> # In .env: CLIP_LABELS='custom1|custom2|custom3'
            >>> classifier = ActionClassifier()  # Loads from .env
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP model '{model_name}' on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print("✓ CLIP model loaded")

        # Set labels with priority: parameter > env var > default
        if labels is not None:
            # User-provided labels take highest priority
            self.labels = labels
            print(f"Using {len(labels)} custom labels (parameter)")
        else:
            # Try to load from environment variable
            env_labels = os.getenv('CLIP_LABELS')
            if env_labels:
                # Parse pipe-delimited labels
                self.labels = [label.strip() for label in env_labels.split('|') if label.strip()]
                print(f"Using {len(self.labels)} custom labels (from .env)")
            else:
                # Fallback to defaults
                self.labels = self.DEFAULT_LABELS
                print(f"Using {len(self.labels)} default labels")

        self.num_labels = len(self.labels)

        # Precompute text features for efficiency
        self._precompute_text_features()

    def _precompute_text_features(self) -> None:
        """Precompute CLIP text embeddings for all labels.

        This method tokenizes and encodes all action labels into CLIP's embedding
        space. The embeddings are normalized and cached to avoid recomputation
        during frame classification.

        The precomputed features significantly improve performance when processing
        multiple frames with the same label set.
        """
        print(f"Encoding {self.num_labels} action labels...")
        text_inputs = clip.tokenize(self.labels).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        print("✓ Text features computed")

    def classify_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Classify a single frame against all action labels.

        Args:
            frame: Frame as numpy array with shape (H, W, C) in RGB format,
                or a PIL Image object. Height and width can be arbitrary as
                CLIP handles resizing internally.

        Returns:
            Dictionary mapping each action label to its probability (0.0-1.0).
            Probabilities sum to 1.0 across all labels.

        Example:
            >>> import numpy as np
            >>> frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            >>> predictions = classifier.classify_frame(frame)
            >>> top_action = max(predictions.items(), key=lambda x: x[1])
            >>> print(f"Top action: {top_action[0]} ({top_action[1]:.2%})")
        """
        # Convert numpy to PIL
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = frame

        # Preprocess image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        # Compute image features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

        # Convert to dictionary
        probs = similarity.cpu().numpy()[0]
        return {label: float(prob) for label, prob in zip(self.labels, probs)}

    def classify_top_k(self, frame: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k action predictions for a frame.

        Args:
            frame: Frame as numpy array with shape (H, W, C) in RGB format.
            k: Number of top predictions to return. Must be positive and not
                exceed the total number of labels.

        Returns:
            List of (label, probability) tuples sorted by probability in
            descending order. Returns exactly k items unless fewer labels exist.

        Example:
            >>> top_3 = classifier.classify_top_k(frame, k=3)
            >>> for action, prob in top_3:
            ...     print(f"{action}: {prob:.2%}")
            player in active combat: 45.2%
            player aiming down sights: 32.1%
            player running between zones: 15.7%
        """
        predictions = self.classify_frame(frame)
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:k]

    def classify_batch(self, frames: List[np.ndarray]) -> List[Dict[str, float]]:
        """Classify multiple frames in a single batch for improved efficiency.

        Batch processing is significantly faster than processing frames individually
        when GPU acceleration is available. All frames are processed in parallel.

        Args:
            frames: List of frames as numpy arrays with shape (H, W, C) in RGB
                format, or PIL Image objects. Frames can have different dimensions.

        Returns:
            List of prediction dictionaries, one per input frame, in the same
            order. Each dictionary maps action labels to probabilities (0.0-1.0).

        Example:
            >>> frames = [frame1, frame2, frame3]  # List of numpy arrays
            >>> predictions = classifier.classify_batch(frames)
            >>> for i, preds in enumerate(predictions):
            ...     top = max(preds.items(), key=lambda x: x[1])
            ...     print(f"Frame {i}: {top[0]} ({top[1]:.2%})")
        """
        # Convert all frames to PIL and preprocess
        images = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                images.append(self.preprocess(Image.fromarray(frame)))
            else:
                images.append(self.preprocess(frame))

        # Stack into batch
        image_batch = torch.stack(images).to(self.device)

        # Compute features
        with torch.no_grad():
            image_features = self.model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarities
            similarities = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)

        # Convert to list of dictionaries
        results = []
        for i in range(len(frames)):
            probs = similarities[i].cpu().numpy()
            results.append({label: float(prob) for label, prob in zip(self.labels, probs)})

        return results

    def get_primary_action(
        self,
        predictions: Dict[str, float],
        threshold: float = 0.3
    ) -> str:
        """Extract the primary action from classification predictions.

        Returns the action with the highest probability if it exceeds the
        confidence threshold, otherwise returns "uncertain".

        Args:
            predictions: Dictionary mapping action labels to probabilities.
            threshold: Minimum probability threshold for a confident prediction.
                Actions below this threshold are considered uncertain. Should be
                between 0.0 and 1.0. Default is 0.3 (30% confidence).

        Returns:
            The action label with highest probability if above threshold,
            otherwise "uncertain".

        Example:
            >>> predictions = classifier.classify_frame(frame)
            >>> action = classifier.get_primary_action(predictions, threshold=0.4)
            >>> if action != "uncertain":
            ...     print(f"Confident action: {action}")
        """
        if not predictions:
            return "uncertain"

        top_action = max(predictions.items(), key=lambda x: x[1])
        if top_action[1] >= threshold:
            return top_action[0]
        return "uncertain"


def _save_checkpoint(checkpoint_file: str, results: List[Dict], video_path: str,
                    interval_seconds: float, labels: List[str]) -> None:
    """Save checkpoint data to file."""
    checkpoint_data = {
        'video_path': video_path,
        'interval_seconds': interval_seconds,
        'labels': labels,
        'results': results,
        'last_processed_frame': len(results)
    }
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)


def _load_checkpoint(checkpoint_file: str) -> Optional[Dict]:
    """Load checkpoint data from file."""
    if not Path(checkpoint_file).exists():
        return None
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return None


def classify_video_actions(
    video_path: str,
    interval_seconds: float = 2.0,
    max_frames: Optional[int] = None,
    batch_size: int = 16,
    output_file: Optional[str] = None,
    checkpoint_file: Optional[str] = None,
    checkpoint_interval: int = 100,
    verbose: bool = True
) -> List[Dict]:
    """Classify actions across an entire video file.

    This convenience function extracts frames at regular intervals from a video
    and classifies each frame using CLIP. Results include timestamps, primary
    actions, and full prediction distributions.

    Args:
        video_path: Path to input video file. Must be a format supported by
            OpenCV (e.g., .mp4, .avi, .mkv).
        interval_seconds: Time interval between sampled frames in seconds.
            Smaller values provide more temporal resolution but increase
            processing time. Default is 2.0 seconds.
        max_frames: Maximum number of frames to process. If None, processes
            entire video. Useful for testing or processing video segments.
        batch_size: Number of frames to process in each batch. Larger batches
            are more efficient but require more GPU memory. Default is 16.
        output_file: Optional path to save results as JSON. If None, results
            are only returned. File will contain metadata and full predictions.
        verbose: If True, prints progress information including timestamps and
            detected actions. Default is True.

    Returns:
        List of dictionaries, one per classified frame. Each dictionary contains:
        - 'timestamp': Frame timestamp in seconds (float)
        - 'primary_action': Most likely action label (str)
        - 'top_3_predictions': List of top 3 predictions with probabilities
        - 'all_predictions': Complete probability distribution (dict)

    Raises:
        FileNotFoundError: If video_path does not exist.
        ValueError: If video cannot be opened or read.

    Example:
        >>> results = classify_video_actions(
        ...     video_path="gameplay.mp4",
        ...     interval_seconds=2.0,
        ...     output_file="results.json",
        ...     verbose=True
        ... )
        >>> print(f"Processed {len(results)} frames")
        >>> print(f"First action at {results[0]['timestamp']}s: "
        ...       f"{results[0]['primary_action']}")
    """
    from .frame_extractor import FrameExtractor

    if verbose:
        print(f"\nClassifying actions in video: {video_path}")
        print(f"Sample interval: {interval_seconds}s")
        print("=" * 60)

    # Initialize classifier
    classifier = ActionClassifier()

    # Extract and classify frames
    results = []
    batch_frames = []
    batch_timestamps = []

    with FrameExtractor(video_path) as extractor:
        if verbose:
            info = extractor.get_info()
            print(f"Video duration: {info['duration_formatted']}")
            print(f"Processing frames...\n")

        for timestamp, frame in extractor.extract_frames(interval_seconds=interval_seconds,
                                                         max_frames=max_frames):
            batch_frames.append(frame)
            batch_timestamps.append(timestamp)

            # Process in batches
            if len(batch_frames) >= batch_size:
                predictions_batch = classifier.classify_batch(batch_frames)

                for ts, preds in zip(batch_timestamps, predictions_batch):
                    top_action = classifier.get_primary_action(preds)
                    top_3 = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:3]

                    results.append({
                        'timestamp': ts,
                        'primary_action': top_action,
                        'top_3_predictions': [{'action': a, 'probability': p} for a, p in top_3],
                        'all_predictions': preds
                    })

                    if verbose:
                        print(f"[{ts:7.1f}s] {top_action} ({top_3[0][1]*100:.1f}%)")

                # Save checkpoint periodically
                if checkpoint_file and len(results) % checkpoint_interval == 0:
                    _save_checkpoint(checkpoint_file, results, video_path, interval_seconds, classifier.labels)
                    if verbose:
                        print(f"  → Checkpoint saved ({len(results)} frames)")

                batch_frames = []
                batch_timestamps = []

        # Process remaining frames
        if batch_frames:
            predictions_batch = classifier.classify_batch(batch_frames)
            for ts, preds in zip(batch_timestamps, predictions_batch):
                top_action = classifier.get_primary_action(preds)
                top_3 = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:3]

                results.append({
                    'timestamp': ts,
                    'primary_action': top_action,
                    'top_3_predictions': [{'action': a, 'probability': p} for a, p in top_3],
                    'all_predictions': preds
                })

                if verbose:
                    print(f"[{ts:7.1f}s] {top_action} ({top_3[0][1]*100:.1f}%)")

    # Save results if requested
    if output_file:
        output_data = {
            'video_path': video_path,
            'timestamp': datetime.now().isoformat(),
            'interval_seconds': interval_seconds,
            'total_frames': len(results),
            'labels': classifier.labels,
            'results': results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\n✓ Results saved to: {output_file}")

    # Clean up checkpoint file on successful completion
    if checkpoint_file and Path(checkpoint_file).exists():
        Path(checkpoint_file).unlink()
        if verbose:
            print(f"✓ Processing complete, checkpoint file removed")

    if verbose:
        print(f"\n✓ Classified {len(results)} frames")

        # Print summary
        action_counts = {}
        for r in results:
            action = r['primary_action']
            action_counts[action] = action_counts.get(action, 0) + 1

        print("\nAction Distribution:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results)) * 100
            print(f"  {action}: {count} ({percentage:.1f}%)")

    return results
