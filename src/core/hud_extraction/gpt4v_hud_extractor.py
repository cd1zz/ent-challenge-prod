"""GPT-4 Vision HUD Extractor with Intelligent Sampling.

This module provides high-accuracy extraction of Heads-Up Display (HUD) elements
from gameplay videos using OpenAI's GPT-4 Vision API. It employs intelligent
sampling strategies to minimize API costs while maintaining comprehensive coverage
of game events.

The extractor analyzes video frames to extract structured data from HUD elements
like health bars, ammo counts, team status, and game statistics. It uses CLIP
action classifications and audio transcripts to intelligently select which frames
to analyze, avoiding redundant extractions during static gameplay.

Key features:
- Intelligent frame sampling based on action triggers and audio events
- Automatic retry and interpolation for failed extractions
- Cost-optimized processing with configurable intervals
- Structured JSON output with timestamp metadata

Typical usage example:

    from gpt4v_hud_extractor import extract_hud_values_gpt4v

    results = extract_hud_values_gpt4v(
        video_path="gameplay.mp4",
        clip_file="actions.json",
        transcript_file="transcript.json",
        output_file="hud_data.json",
        verbose=True
    )

References:
    GPT-4V: https://platform.openai.com/docs/guides/vision
"""

import base64
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .hud_config import get_gpt4v_field_descriptions
from .intelligent_sampler import IntelligentSampler

load_dotenv()


class GPT4VHUDExtractor:
    """Extract HUD values using GPT-4 Vision with intelligent sampling.

    This class manages the extraction pipeline for HUD data from gameplay videos.
    It coordinates frame extraction, GPT-4V API calls, and result aggregation
    with intelligent sampling to optimize costs.

    The extractor uses a centralized HUD configuration to define which fields to
    extract and provides detailed prompts to GPT-4V for accurate recognition.

    Attributes:
        model (str): OpenAI model identifier (e.g., "gpt-4o", "gpt-4-vision-preview").
        client (OpenAI): Authenticated OpenAI API client.
        hud_fields (Dict[str, str]): Mapping of HUD field names to descriptions.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        """Initialize GPT-4V HUD extractor.

        Args:
            model: OpenAI model identifier to use for vision analysis.
                Common options:
                - "gpt-4o": Faster and cheaper, good for most use cases
                - "gpt-4-turbo": Balanced performance
                - "gpt-4-vision-preview": Original vision model
                Default is "gpt-4o".

        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set.

        Example:
            >>> from dotenv import load_dotenv
            >>> load_dotenv()
            >>> extractor = GPT4VHUDExtractor(model="gpt-4o")
            >>> print(f"Using model: {extractor.model}")
            >>> print(f"Extracting {len(extractor.hud_fields)} HUD fields")
        """
        self.model = model
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Load HUD fields from centralized config
        self.hud_fields = get_gpt4v_field_descriptions()

    def extract_frame(
        self,
        video_path: str,
        timestamp: float,
        output_path: str
    ) -> None:
        """Extract single frame from video at specified timestamp using ffmpeg.

        Args:
            video_path: Path to input video file.
            timestamp: Time in seconds at which to extract frame.
            output_path: Path where extracted frame image will be saved.

        Note:
            Uses ffmpeg's -ss (seek) option for accurate frame extraction.
            Errors are suppressed (check=False) to handle edge cases gracefully.
        """
        cmd = [
            'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
            '-vframes', '1', '-f', 'image2', output_path
        ]
        subprocess.run(cmd, capture_output=True, check=False)

    def encode_image(self, image_path: str) -> str:
        """Encode image file to base64 string for API transmission.

        Args:
            image_path: Path to image file to encode.

        Returns:
            Base64-encoded string representation of the image.

        Raises:
            FileNotFoundError: If image_path does not exist.
            IOError: If image cannot be read.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract_hud_from_frame(self, frame_path: str) -> Dict:
        """Extract all HUD values from a single frame using GPT-4 Vision.

        Sends the frame image to GPT-4V with a structured prompt describing
        the HUD fields to extract. The prompt includes detailed instructions
        for accuracy and exact formatting requirements.

        Args:
            frame_path: Path to frame image file (PNG or JPEG format).

        Returns:
            Dictionary containing extracted HUD values with field names as keys.
            Fields not visible in the frame will have null values. Example:
            {
                "compass_heading": "330",
                "game_status": "Kills: 3, Teams: 18, Soldiers: 67",
                "player_health": "110/110",
                "weapon_and_ammo": "MG 30.0 | 30/120",
                ...
            }

        Raises:
            FileNotFoundError: If frame_path does not exist.
            json.JSONDecodeError: If GPT-4V response is not valid JSON.
            Exception: If API call fails or returns unexpected format.

        Note:
            The function automatically strips markdown code blocks from the
            response if present, as GPT-4V sometimes wraps JSON in ```json```.
        """
        # Build field list for prompt
        field_list = '\n'.join([
            f'- {name}: {desc}'
            for name, desc in self.hud_fields.items()
        ])

        prompt = f"""You are analyzing a Super People gameplay HUD screenshot. Extract ALL visible UI values with MAXIMUM ACCURACY.

HUD Fields to extract:
{field_list}

CRITICAL INSTRUCTIONS:
1. Extract text EXACTLY as displayed (preserve spaces, slashes, formatting)
2. Use null for fields not visible or obscured
3. For health bars, estimate percentage (0-100) if exact value not shown
4. For inventory items, return true/false based on visibility
5. Be precise - this data is used for behavioral analysis

Return ONLY a JSON object with these exact field names:
{{
  "compass_heading": "330",
  "game_status": "Kills: 3, Teams: 18, Soldiers: 67",
  "player_rank": "Lv 32 Firearms Expert",
  "equipped_accessories": "Enhanced Aim, Tracer Rounds, Fast Reload",
  "weapon_and_ammo": "MG 30.0 | 30/120",
  "player_health": "110/110",
  "team_health": "Player1: 85/100, Player2: 92/100, Player3: 78/100"
}}

Extract ALL fields, use exact formatting."""

        # Encode image
        base64_image = self.encode_image(frame_path)

        # Call GPT-4V
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=600
        )

        # Parse response
        result = response.choices[0].message.content

        # Extract JSON
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()

        return json.loads(result)

    def extract_from_video_intelligent(
        self,
        video_path: str,
        clip_file: str,
        transcript_file: Optional[str] = None,
        output_file: str = None,
        base_interval: float = 2.0,
        min_interval: float = 4.0,
        max_interval: float = 18.0,
        temp_dir: Optional[str] = None,
        verbose: bool = True
    ) -> tuple[List[Dict], Dict]:
        """Extract HUD values from video using intelligent sampling strategy.

        This method analyzes the video using CLIP action classifications and
        optional audio transcripts to identify important moments that warrant
        HUD extraction. It balances comprehensive coverage with cost optimization
        by avoiding redundant extractions during static gameplay.

        The intelligent sampler triggers extractions based on:
        - Action state changes (combat, looting, healing, etc.)
        - Audio events (callouts, important game sounds)
        - Maximum time thresholds to ensure minimum coverage

        Args:
            video_path: Path to input video file. Must be accessible by ffmpeg.
            clip_file: Path to JSON file containing CLIP action classifications
                with timestamps and action labels.
            transcript_file: Optional path to JSON file containing audio transcript
                with timestamps and text chunks. Enables audio-based triggers.
            output_file: Path where results will be saved. Supports .json (full
                metadata) or .csv (tabular format) extensions. If None, results
                are only returned.
            base_interval: Base time interval in seconds for checking triggers.
                Smaller values provide finer temporal resolution. Default is 2.0s.
            min_interval: Minimum seconds between consecutive extractions to avoid
                redundancy. Default is 4.0s for production use.
            max_interval: Maximum seconds without extraction to ensure minimum
                coverage density. Default is 18.0s.
            temp_dir: Directory for storing temporary frame images. If None,
                creates 'gpt4v_frames' subdirectory in output file's directory.
            verbose: If True, prints detailed progress including frame-by-frame
                results, costs, and sampling statistics. Default is True.

        Returns:
            Tuple of (results, stats) where:
            - results: List of dictionaries containing extracted HUD values for
              each sampled frame. Each dict includes 'timestamp' and
              'extraction_reason' fields plus all HUD fields.
            - stats: Dictionary with sampling statistics including total frames
              sampled, baseline comparison, cost estimates, and savings.

        Raises:
            FileNotFoundError: If video_path, clip_file, or transcript_file not found.
            subprocess.CalledProcessError: If ffmpeg/ffprobe fails.
            json.JSONDecodeError: If input files are malformed JSON.

        Example:
            >>> extractor = GPT4VHUDExtractor()
            >>> results, stats = extractor.extract_from_video_intelligent(
            ...     video_path="gameplay.mp4",
            ...     clip_file="actions.json",
            ...     transcript_file="transcript.json",
            ...     output_file="hud_data.json",
            ...     min_interval=4.0,
            ...     max_interval=18.0,
            ...     verbose=True
            ... )
            ======================================================================
            GPT-4V HUD EXTRACTION (Intelligent Sampling)
            ======================================================================

            Video: gameplay.mp4
            Duration: 1847.3s

            Generating intelligent sampling schedule...
            ✓ Selected 142 frames (vs 924 baseline)
              Cost savings: 84.6%
              Estimated cost: $1.42

            Extracting HUD values from 142 frames...
            [  1/142]   45.2s (action_change_combat    ) ✓ Health:100/100      ...
            ...
        """
        if verbose:
            print("=" * 70)
            print("GPT-4V HUD EXTRACTION (Intelligent Sampling)")
            print("=" * 70)

        # Get video duration
        probe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_duration = float(result.stdout.strip())

        if verbose:
            print(f"\nVideo: {video_path}")
            print(f"Duration: {video_duration:.1f}s")

        # Load intelligent sampler with adjusted intervals for production
        sampler = IntelligentSampler.from_files(
            clip_file=clip_file,
            transcript_file=transcript_file,
            min_interval=min_interval,
            max_interval=max_interval
        )

        # Generate sampling schedule
        if verbose:
            print(f"\nGenerating intelligent sampling schedule...")

        schedule = sampler.generate_sampling_schedule(video_duration, base_interval)
        stats = sampler.get_sampling_stats(schedule, video_duration)

        if verbose:
            print(f"✓ Selected {len(schedule)} frames (vs {stats['baseline_samples']} baseline)")
            print(f"  Cost savings: {stats['cost_savings_percentage']:.1f}%")
            print(f"  Estimated cost: ${stats['estimated_cost_intelligent']:.2f}")

        # Create temp directory
        if temp_dir is None:
            temp_dir = Path(output_file).parent / 'gpt4v_frames'
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Extract HUD values
        results = []
        last_values = {}

        if verbose:
            print(f"\nExtracting HUD values from {len(schedule)} frames...")

        for i, point in enumerate(schedule):
            timestamp = point['timestamp']
            reason = point['reason']

            # Extract frame
            frame_path = temp_dir / f"frame_{timestamp:.1f}s.png"
            self.extract_frame(video_path, timestamp, str(frame_path))

            # Call GPT-4V
            if verbose:
                print(f"[{i+1:3d}/{len(schedule):3d}] {timestamp:6.1f}s ({reason:25s})", end=' ')

            try:
                hud_values = self.extract_hud_from_frame(str(frame_path))
                hud_values['timestamp'] = timestamp
                hud_values['extraction_reason'] = reason
                results.append(hud_values)
                last_values = hud_values

                if verbose:
                    # Show key values
                    health = hud_values.get('player_health', 'N/A')
                    status = hud_values.get('game_status', 'N/A')
                    print(f"✓ Health:{health:12s} Status:{status:30s}")

            except Exception as e:
                if verbose:
                    print(f"✗ Error: {e}")
                # Use interpolation from last values
                if last_values:
                    interpolated = last_values.copy()
                    interpolated['timestamp'] = timestamp
                    interpolated['extraction_reason'] = f'{reason}_failed_interpolated'
                    results.append(interpolated)

        # Save results
        output_path = Path(output_file)

        if output_path.suffix == '.json':
            # Save as JSON
            output_data = {
                'video_path': video_path,
                'extraction_timestamp': datetime.now().isoformat(),
                'model': self.model,
                'sampling_stats': stats,
                'results': results
            }
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

        else:
            # Save as CSV
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)

        if verbose:
            print(f"\n✓ Results saved to: {output_path}")
            print(f"\nExtraction complete:")
            print(f"  - Frames processed: {len(results)}")
            print(f"  - Cost: ~${len(results) * 0.01:.2f}")
            print(f"  - Savings vs baseline: ${(stats['baseline_samples'] - len(results)) * 0.01:.2f}")

        return results, stats


def extract_hud_values_gpt4v(
    video_path: str,
    clip_file: str,
    transcript_file: Optional[str] = None,
    output_file: str = None,
    min_interval: float = 4.0,
    max_interval: float = 18.0,
    verbose: bool = True
) -> Dict:
    """Convenience function to extract HUD values using GPT-4V with intelligent sampling.

    This is the main entry point for HUD extraction. It creates a GPT4VHUDExtractor
    instance and performs intelligent sampling-based extraction in a single call.

    Args:
        video_path: Path to input video file. Must be a format supported by ffmpeg.
        clip_file: Path to JSON file containing CLIP action classifications.
            Must include 'results' list with timestamp and action data.
        transcript_file: Optional path to JSON file containing audio transcript.
            Must include 'chunks' list with timestamp and text data. Enables
            audio-triggered extractions. If None, uses only visual triggers.
        output_file: Path where results will be saved. Extension determines format:
            - .json: Full metadata including model, timestamps, and statistics
            - .csv: Tabular format with one row per extraction
            If None, results are only returned (not saved).
        min_interval: Minimum seconds between consecutive HUD extractions to
            prevent over-sampling during rapid events. Default is 4.0s which
            balances coverage and cost for production use.
        max_interval: Maximum seconds without extraction to ensure baseline
            coverage even during static gameplay. Default is 18.0s.
        verbose: If True, prints detailed progress information including:
            - Video metadata and duration
            - Sampling schedule statistics
            - Frame-by-frame extraction results
            - Cost estimates and savings
            Default is True.

    Returns:
        Dictionary containing:
        - 'results': List of HUD extraction dictionaries with timestamps
        - 'stats': Sampling statistics including costs and frame counts

    Raises:
        FileNotFoundError: If video_path, clip_file, or transcript_file not found.
        ValueError: If OPENAI_API_KEY environment variable is not set.
        json.JSONDecodeError: If input JSON files are malformed.

    Example:
        >>> from dotenv import load_dotenv
        >>> load_dotenv()
        >>> result = extract_hud_values_gpt4v(
        ...     video_path="gameplay.mp4",
        ...     clip_file="output/actions.json",
        ...     transcript_file="output/transcript.json",
        ...     output_file="output/hud_data.json",
        ...     min_interval=4.0,
        ...     max_interval=18.0,
        ...     verbose=True
        ... )
        >>> print(f"Extracted {len(result['results'])} frames")
        >>> print(f"Saved ${result['stats']['cost_savings_dollars']:.2f}")
    """
    extractor = GPT4VHUDExtractor()

    results, stats = extractor.extract_from_video_intelligent(
        video_path=video_path,
        clip_file=clip_file,
        transcript_file=transcript_file,
        output_file=output_file,
        min_interval=min_interval,
        max_interval=max_interval,
        verbose=verbose
    )

    return {
        'results': results,
        'stats': stats
    }
