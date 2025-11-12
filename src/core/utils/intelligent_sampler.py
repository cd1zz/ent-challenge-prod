"""
Intelligent Frame Sampling
Multi-modal detection to identify when GPT-4V extraction is needed.
Combines CLIP visual actions, transcript analysis, and temporal logic.
"""

import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Optional import - TranscriptAnalyzer is not yet implemented
try:
    from .transcript_analyzer import TranscriptAnalyzer
except ImportError:
    TranscriptAnalyzer = None


class IntelligentSampler:
    """
    Determines which frames need GPT-4V extraction based on multi-modal signals.
    """

    def __init__(self,
                 audio_data: Optional[Dict] = None,
                 clip_data: Optional[Dict] = None,
                 transcript_analyzer: Optional[TranscriptAnalyzer] = None,
                 min_interval: float = 3.0,
                 max_interval: float = 15.0,
                 combat_window: float = 3.0):
        """
        Initialize intelligent sampler.

        Args:
            audio_data: Audio analysis results (DEPRECATED - use transcript instead)
            clip_data: CLIP action classification results
            transcript_analyzer: TranscriptAnalyzer instance for speech-based signals
            min_interval: Minimum seconds between extractions (rate limiting)
            max_interval: Maximum seconds without extraction (catch slow changes)
            combat_window: Window in seconds to search for combat signals
        """
        self.audio_data = audio_data or {}
        self.clip_data = clip_data or {}
        self.transcript_analyzer = transcript_analyzer
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.combat_window = combat_window

        # Pre-process data for fast lookup (DEPRECATED - kept for backward compatibility)
        self.gunshots = set(self.audio_data.get('gunshots', []))
        self.combat_events = self.audio_data.get('combat_events', [])
        self.silence_periods = self.audio_data.get('silence_periods', [])

        # Combat action keywords from CLIP
        self.combat_actions = {
            'shooting', 'aiming', 'firing', 'combat', 'fighting',
            'taking cover', 'reloading', 'throwing grenade', 'explosion'
        }

        self.looting_actions = {
            'looting', 'inventory', 'menu', 'healing', 'using item'
        }

    @classmethod
    def from_files(cls, audio_file: Optional[str] = None,
                   clip_file: Optional[str] = None,
                   transcript_file: Optional[str] = None,
                   **kwargs):
        """
        Load sampler from analysis result files.

        Args:
            audio_file: Path to audio analysis JSON (DEPRECATED)
            clip_file: Path to CLIP actions JSON
            transcript_file: Path to transcript JSON (with chunks and timestamps)
            **kwargs: Additional IntelligentSampler parameters
        """
        audio_data = None
        clip_data = None
        transcript_analyzer = None

        if audio_file and Path(audio_file).exists():
            with open(audio_file) as f:
                audio_data = json.load(f)

        if clip_file and Path(clip_file).exists():
            with open(clip_file) as f:
                clip_data = json.load(f)

        if transcript_file and Path(transcript_file).exists() and TranscriptAnalyzer is not None:
            transcript_analyzer = TranscriptAnalyzer.from_file(transcript_file)

        return cls(audio_data=audio_data, clip_data=clip_data,
                   transcript_analyzer=transcript_analyzer, **kwargs)

    def has_gunshot_near(self, timestamp: float) -> bool:
        """Check if gunshot detected within combat_window seconds"""
        return any(
            abs(gunshot - timestamp) <= self.combat_window
            for gunshot in self.gunshots
        )

    def in_combat_segment(self, timestamp: float) -> bool:
        """Check if timestamp is within a combat audio segment"""
        return any(
            event['start'] <= timestamp <= event['end']
            for event in self.combat_events
        )

    def in_silence_period(self, timestamp: float) -> bool:
        """Check if timestamp is in a silence period (no action)"""
        return any(
            period['start'] <= timestamp <= period['end']
            for period in self.silence_periods
        )

    def has_combat_action_near(self, timestamp: float) -> Tuple[bool, Optional[str]]:
        """
        Check if CLIP detected combat action near timestamp.

        Returns:
            (has_combat, action_name) tuple
        """
        if not self.clip_data:
            return False, None

        results = self.clip_data.get('results', [])

        for result in results:
            result_time = result.get('timestamp', 0)
            if abs(result_time - timestamp) <= self.combat_window:
                # Check top predictions for combat actions
                predictions = result.get('predictions', [])
                for pred in predictions[:3]:  # Check top 3
                    action = pred.get('label', '').lower()
                    if any(combat_word in action for combat_word in self.combat_actions):
                        return True, pred.get('label')

        return False, None

    def has_looting_action_near(self, timestamp: float) -> bool:
        """Check if CLIP detected looting/menu action (skip these frames)"""
        if not self.clip_data:
            return False

        results = self.clip_data.get('results', [])

        for result in results:
            result_time = result.get('timestamp', 0)
            if abs(result_time - timestamp) <= self.combat_window:
                predictions = result.get('predictions', [])
                for pred in predictions[:3]:
                    action = pred.get('label', '').lower()
                    if any(loot_word in action for loot_word in self.looting_actions):
                        return True

        return False

    def should_extract(self, timestamp: float, last_extraction_time: float,
                      previous_values: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Determine if we should extract HUD values for this frame using multi-modal signals.

        Args:
            timestamp: Current frame timestamp
            last_extraction_time: When we last called GPT-4V
            previous_values: Last extracted HUD values (for state change detection)

        Returns:
            (should_extract, reason) tuple
        """
        # Rule 0: Enforce minimum interval (rate limiting)
        time_since_last = timestamp - last_extraction_time
        if time_since_last < self.min_interval:
            return False, "too_soon"

        # Rule 1: Skip if in looting/menu (HUD frozen)
        if self.has_looting_action_near(timestamp):
            return False, "looting_action"

        # Get CLIP visual signal
        has_combat_visual, combat_action = self.has_combat_action_near(timestamp)

        # Get transcript enrichment signal (if available)
        transcript_signal = None
        if self.transcript_analyzer:
            transcript_signal = self.transcript_analyzer.get_enrichment_signal(timestamp)

        # Rule 2: Multi-modal combat detection (HIGHEST CONFIDENCE)
        # Both CLIP sees combat AND transcript has combat keywords
        if has_combat_visual and transcript_signal and transcript_signal['combat_detected']:
            return True, f"multimodal_combat:{combat_action}"

        # Rule 3: CLIP combat + high speech density (team coordination)
        if has_combat_visual and transcript_signal and transcript_signal['is_high_density']:
            return True, f"visual_combat_coordinated:{combat_action}"

        # Rule 4: Strong visual signal alone
        if has_combat_visual:
            # Skip if in silence (likely false positive or old footage)
            if transcript_signal and transcript_signal['in_silence']:
                return False, "visual_but_silence"
            return True, f"visual_combat:{combat_action}"

        # Rule 5: Transcript combat without visual (audio-only engagement)
        if transcript_signal and transcript_signal['combat_detected'] and transcript_signal['combat_confidence'] > 0.5:
            return True, "audio_combat_confirmed"

        # Rule 6: High speech density (important moment, team coordination)
        if transcript_signal and transcript_signal['is_high_density'] and time_since_last >= 10:
            return True, "high_activity"

        # Rule 7: Maximum interval exceeded (catch slow changes)
        if time_since_last >= self.max_interval:
            # Skip if in extended silence
            if transcript_signal and transcript_signal['in_silence']:
                if time_since_last >= self.max_interval * 2:
                    return True, "max_interval_silence"
                return False, "silence_period"
            # Also skip old gunshot-based silence detection for backward compatibility
            if self.in_silence_period(timestamp):
                if time_since_last >= self.max_interval * 2:
                    return True, "max_interval_silence"
                return False, "silence_period"
            return True, "max_interval"

        # Default: skip this frame
        return False, "no_triggers"

    def generate_sampling_schedule(self,
                                   video_duration: float,
                                   base_interval: float = 2.0) -> List[Dict]:
        """
        Generate complete sampling schedule for a video.

        Args:
            video_duration: Total video duration in seconds
            base_interval: Base sampling interval for checking triggers

        Returns:
            List of extraction points with metadata
        """
        schedule = []
        last_extraction = -self.max_interval  # Force extraction at start

        # Generate candidate timestamps
        candidates = []
        t = 0
        while t <= video_duration:
            candidates.append(t)
            t += base_interval

        # Evaluate each candidate
        for timestamp in candidates:
            should_extract, reason = self.should_extract(
                timestamp,
                last_extraction,
                previous_values=None  # Could track this for state changes
            )

            if should_extract:
                schedule.append({
                    'timestamp': timestamp,
                    'reason': reason,
                    'time_since_last': timestamp - last_extraction
                })
                last_extraction = timestamp

        return schedule

    def get_sampling_stats(self, schedule: List[Dict],
                          video_duration: float) -> Dict:
        """
        Calculate statistics about sampling efficiency.

        Args:
            schedule: Sampling schedule from generate_sampling_schedule
            video_duration: Total video duration

        Returns:
            Statistics dictionary
        """
        total_samples = len(schedule)

        # Baseline: every 2 seconds
        baseline_samples = int(video_duration / 2.0)

        # Count by reason
        reason_counts = {}
        for point in schedule:
            reason = point['reason']
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Calculate intervals
        intervals = [point['time_since_last'] for point in schedule if 'time_since_last' in point]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0

        return {
            'total_samples': total_samples,
            'baseline_samples': baseline_samples,
            'reduction_percentage': ((baseline_samples - total_samples) / baseline_samples * 100) if baseline_samples > 0 else 0,
            'avg_interval': avg_interval,
            'min_interval': min(intervals) if intervals else 0,
            'max_interval': max(intervals) if intervals else 0,
            'reason_counts': reason_counts,
            'estimated_cost_baseline': baseline_samples * 0.01,
            'estimated_cost_intelligent': total_samples * 0.01,
            'cost_savings_percentage': ((baseline_samples - total_samples) / baseline_samples * 100) if baseline_samples > 0 else 0
        }


def test_sampling_schedule(audio_file: Optional[str] = None,
                           clip_file: Optional[str] = None,
                           transcript_file: Optional[str] = None,
                           video_duration: float = 600.0,
                           output_file: Optional[str] = None):
    """
    Test intelligent sampling on analysis results.

    Args:
        audio_file: Path to audio analysis JSON (DEPRECATED)
        clip_file: Path to CLIP actions JSON
        transcript_file: Path to transcript JSON (with chunks)
        video_duration: Video duration in seconds
        output_file: Optional path to save schedule
    """
    print("=" * 70)
    print("TRANSCRIPT-ENHANCED INTELLIGENT SAMPLING TEST")
    print("=" * 70)

    # Load sampler
    sampler = IntelligentSampler.from_files(
        audio_file=audio_file,
        clip_file=clip_file,
        transcript_file=transcript_file
    )

    # Generate schedule
    print(f"\nGenerating sampling schedule for {video_duration}s video...")
    schedule = sampler.generate_sampling_schedule(video_duration)

    # Get stats
    stats = sampler.get_sampling_stats(schedule, video_duration)

    # Display results
    print(f"\n{'='*70}")
    print("SAMPLING STATISTICS")
    print(f"{'='*70}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Baseline samples (2s interval): {stats['baseline_samples']}")
    print(f"Reduction: {stats['reduction_percentage']:.1f}%")
    print(f"\nAverage interval: {stats['avg_interval']:.1f}s")
    print(f"Min interval: {stats['min_interval']:.1f}s")
    print(f"Max interval: {stats['max_interval']:.1f}s")

    print(f"\n{'='*70}")
    print("COST ANALYSIS")
    print(f"{'='*70}")
    print(f"Baseline cost (every 2s): ${stats['estimated_cost_baseline']:.2f}")
    print(f"Intelligent cost: ${stats['estimated_cost_intelligent']:.2f}")
    print(f"Savings: ${stats['estimated_cost_baseline'] - stats['estimated_cost_intelligent']:.2f} ({stats['cost_savings_percentage']:.1f}%)")

    print(f"\n{'='*70}")
    print("TRIGGER REASONS")
    print(f"{'='*70}")
    for reason, count in sorted(stats['reason_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"{reason:30s}: {count:3d} ({count/stats['total_samples']*100:5.1f}%)")

    # Save schedule
    if output_file:
        output_data = {
            'video_duration': video_duration,
            'schedule': schedule,
            'statistics': stats
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Schedule saved to: {output_file}")

    return schedule, stats


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 4:
        clip_file = sys.argv[1]
        transcript_file = sys.argv[2]
        duration = float(sys.argv[3])
        output = sys.argv[4] if len(sys.argv) > 4 else None
        test_sampling_schedule(
            clip_file=clip_file,
            transcript_file=transcript_file,
            video_duration=duration,
            output_file=output
        )
    else:
        print("Usage: python intelligent_sampler.py <clip.json> <transcript.json> <duration> [output.json]")
