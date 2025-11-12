"""
Unit tests for IntelligentSampler module.

Tests intelligent frame sampling logic for GPT-4V extraction.
"""

import pytest
from src.core.utils.intelligent_sampler import IntelligentSampler


@pytest.mark.unit
class TestIntelligentSampler:
    """Test suite for IntelligentSampler class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        sampler = IntelligentSampler()

        assert sampler.min_interval == 3.0
        assert sampler.max_interval == 15.0
        assert sampler.combat_window == 3.0

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        sampler = IntelligentSampler(
            min_interval=5.0,
            max_interval=20.0,
            combat_window=5.0
        )

        assert sampler.min_interval == 5.0
        assert sampler.max_interval == 20.0
        assert sampler.combat_window == 5.0

    def test_init_with_clip_data(self, sample_actions_data):
        """Test initialization with CLIP data."""
        sampler = IntelligentSampler(clip_data=sample_actions_data)

        assert sampler.clip_data == sample_actions_data

    def test_should_extract_min_interval(self):
        """Test that minimum interval is enforced."""
        sampler = IntelligentSampler(min_interval=10.0)

        # Try to extract 5s after last extraction (< min_interval)
        should_extract, reason = sampler.should_extract(
            timestamp=5.0,
            last_extraction_time=0.0
        )

        assert should_extract is False
        assert reason == "too_soon"

    def test_should_extract_max_interval(self):
        """Test that max interval triggers extraction."""
        sampler = IntelligentSampler(
            min_interval=3.0,
            max_interval=15.0
        )

        # 20s since last extraction (> max_interval)
        should_extract, reason = sampler.should_extract(
            timestamp=20.0,
            last_extraction_time=0.0
        )

        assert should_extract is True
        assert "max_interval" in reason

    def test_has_combat_action_near(self):
        """Test combat action detection near timestamp."""
        clip_data = {
            'results': [
                {
                    'timestamp': 10.0,
                    'predictions': [
                        {'label': 'player in active combat shooting', 'probability': 0.9}
                    ]
                }
            ]
        }

        sampler = IntelligentSampler(
            clip_data=clip_data,
            combat_window=5.0
        )

        # Within combat window
        has_combat, action = sampler.has_combat_action_near(12.0)
        assert has_combat is True
        assert action == 'player in active combat shooting'

        # Outside combat window
        has_combat, action = sampler.has_combat_action_near(20.0)
        assert has_combat is False
        assert action is None

    def test_has_looting_action_near(self):
        """Test looting action detection near timestamp."""
        clip_data = {
            'results': [
                {
                    'timestamp': 10.0,
                    'predictions': [
                        {'label': 'player looting items', 'probability': 0.9}
                    ]
                }
            ]
        }

        sampler = IntelligentSampler(
            clip_data=clip_data,
            combat_window=5.0
        )

        # Within window
        has_looting = sampler.has_looting_action_near(12.0)
        assert has_looting is True

        # Outside window
        has_looting = sampler.has_looting_action_near(20.0)
        assert has_looting is False

    def test_should_extract_skip_looting(self):
        """Test that looting actions are skipped."""
        clip_data = {
            'results': [
                {
                    'timestamp': 10.0,
                    'predictions': [
                        {'label': 'player looting items', 'probability': 0.9}
                    ]
                }
            ]
        }

        sampler = IntelligentSampler(
            clip_data=clip_data,
            min_interval=3.0,
            combat_window=5.0
        )

        should_extract, reason = sampler.should_extract(
            timestamp=10.0,
            last_extraction_time=0.0
        )

        assert should_extract is False
        assert reason == "looting_action"

    def test_generate_sampling_schedule(self):
        """Test generation of complete sampling schedule."""
        sampler = IntelligentSampler(
            min_interval=5.0,
            max_interval=20.0
        )

        schedule = sampler.generate_sampling_schedule(
            video_duration=100.0,
            base_interval=2.0
        )

        assert isinstance(schedule, list)
        assert len(schedule) > 0
        assert all('timestamp' in point for point in schedule)
        assert all('reason' in point for point in schedule)

        # Check intervals are respected
        for i in range(1, len(schedule)):
            interval = schedule[i]['timestamp'] - schedule[i-1]['timestamp']
            assert interval >= sampler.min_interval

    def test_get_sampling_stats(self):
        """Test calculation of sampling statistics."""
        sampler = IntelligentSampler()

        schedule = [
            {'timestamp': 0.0, 'reason': 'max_interval', 'time_since_last': 20.0},
            {'timestamp': 20.0, 'reason': 'combat', 'time_since_last': 20.0},
            {'timestamp': 40.0, 'reason': 'max_interval', 'time_since_last': 20.0},
        ]

        stats = sampler.get_sampling_stats(schedule, video_duration=100.0)

        assert stats['total_samples'] == 3
        assert stats['baseline_samples'] == 50  # 100s / 2s
        assert stats['reduction_percentage'] > 0
        assert 'reason_counts' in stats
        assert stats['reason_counts']['max_interval'] == 2
        assert stats['reason_counts']['combat'] == 1

    def test_from_files_missing_files(self):
        """Test loading sampler from non-existent files."""
        sampler = IntelligentSampler.from_files(
            clip_file="nonexistent.json",
            transcript_file="nonexistent.json"
        )

        assert sampler is not None
        assert sampler.clip_data == {}

    def test_from_files_with_clip_data(self, sample_actions_json):
        """Test loading sampler from real CLIP file."""
        sampler = IntelligentSampler.from_files(
            clip_file=str(sample_actions_json)
        )

        assert sampler is not None
        assert sampler.clip_data is not None
        assert 'results' in sampler.clip_data

    def test_combat_action_keywords(self):
        """Test that combat action keywords are properly defined."""
        sampler = IntelligentSampler()

        assert len(sampler.combat_actions) > 0
        assert 'combat' in sampler.combat_actions or 'shooting' in sampler.combat_actions

    def test_looting_action_keywords(self):
        """Test that looting action keywords are properly defined."""
        sampler = IntelligentSampler()

        assert len(sampler.looting_actions) > 0
        assert 'looting' in sampler.looting_actions or 'inventory' in sampler.looting_actions

    def test_empty_schedule_short_video(self):
        """Test schedule generation for very short video."""
        sampler = IntelligentSampler(
            min_interval=10.0,
            max_interval=20.0
        )

        schedule = sampler.generate_sampling_schedule(
            video_duration=5.0,  # Very short
            base_interval=2.0
        )

        # Should have at least one sample
        assert len(schedule) >= 1

    def test_transcript_analyzer_optional(self):
        """Test that transcript analyzer is optional."""
        sampler = IntelligentSampler(transcript_analyzer=None)

        # Should work without transcript analyzer
        should_extract, reason = sampler.should_extract(
            timestamp=20.0,
            last_extraction_time=0.0
        )

        assert isinstance(should_extract, bool)
        assert isinstance(reason, str)
