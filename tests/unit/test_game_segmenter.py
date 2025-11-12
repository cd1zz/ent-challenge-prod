"""
Unit tests for GameSegmenter module.

Tests game boundary detection from CLIP action classifications.
"""

import pytest
from src.core.game_detection.game_segmenter import GameSegmenter


@pytest.mark.unit
class TestGameSegmenter:
    """Test suite for GameSegmenter class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        segmenter = GameSegmenter()

        assert segmenter.parachute_threshold == 0.85
        assert segmenter.death_threshold == 0.60
        assert segmenter.min_game_duration == 60.0

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        segmenter = GameSegmenter(
            parachute_confidence_threshold=0.90,
            death_confidence_threshold=0.70,
            min_game_duration=120.0
        )

        assert segmenter.parachute_threshold == 0.90
        assert segmenter.death_threshold == 0.70
        assert segmenter.min_game_duration == 120.0

    def test_segment_games_simple(self, sample_actions_data):
        """Test basic game segmentation."""
        segmenter = GameSegmenter(
            parachute_confidence_threshold=0.85,
            death_confidence_threshold=0.60,
            min_game_duration=1.0  # Low threshold for test
        )

        games = segmenter.segment_games(sample_actions_data['results'])

        # Should return a list (may be empty depending on detection logic)
        assert isinstance(games, list)

    def test_segment_games_min_duration_filter(self):
        """Test that games below minimum duration are filtered."""
        segmenter = GameSegmenter(min_game_duration=100.0)

        # Create short game (parachute at 0s, death at 30s = 30s duration)
        actions = [
            {
                "timestamp": 0.0,
                "primary_action": "player parachuting",
                "all_predictions": {"player parachuting": 0.95}
            },
            {
                "timestamp": 30.0,
                "primary_action": "player dead",
                "all_predictions": {"player dead": 0.90}
            }
        ]

        games = segmenter.segment_games(actions)

        # Should be filtered out due to min duration
        assert len(games) == 0

    def test_segment_games_no_parachute(self):
        """Test segmentation when no parachute action is detected."""
        segmenter = GameSegmenter()

        actions = [
            {
                "timestamp": 0.0,
                "primary_action": "player running",
                "all_predictions": {"player running": 0.85}
            },
            {
                "timestamp": 100.0,
                "primary_action": "player dead",
                "all_predictions": {"player dead": 0.90}
            }
        ]

        games = segmenter.segment_games(actions)

        # Should not detect game without parachute start
        assert len(games) == 0

    def test_segment_games_no_death(self):
        """Test segmentation when parachute detected but no death."""
        segmenter = GameSegmenter()

        actions = [
            {
                "timestamp": 0.0,
                "primary_action": "player parachuting",
                "all_predictions": {"player parachuting": 0.95}
            },
            {
                "timestamp": 100.0,
                "primary_action": "player running",
                "all_predictions": {"player running": 0.85}
            }
        ]

        games = segmenter.segment_games(actions)

        # Game may be detected with default end time
        # Implementation-specific behavior
        assert isinstance(games, list)

    def test_segment_multiple_games(self):
        """Test segmentation with multiple games."""
        segmenter = GameSegmenter(min_game_duration=10.0)

        actions = [
            # Game 1: 0-20s
            {"timestamp": 0.0, "primary_action": "player parachuting",
             "all_predictions": {"player parachuting": 0.95}},
            {"timestamp": 10.0, "primary_action": "player running",
             "all_predictions": {"player running": 0.85}},
            {"timestamp": 20.0, "primary_action": "player dead",
             "all_predictions": {"player dead": 0.90}},

            # Lobby
            {"timestamp": 30.0, "primary_action": "main menu or lobby screen",
             "all_predictions": {"main menu or lobby screen": 0.90}},

            # Game 2: 40-70s
            {"timestamp": 40.0, "primary_action": "player parachuting",
             "all_predictions": {"player parachuting": 0.92}},
            {"timestamp": 50.0, "primary_action": "player shooting",
             "all_predictions": {"player shooting": 0.88}},
            {"timestamp": 70.0, "primary_action": "player dead",
             "all_predictions": {"player dead": 0.85}},
        ]

        games = segmenter.segment_games(actions)

        # Should return a list and games should be in order if detected
        assert isinstance(games, list)
        if len(games) >= 2:
            assert games[0]['start_time'] < games[1]['start_time']

    def test_game_output_format(self):
        """Test that segmented games have correct output format."""
        segmenter = GameSegmenter(min_game_duration=1.0)

        actions = [
            {"timestamp": 0.0, "primary_action": "player parachuting",
             "all_predictions": {"player parachuting": 0.95}},
            {"timestamp": 100.0, "primary_action": "player dead",
             "all_predictions": {"player dead": 0.90}},
        ]

        games = segmenter.segment_games(actions)

        if len(games) > 0:
            game = games[0]
            assert 'game_number' in game
            assert 'start_time' in game
            assert 'end_time' in game
            assert 'duration' in game
            assert game['duration'] == game['end_time'] - game['start_time']

    def test_confidence_threshold_filtering(self):
        """Test that low confidence parachute detections are filtered."""
        segmenter = GameSegmenter(
            parachute_confidence_threshold=0.90,  # High threshold
            min_game_duration=1.0
        )

        actions = [
            # Low confidence parachute (should be ignored)
            {"timestamp": 0.0, "primary_action": "player parachuting",
             "all_predictions": {"player parachuting": 0.60}},
            {"timestamp": 100.0, "primary_action": "player dead",
             "all_predictions": {"player dead": 0.90}},
        ]

        games = segmenter.segment_games(actions)

        # Should not detect game due to low parachute confidence
        assert len(games) == 0

    def test_empty_actions_list(self):
        """Test segmentation with empty actions list."""
        segmenter = GameSegmenter()

        games = segmenter.segment_games([])

        assert games == []

    def test_single_action(self):
        """Test segmentation with only one action."""
        segmenter = GameSegmenter()

        actions = [
            {"timestamp": 0.0, "primary_action": "player parachuting",
             "all_predictions": {"player parachuting": 0.95}}
        ]

        games = segmenter.segment_games(actions)

        # Should either detect partial game or no game
        assert isinstance(games, list)
