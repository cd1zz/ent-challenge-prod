"""
Integration tests for complete workflows.

Tests end-to-end functionality across multiple modules.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.mark.integration
class TestDetectActionsToGamesWorkflow:
    """Test the complete workflow from actions to games detection."""

    def test_actions_to_games_workflow(self, sample_actions_json, temp_output_dir):
        """Test complete workflow: actions JSON -> games JSON."""
        from src.core.game_detection import GameSegmenter

        # Load actions
        with open(sample_actions_json, 'r') as f:
            actions_data = json.load(f)

        # Segment games
        segmenter = GameSegmenter(min_game_duration=1.0)
        games = segmenter.segment_games(actions_data['results'])

        # Verify results
        assert isinstance(games, list)

        # Save games
        games_output = temp_output_dir / "games.json"
        output_data = {
            'video_path': actions_data['video_path'],
            'games': games
        }
        with open(games_output, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Verify output file
        assert games_output.exists()
        with open(games_output, 'r') as f:
            loaded = json.load(f)
            assert 'games' in loaded


@pytest.mark.integration
class TestOutputDirectoryCreation:
    """Test that output directories are created correctly."""

    def test_default_output_directory(self, tmp_path, monkeypatch):
        """Test that default output directory is created."""
        import main

        monkeypatch.chdir(tmp_path)

        # Get output path (should create directory)
        output_path = main.get_output_path("video.mkv", "_test.json")

        # Verify directory exists
        assert Path("output").exists()
        assert Path("output").is_dir()

    def test_nested_output_directory(self, tmp_path):
        """Test that nested output directories are created."""
        import main

        output_path = tmp_path / "nested" / "deep" / "output.json"
        main.ensure_output_dir(str(output_path))

        assert output_path.parent.exists()
        assert output_path.parent.is_dir()


@pytest.mark.integration
@pytest.mark.skip(reason="Requires actual video file")
class TestEndToEndPipeline:
    """End-to-end pipeline tests (requires real files)."""

    def test_complete_pipeline(self, temp_output_dir):
        """
        Test complete pipeline:
        1. detect-actions (CLIP classification)
        2. detect-games (segmentation)
        3. extract-hud (HUD extraction)
        """
        # This test would require:
        # - Real video file
        # - API keys configured
        # - Sufficient processing time
        pass

    def test_actions_then_games(self, temp_output_dir):
        """
        Test two-step workflow:
        1. Run detect-actions on video
        2. Run detect-games on actions output
        """
        # This test would require real video
        pass


@pytest.mark.integration
class TestFileFormatCompatibility:
    """Test compatibility between different file formats."""

    def test_actions_json_format(self, sample_actions_data, temp_output_dir):
        """Test that actions JSON has correct format."""
        # Required fields
        assert 'video_path' in sample_actions_data
        assert 'results' in sample_actions_data
        assert 'labels' in sample_actions_data

        # Results structure
        for result in sample_actions_data['results']:
            assert 'timestamp' in result
            assert 'primary_action' in result
            assert 'all_predictions' in result

    def test_games_json_format(self, sample_games_data):
        """Test that games JSON has correct format."""
        assert 'games' in sample_games_data
        assert 'summary' in sample_games_data

        for game in sample_games_data['games']:
            assert 'game_number' in game
            assert 'start_time' in game
            assert 'end_time' in game
            assert 'duration' in game

    def test_transcript_json_format(self, sample_transcript_data):
        """Test that transcript JSON has correct format."""
        assert 'chunks' in sample_transcript_data
        assert 'duration' in sample_transcript_data

        for chunk in sample_transcript_data['chunks']:
            assert 'chunk_number' in chunk
            assert 'start_time' in chunk
            assert 'end_time' in chunk
