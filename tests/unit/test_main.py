"""
Unit tests for main.py CLI entry point.

Tests command-line argument parsing and routing.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import main


@pytest.mark.unit
class TestMainCLI:
    """Test suite for main.py CLI functionality."""

    def test_get_output_path_explicit(self):
        """Test get_output_path with explicit output."""
        result = main.get_output_path(
            input_path="video.mkv",
            suffix="_actions.json",
            explicit_output="custom/path.json"
        )

        assert result == "custom/path.json"

    def test_get_output_path_default(self, tmp_path, monkeypatch):
        """Test get_output_path with default output directory."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        result = main.get_output_path(
            input_path="video.mkv",
            suffix="_actions.json",
            explicit_output=None
        )

        # Check that output is in output directory and contains base name
        assert result.startswith("output")
        assert "video_actions" in result
        assert result.endswith(".json")
        assert Path("output").exists()

    def test_get_output_path_suffix_variations(self, tmp_path, monkeypatch):
        """Test various suffix patterns."""
        monkeypatch.chdir(tmp_path)

        # Test different suffixes (now with timestamps)
        result1 = main.get_output_path("test.mkv", "_hud.json")
        assert "test_hud" in result1
        assert result1.endswith(".json")

        result2 = main.get_output_path("test.mkv", "_games.json")
        assert "test_games" in result2
        assert result2.endswith(".json")

        result3 = main.get_output_path("actions.json", "_games.json")
        assert "actions_games" in result3
        assert result3.endswith(".json")

    def test_ensure_output_dir(self, tmp_path):
        """Test ensure_output_dir creates necessary directories."""
        output_path = tmp_path / "nested" / "dir" / "output.json"

        main.ensure_output_dir(str(output_path))

        assert output_path.parent.exists()

    def test_main_no_args(self):
        """Test main with no arguments shows help."""
        with patch('sys.argv', ['main.py']):
            exit_code = main.main()

            # Argparse returns 1 when no subcommand is provided
            assert exit_code == 1

    def test_main_help(self):
        """Test main with --help flag."""
        with patch('sys.argv', ['main.py', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main.main()

            # --help exits with 0
            assert exc_info.value.code == 0

    @pytest.mark.parametrize("command", [
        "detect-actions",
        "detect-audio-events",
        "detect-games",
        "extract-hud",
        "transcribe",
        "detect-gameover",
        "setup-regions"
    ])
    def test_command_help(self, command):
        """Test that all commands have help."""
        with patch('sys.argv', ['main.py', command, '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main.main()

            # --help exits with 0
            assert exc_info.value.code == 0


@pytest.mark.unit
class TestDetectActionsCommand:
    """Test suite for detect-actions command."""

    @patch('main.cmd_detect_actions')
    def test_detect_actions_basic(self, mock_cmd):
        """Test basic detect-actions command."""
        mock_cmd.return_value = 0

        with patch('sys.argv', ['main.py', 'detect-actions', 'video.mkv']):
            exit_code = main.main()

        assert exit_code == 0
        mock_cmd.assert_called_once()

    @patch('main.cmd_detect_actions')
    def test_detect_actions_with_labels(self, mock_cmd):
        """Test detect-actions with custom labels."""
        mock_cmd.return_value = 0

        with patch('sys.argv', [
            'main.py', 'detect-actions', 'video.mkv',
            '--labels', 'label1|label2|label3'
        ]):
            exit_code = main.main()

        assert exit_code == 0
        args = mock_cmd.call_args[0][0]
        assert args.labels == 'label1|label2|label3'

    @patch('main.cmd_detect_actions')
    def test_detect_actions_with_add_labels(self, mock_cmd):
        """Test detect-actions with additional labels."""
        mock_cmd.return_value = 0

        with patch('sys.argv', [
            'main.py', 'detect-actions', 'video.mkv',
            '--add-labels', 'extra1|extra2'
        ]):
            exit_code = main.main()

        assert exit_code == 0
        args = mock_cmd.call_args[0][0]
        assert args.add_labels == 'extra1|extra2'

    @patch('main.cmd_detect_actions')
    def test_detect_actions_with_interval(self, mock_cmd):
        """Test detect-actions with custom interval."""
        mock_cmd.return_value = 0

        with patch('sys.argv', [
            'main.py', 'detect-actions', 'video.mkv',
            '--interval', '1.5'
        ]):
            exit_code = main.main()

        assert exit_code == 0
        args = mock_cmd.call_args[0][0]
        assert args.interval == 1.5


@pytest.mark.unit
class TestDetectGamesCommand:
    """Test suite for detect-games command."""

    @patch('main.cmd_detect_games')
    def test_detect_games_basic(self, mock_cmd):
        """Test basic detect-games command."""
        mock_cmd.return_value = 0

        with patch('sys.argv', ['main.py', 'detect-games', 'actions.json']):
            exit_code = main.main()

        assert exit_code == 0
        mock_cmd.assert_called_once()

    @patch('main.cmd_detect_games')
    def test_detect_games_with_thresholds(self, mock_cmd):
        """Test detect-games with custom thresholds."""
        mock_cmd.return_value = 0

        with patch('sys.argv', [
            'main.py', 'detect-games', 'actions.json',
            '--parachute-thresh', '0.90',
            '--death-thresh', '0.70',
            '--min-duration', '120.0'
        ]):
            exit_code = main.main()

        assert exit_code == 0
        args = mock_cmd.call_args[0][0]
        assert args.parachute_thresh == 0.90
        assert args.death_thresh == 0.70
        assert args.min_duration == 120.0


@pytest.mark.unit
class TestExtractHudCommand:
    """Test suite for extract-hud command."""

    @patch('main.cmd_extract_hud')
    def test_extract_hud_basic(self, mock_cmd):
        """Test basic extract-hud command."""
        mock_cmd.return_value = 0

        with patch('sys.argv', [
            'main.py', 'extract-hud', 'video.mkv',
            '--method', 'gpt4v',
            '--clip', 'actions.json'
        ]):
            exit_code = main.main()

        assert exit_code == 0
        mock_cmd.assert_called_once()

    @patch('main.cmd_extract_hud')
    def test_extract_hud_method_selection(self, mock_cmd):
        """Test extract-hud with different methods."""
        mock_cmd.return_value = 0

        for method in ['gpt4v', 'paddleocr', 'ocr']:
            with patch('sys.argv', [
                'main.py', 'extract-hud', 'video.mkv',
                '--method', method
            ]):
                exit_code = main.main()
                args = mock_cmd.call_args[0][0]
                assert args.method == method


@pytest.mark.unit
class TestTranscribeCommand:
    """Test suite for transcribe command."""

    @patch('main.cmd_transcribe')
    def test_transcribe_basic(self, mock_cmd):
        """Test basic transcribe command."""
        mock_cmd.return_value = 0

        with patch('sys.argv', ['main.py', 'transcribe', 'audio.wav']):
            exit_code = main.main()

        assert exit_code == 0
        mock_cmd.assert_called_once()

    @patch('main.cmd_transcribe')
    def test_transcribe_format_selection(self, mock_cmd):
        """Test transcribe with format selection."""
        mock_cmd.return_value = 0

        with patch('sys.argv', [
            'main.py', 'transcribe', 'audio.wav',
            '--format', 'text'
        ]):
            exit_code = main.main()

        assert exit_code == 0
        args = mock_cmd.call_args[0][0]
        assert args.format == 'text'
