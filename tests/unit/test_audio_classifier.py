"""
Unit tests for AudioClassifier module.

Tests CLAP-based audio event classification functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.core.utils.audio_classifier import AudioClassifier


@pytest.mark.unit
class TestAudioClassifier:
    """Test suite for AudioClassifier class."""

    @pytest.fixture
    def mock_clap_module(self):
        """Create a mock CLAP module."""
        mock_module = MagicMock()
        mock_module.load_ckpt.return_value = None
        return mock_module

    @patch('src.core.utils.audio_classifier.laion_clap.CLAP_Module')
    def test_init_default_labels(self, mock_clap_class, mock_clap_module):
        """Test initialization with default labels."""
        mock_clap_class.return_value = mock_clap_module

        classifier = AudioClassifier(labels=None)

        assert classifier is not None
        assert classifier.num_labels > 0
        assert classifier.labels == AudioClassifier.DEFAULT_LABELS

    @patch('src.core.utils.audio_classifier.laion_clap.CLAP_Module')
    def test_init_custom_labels(self, mock_clap_class, mock_clap_module):
        """Test initialization with custom labels."""
        mock_clap_class.return_value = mock_clap_module
        custom_labels = ["gunshots", "footsteps", "explosions"]

        classifier = AudioClassifier(labels=custom_labels)

        assert classifier.labels == custom_labels
        assert classifier.num_labels == len(custom_labels)

    @patch('src.core.utils.audio_classifier.laion_clap.CLAP_Module')
    @patch.dict('os.environ', {'CLAP_LABELS': 'label1|label2|label3'})
    def test_init_env_labels(self, mock_clap_class, mock_clap_module):
        """Test initialization with labels from environment."""
        mock_clap_class.return_value = mock_clap_module

        classifier = AudioClassifier(labels=None)

        assert classifier.labels == ['label1', 'label2', 'label3']
        assert classifier.num_labels == 3

    @patch('src.core.utils.audio_classifier.laion_clap.CLAP_Module')
    def test_device_selection_cpu(self, mock_clap_class, mock_clap_module):
        """Test device selection when GPU not available."""
        mock_clap_class.return_value = mock_clap_module

        with patch('torch.cuda.is_available', return_value=False):
            classifier = AudioClassifier()

        assert classifier.device == "cpu"

    @pytest.mark.gpu
    @pytest.mark.skip(reason="Requires actual CUDA-enabled PyTorch")
    @patch('src.core.utils.audio_classifier.laion_clap.CLAP_Module')
    def test_device_selection_cuda(self, mock_clap_class, mock_clap_module):
        """Test device selection when GPU is available."""
        mock_clap_class.return_value = mock_clap_module

        with patch('torch.cuda.is_available', return_value=True):
            classifier = AudioClassifier()

        assert classifier.device == "cuda"

    @patch('src.core.utils.audio_classifier.laion_clap.CLAP_Module')
    def test_get_primary_event(self, mock_clap_class, mock_clap_module):
        """Test getting primary event from predictions."""
        mock_clap_class.return_value = mock_clap_module
        classifier = AudioClassifier()

        predictions = {
            "gunshots and weapon fire": 0.85,
            "footsteps on various surfaces": 0.10,
            "explosions and blast sounds": 0.05
        }

        primary = classifier.get_primary_event(predictions)

        assert primary == "gunshots and weapon fire"

    @patch('src.core.utils.audio_classifier.laion_clap.CLAP_Module')
    def test_get_primary_event_empty(self, mock_clap_class, mock_clap_module):
        """Test getting primary event with empty predictions."""
        mock_clap_class.return_value = mock_clap_module
        classifier = AudioClassifier()

        predictions = {}

        primary = classifier.get_primary_event(predictions)

        # Should return "uncertain" for empty predictions
        assert primary == "uncertain"

    @patch('src.core.utils.audio_classifier.laion_clap.CLAP_Module')
    @patch('subprocess.run')
    def test_extract_audio_segment(self, mock_subprocess, mock_clap_class, mock_clap_module, tmp_path):
        """Test audio segment extraction using ffmpeg."""
        mock_clap_class.return_value = mock_clap_module
        mock_subprocess.return_value = Mock(returncode=0, stdout=b"", stderr=b"")

        classifier = AudioClassifier()

        video_path = str(tmp_path / "test_video.mkv")
        output_path = classifier.extract_audio_segment(
            video_path,
            start_time=10.0,
            duration=2.0
        )

        # Verify ffmpeg was called
        assert mock_subprocess.called

        # Verify output path is in temp directory
        assert "audio_segment" in output_path
        assert output_path.endswith(".wav")

    @patch('src.core.utils.audio_classifier.laion_clap.CLAP_Module')
    def test_ffmpeg_path_custom(self, mock_clap_class, mock_clap_module):
        """Test custom ffmpeg path."""
        mock_clap_class.return_value = mock_clap_module
        custom_path = "/custom/path/to/ffmpeg"

        classifier = AudioClassifier(ffmpeg_path=custom_path)

        assert classifier.ffmpeg_path == custom_path


@pytest.mark.unit
class TestAudioClassifierIntegration:
    """Integration-like tests for AudioClassifier (mocked external deps)."""

    @pytest.mark.skip(reason="Requires actual CLAP model loading")
    def test_classify_audio_segment_real(self, tmp_path):
        """Test audio segment classification with real CLAP model (slow)."""
        # This would require actual audio file and CLAP model
        classifier = AudioClassifier()

        # Create dummy audio file
        audio_path = str(tmp_path / "test_audio.wav")

        predictions = classifier.classify_audio_segment(audio_path)

        assert isinstance(predictions, dict)
        assert len(predictions) == classifier.num_labels
        assert all(0 <= p <= 1 for p in predictions.values())

    @pytest.mark.skip(reason="Requires actual video file and ffmpeg")
    @patch('src.core.utils.audio_classifier.laion_clap.CLAP_Module')
    def test_classify_video_audio_real(self, mock_clap_class, tmp_path):
        """Test video audio classification with real video (slow)."""
        # This would require actual video file
        mock_module = MagicMock()
        mock_clap_class.return_value = mock_module

        classifier = AudioClassifier()

        video_path = str(tmp_path / "test_video.mkv")

        results = classifier.classify_video_audio(
            video_path,
            interval_seconds=2.0,
            segment_duration=2.0,
            max_duration=10.0
        )

        assert isinstance(results, list)
        assert all('timestamp' in r for r in results)
        assert all('primary_event' in r for r in results)
        assert all('confidence' in r for r in results)
