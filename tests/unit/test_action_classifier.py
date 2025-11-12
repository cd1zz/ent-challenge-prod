"""
Unit tests for ActionClassifier module.

Tests CLIP-based action classification functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.core.utils.action_classifier import ActionClassifier


@pytest.mark.unit
class TestActionClassifier:
    """Test suite for ActionClassifier class."""

    @pytest.fixture
    def mock_clip_model(self):
        """Create a mock CLIP model."""
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        return mock_model, mock_preprocess

    @patch('src.core.utils.action_classifier.clip.load')
    def test_init_default_labels(self, mock_clip_load, mock_clip_model):
        """Test initialization with default labels."""
        mock_clip_load.return_value = mock_clip_model

        classifier = ActionClassifier(labels=None)

        assert classifier is not None
        assert classifier.num_labels > 0
        assert classifier.labels == ActionClassifier.DEFAULT_LABELS

    @patch('src.core.utils.action_classifier.clip.load')
    def test_init_custom_labels(self, mock_clip_load, mock_clip_model, sample_clip_labels):
        """Test initialization with custom labels."""
        mock_clip_load.return_value = mock_clip_model

        classifier = ActionClassifier(labels=sample_clip_labels)

        assert classifier.labels == sample_clip_labels
        assert classifier.num_labels == len(sample_clip_labels)

    @patch('src.core.utils.action_classifier.clip.load')
    @patch.dict('os.environ', {'CLIP_LABELS': 'label1|label2|label3'})
    def test_init_env_labels(self, mock_clip_load, mock_clip_model):
        """Test initialization with labels from environment."""
        mock_clip_load.return_value = mock_clip_model

        classifier = ActionClassifier(labels=None)

        assert classifier.labels == ['label1', 'label2', 'label3']
        assert classifier.num_labels == 3

    @patch('src.core.utils.action_classifier.clip.load')
    def test_device_selection_cpu(self, mock_clip_load, mock_clip_model):
        """Test device selection when GPU not available."""
        mock_clip_load.return_value = mock_clip_model

        with patch('torch.cuda.is_available', return_value=False):
            classifier = ActionClassifier()

        assert classifier.device == "cpu"

    @pytest.mark.gpu
    @pytest.mark.skip(reason="Requires actual CUDA-enabled PyTorch")
    @patch('src.core.utils.action_classifier.clip.load')
    def test_device_selection_cuda(self, mock_clip_load, mock_clip_model):
        """Test device selection when GPU is available."""
        mock_clip_load.return_value = mock_clip_model

        with patch('torch.cuda.is_available', return_value=True):
            classifier = ActionClassifier()

        assert classifier.device == "cuda"

    @patch('src.core.utils.action_classifier.clip.load')
    def test_get_primary_action(self, mock_clip_load, mock_clip_model):
        """Test getting primary action from predictions."""
        mock_clip_load.return_value = mock_clip_model
        classifier = ActionClassifier()

        predictions = {
            "player running": 0.85,
            "player jumping": 0.10,
            "player shooting": 0.05
        }

        primary = classifier.get_primary_action(predictions)

        assert primary == "player running"

    @patch('src.core.utils.action_classifier.clip.load')
    def test_get_primary_action_empty(self, mock_clip_load, mock_clip_model):
        """Test getting primary action with empty predictions."""
        mock_clip_load.return_value = mock_clip_model
        classifier = ActionClassifier()

        predictions = {}

        primary = classifier.get_primary_action(predictions)

        # Should return "uncertain" for empty predictions
        assert primary == "uncertain"


@pytest.mark.unit
class TestActionClassifierIntegration:
    """Integration-like tests for ActionClassifier (mocked external deps)."""

    @pytest.mark.skip(reason="Requires actual CLIP model loading")
    def test_classify_frame_real(self, sample_pil_image):
        """Test frame classification with real CLIP model (slow)."""
        classifier = ActionClassifier()
        predictions = classifier.classify_frame(sample_pil_image)

        assert isinstance(predictions, dict)
        assert len(predictions) == classifier.num_labels
        assert all(0 <= p <= 1 for p in predictions.values())
        assert abs(sum(predictions.values()) - 1.0) < 0.01  # Should sum to ~1

    @pytest.mark.skip(reason="Requires actual CLIP model loading")
    def test_classify_batch_real(self, sample_frame):
        """Test batch classification with real CLIP model (slow)."""
        classifier = ActionClassifier()

        # Create batch of 4 identical frames
        frames = [sample_frame for _ in range(4)]

        predictions_batch = classifier.classify_batch(frames)

        assert len(predictions_batch) == 4
        assert all(isinstance(p, dict) for p in predictions_batch)
        assert all(len(p) == classifier.num_labels for p in predictions_batch)
