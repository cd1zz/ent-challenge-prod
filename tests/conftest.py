"""
Pytest configuration and shared fixtures.

This module provides fixtures used across multiple test files.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def project_root_path() -> Path:
    """Return the project root directory path."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root_path) -> Path:
    """Return the test data directory path."""
    data_dir = project_root_path / "tests" / "fixtures"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Return a temporary output directory for tests."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a sample video frame (numpy array) for testing."""
    # Create a 1920x1080 RGB frame with some patterns
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Add some colored regions to make it more realistic
    frame[0:360, 0:640] = [255, 0, 0]  # Red region
    frame[360:720, 640:1280] = [0, 255, 0]  # Green region
    frame[720:1080, 1280:1920] = [0, 0, 255]  # Blue region

    return frame


@pytest.fixture
def sample_pil_image(sample_frame) -> Image.Image:
    """Create a sample PIL Image from the sample frame."""
    return Image.fromarray(sample_frame)


@pytest.fixture
def sample_actions_data() -> Dict[str, Any]:
    """Create sample CLIP actions data structure."""
    return {
        "video_path": "test_video.mkv",
        "timestamp": "2024-01-15T10:30:00",
        "interval_seconds": 2.0,
        "total_frames": 10,
        "labels": [
            "player running",
            "player jumping",
            "player shooting",
            "player parachuting",
            "player dead"
        ],
        "results": [
            {
                "timestamp": 0.0,
                "primary_action": "player parachuting",
                "top_3_predictions": [
                    {"action": "player parachuting", "probability": 0.92},
                    {"action": "player running", "probability": 0.05},
                    {"action": "player jumping", "probability": 0.02}
                ],
                "all_predictions": {
                    "player running": 0.05,
                    "player jumping": 0.02,
                    "player shooting": 0.01,
                    "player parachuting": 0.92,
                    "player dead": 0.00
                }
            },
            {
                "timestamp": 2.0,
                "primary_action": "player running",
                "top_3_predictions": [
                    {"action": "player running", "probability": 0.85},
                    {"action": "player jumping", "probability": 0.10},
                    {"action": "player shooting", "probability": 0.03}
                ],
                "all_predictions": {
                    "player running": 0.85,
                    "player jumping": 0.10,
                    "player shooting": 0.03,
                    "player parachuting": 0.01,
                    "player dead": 0.01
                }
            },
            {
                "timestamp": 4.0,
                "primary_action": "player shooting",
                "top_3_predictions": [
                    {"action": "player shooting", "probability": 0.78},
                    {"action": "player running", "probability": 0.15},
                    {"action": "player jumping", "probability": 0.05}
                ],
                "all_predictions": {
                    "player running": 0.15,
                    "player jumping": 0.05,
                    "player shooting": 0.78,
                    "player parachuting": 0.01,
                    "player dead": 0.01
                }
            },
            {
                "timestamp": 18.0,
                "primary_action": "player dead",
                "top_3_predictions": [
                    {"action": "player dead", "probability": 0.88},
                    {"action": "player running", "probability": 0.08},
                    {"action": "player shooting", "probability": 0.03}
                ],
                "all_predictions": {
                    "player running": 0.08,
                    "player jumping": 0.01,
                    "player shooting": 0.03,
                    "player parachuting": 0.00,
                    "player dead": 0.88
                }
            }
        ]
    }


@pytest.fixture
def sample_actions_json(test_data_dir, sample_actions_data) -> Path:
    """Create a sample actions JSON file."""
    json_path = test_data_dir / "sample_actions.json"
    with open(json_path, 'w') as f:
        json.dump(sample_actions_data, f, indent=2)
    return json_path


@pytest.fixture
def sample_games_data() -> Dict[str, Any]:
    """Create sample game segmentation data."""
    return {
        "video_path": "test_video.mkv",
        "actions_file": "sample_actions.json",
        "segmentation_params": {
            "min_game_duration": 60.0,
            "parachute_threshold": 0.85,
            "death_threshold": 0.60
        },
        "summary": {
            "total_games": 2,
            "total_duration": 600.0
        },
        "games": [
            {
                "game_number": 1,
                "start_time": 10.0,
                "end_time": 320.5,
                "duration": 310.5,
                "confidence_scores": {
                    "start_confidence": 0.92,
                    "end_confidence": 0.88
                }
            },
            {
                "game_number": 2,
                "start_time": 350.0,
                "end_time": 639.5,
                "duration": 289.5,
                "confidence_scores": {
                    "start_confidence": 0.89,
                    "end_confidence": 0.91
                }
            }
        ]
    }


@pytest.fixture
def sample_transcript_data() -> Dict[str, Any]:
    """Create sample transcript data."""
    return {
        "audio_path": "test_audio.wav",
        "duration": 600.0,
        "num_chunks": 2,
        "timestamp": "2024-01-15T10:30:00",
        "chunks": [
            {
                "chunk_number": 0,
                "start_time": 0.0,
                "end_time": 300.0,
                "speakers": [
                    {
                        "speaker_id": "speaker_0",
                        "segments": [
                            {
                                "start": 5.2,
                                "end": 8.1,
                                "text": "Enemy spotted at 180 degrees!"
                            },
                            {
                                "start": 12.5,
                                "end": 15.3,
                                "text": "I'm pushing to the building."
                            }
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_ui_regions() -> Dict[str, Any]:
    """Create sample UI regions configuration."""
    return {
        "health": {"x": 50, "y": 950, "width": 100, "height": 40},
        "ammo": {"x": 1750, "y": 950, "width": 120, "height": 40},
        "kills": {"x": 1800, "y": 50, "width": 80, "height": 30}
    }


@pytest.fixture
def mock_openai_api_key(monkeypatch):
    """Mock OpenAI API key for tests that don't actually call the API."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234567890")


@pytest.fixture
def sample_clip_labels() -> list:
    """Return sample CLIP labels for testing."""
    return [
        "player running",
        "player jumping",
        "player shooting",
        "player parachuting",
        "player dead",
        "main menu screen",
        "victory screen",
        "defeat screen"
    ]


# Markers for conditional test skipping
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU/CUDA"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring OpenAI API key"
    )
    config.addinivalue_line(
        "markers", "requires_video_file: mark test as requiring video files"
    )
