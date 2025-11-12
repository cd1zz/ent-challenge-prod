"""
Utility Modules

Supporting utilities for core functionality:
- CLIP action classification
- CLAP audio event classification
- Intelligent frame sampling
- Video frame extraction
- Video slicing
- UI region setup for OCR
"""

from .action_classifier import ActionClassifier, classify_video_actions
from .audio_classifier import AudioClassifier
from .intelligent_sampler import IntelligentSampler
from .frame_extractor import FrameExtractor
from .region_setup import RegionSetupTool

__all__ = [
    'ActionClassifier',
    'classify_video_actions',
    'AudioClassifier',
    'IntelligentSampler',
    'FrameExtractor',
    'RegionSetupTool'
]
