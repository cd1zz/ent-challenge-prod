"""
HUD Extraction Module

Provides three approaches for extracting HUD/UI data from gameplay videos:
1. GPT-4V Vision Model - High accuracy, intelligent sampling
2. PaddleOCR - High accuracy, supports 80+ languages, GPU-accelerated
3. Tesseract OCR - Fast, requires region calibration
"""

from .gpt4v_hud_extractor import GPT4VHUDExtractor, extract_hud_values_gpt4v
from .paddleocr_extractor import PaddleOCRExtractor, load_regions_from_file, extract_hud_data
from .ui_ocr_extractor import SuperPeopleUIExtractor, extract_ui_metrics_from_video
from .hud_config import get_gpt4v_field_descriptions

__all__ = [
    'GPT4VHUDExtractor',
    'extract_hud_values_gpt4v',
    'PaddleOCRExtractor',
    'load_regions_from_file',
    'extract_hud_data',
    'SuperPeopleUIExtractor',
    'extract_ui_metrics_from_video',
    'get_gpt4v_field_descriptions'
]
