"""
Game Detection Module

Provides automatic detection of:
1. Game boundaries (start/stop) from CLIP action classifications
2. Game outcomes (win/loss/placement/kills) from GPT-4V screen analysis
3. Game over screens and stats extraction using PaddleOCR
"""

from .game_segmenter import GameSegmenter, segment_video_games
from .game_outcome_detector import GameOutcomeDetector, detect_outcomes
from .game_over_detector import GameOverDetector, scan_video_for_game_over

__all__ = [
    'GameSegmenter',
    'segment_video_games',
    'GameOutcomeDetector',
    'detect_outcomes',
    'GameOverDetector',
    'scan_video_for_game_over'
]
