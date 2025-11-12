"""Core Gameplay Analysis Toolkit.

This package provides three core functionalities for extracting structured
data from gameplay videos using AI models.

Modules:
    hud_extraction: Extract HUD/UI data using GPT-4V or OCR
    audio_transcription: Transcribe audio with speaker diarization
    game_detection: Detect game boundaries and outcomes
    utils: Supporting utilities (CLIP, frame extraction, etc.)

Example:
    >>> from src.core.hud_extraction import extract_hud_values_gpt4v
    >>> from src.core.audio_transcription import transcribe_audio
    >>> from src.core.game_detection import segment_video_games
    >>>
    >>> # Extract HUD data
    >>> result = extract_hud_values_gpt4v(
    ...     video_path="gameplay.mp4",
    ...     clip_file="actions.json",
    ...     output_file="hud.json"
    ... )
"""

__version__ = "2.0.0"
__author__ = "Gameplay Analysis Toolkit"
__all__ = ['hud_extraction', 'audio_transcription', 'game_detection', 'utils']
