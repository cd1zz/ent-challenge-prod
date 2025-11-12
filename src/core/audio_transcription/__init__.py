"""
Audio Transcription Module

Provides GPT-4o audio transcription with speaker diarization,
parallel processing, and checkpoint/resume capabilities.
"""

from .gpt4o_transcribe import GPT4oTranscriber, TranscriptionChunk, transcribe_audio

__all__ = [
    'GPT4oTranscriber',
    'TranscriptionChunk',
    'transcribe_audio'
]
