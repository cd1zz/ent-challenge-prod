"""
GPT-4o Audio Transcription using Responses API with Parallel Processing

This module provides functionality to transcribe audio files using OpenAI's GPT-4o
audio models via the Responses API. It supports parallel processing of audio chunks
for improved performance on longer audio files.
"""

import os
import base64
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptionChunk:
    """Represents a transcribed audio chunk with metadata."""
    chunk_index: int
    start_time: float
    end_time: float
    text: str
    duration: float
    raw_response: Optional[Dict] = None  # Store full API response for debugging/diarization


class GPT4oTranscriber:
    """
    Transcribes audio files using GPT-4o audio models with parallel processing.
    
    Supports chunking large audio files and processing them in parallel for
    improved performance.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        chunk_duration: Optional[float] = None,
        max_workers: Optional[int] = None,
        temperature: Optional[float] = None,
        audio_voice: Optional[str] = None,
        transcription_prompt: Optional[str] = None
    ):
        """
        Initialize the transcriber.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable
            model: Model to use. If None, uses GPT4O_TRANSCRIBE_MODEL env var (default: gpt-4o-audio-preview)
            chunk_duration: Duration in seconds for each chunk. If None, uses GPT4O_CHUNK_DURATION env var (default: 300.0)
            max_workers: Maximum number of parallel workers. If None, uses GPT4O_MAX_WORKERS env var (default: 4)
            temperature: Sampling temperature. If None, uses GPT4O_TEMPERATURE env var (default: 0.0)
            audio_voice: Voice for audio output. If None, uses GPT4O_AUDIO_VOICE env var (default: alloy)
            transcription_prompt: Custom prompt for transcription. If None, uses GPT4O_TRANSCRIPTION_PROMPT env var
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model or os.getenv("GPT4O_TRANSCRIBE_MODEL", "gpt-4o-audio-preview")
        self.chunk_duration = chunk_duration if chunk_duration is not None else float(os.getenv("GPT4O_CHUNK_DURATION", "300.0"))
        self.max_workers = max_workers if max_workers is not None else int(os.getenv("GPT4O_MAX_WORKERS", "4"))
        self.temperature = temperature if temperature is not None else float(os.getenv("GPT4O_TEMPERATURE", "0.0"))
        self.audio_voice = audio_voice or os.getenv("GPT4O_AUDIO_VOICE", "alloy")
        self.transcription_prompt = transcription_prompt or os.getenv(
            "GPT4O_TRANSCRIPTION_PROMPT",
            "Please transcribe this audio file accurately. Provide only the transcription text without any additional commentary or formatting."
        )
        
        logger.info(f"Initialized GPT4oTranscriber with model: {self.model}")
    
    def _encode_audio_to_base64(self, audio_path: str) -> str:
        """
        Encode audio file to base64 string.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Base64 encoded audio string
        """
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get the duration of an audio file in seconds.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Duration in seconds
        """
        try:
            import wave
            with wave.open(audio_path, 'rb') as audio_file:
                frames = audio_file.getnframes()
                rate = audio_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return 0.0
    
    def _split_audio_file(self, audio_path: str, output_dir: str) -> List[Tuple[str, float, float]]:
        """
        Split audio file into chunks for parallel processing.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to store audio chunks
            
        Returns:
            List of tuples (chunk_path, start_time, end_time)
        """
        try:
            from pydub import AudioSegment
        except ImportError:
            logger.error("pydub is required for audio splitting. Install with: pip install pydub")
            raise
        
        # Load audio file
        audio = AudioSegment.from_wav(audio_path)
        duration_ms = len(audio)
        duration_sec = duration_ms / 1000.0
        
        # If audio is shorter than chunk duration, no need to split
        if duration_sec <= self.chunk_duration:
            return [(audio_path, 0.0, duration_sec)]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Split into chunks
        chunk_duration_ms = int(self.chunk_duration * 1000)
        chunks = []
        
        for i, start_ms in enumerate(range(0, duration_ms, chunk_duration_ms)):
            end_ms = min(start_ms + chunk_duration_ms, duration_ms)
            chunk = audio[start_ms:end_ms]
            
            chunk_path = os.path.join(output_dir, f"chunk_{i:04d}.wav")
            chunk.export(chunk_path, format="wav")
            
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            chunks.append((chunk_path, start_sec, end_sec))
            
            logger.info(f"Created chunk {i}: {start_sec:.2f}s - {end_sec:.2f}s")
        
        return chunks
    
    def _transcribe_chunk(
        self,
        audio_path: str,
        chunk_index: int,
        start_time: float,
        end_time: float
    ) -> TranscriptionChunk:
        """
        Transcribe a single audio chunk using GPT-4o.
        
        Args:
            audio_path: Path to the audio chunk
            chunk_index: Index of this chunk
            start_time: Start time in the original audio
            end_time: End time in the original audio
            
        Returns:
            TranscriptionChunk object
        """
        logger.info(f"Transcribing chunk {chunk_index}: {start_time:.2f}s - {end_time:.2f}s")
        
        try:
            # Check if we need to use the Transcription API (for transcribe/diarize models)
            # or the Chat Completions API (for general audio models)
            use_transcription_api = "transcribe" in self.model.lower() or "whisper" in self.model.lower()
            
            if use_transcription_api:
                # Use the Transcription API endpoint for transcribe/diarize models
                logger.info(f"Using Transcription API for model: {self.model}")
                
                # Prepare parameters for the API call
                api_params = {
                    "model": self.model,
                    "file": None,  # Will be set in context manager
                    "response_format": "json"
                }
                
                # Add chunking_strategy and diarized_json for diarization models
                if "diarize" in self.model.lower():
                    api_params["chunking_strategy"] = "auto"
                    api_params["response_format"] = "diarized_json"  # Get speaker segments
                
                with open(audio_path, "rb") as audio_file:
                    api_params["file"] = audio_file
                    response = self.client.audio.transcriptions.create(**api_params)
                
                # Store the full raw response
                raw_response = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
                
                # Extract transcription text
                transcription_text = response.text if hasattr(response, 'text') else str(response)
                
            else:
                # Use Chat Completions API for audio-enabled chat models
                logger.info(f"Using Chat Completions API for model: {self.model}")
                
                # Encode audio to base64
                audio_base64 = self._encode_audio_to_base64(audio_path)
                
                # Create the API request
                response = self.client.chat.completions.create(
                    model=self.model,
                    modalities=["text"],
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": self.transcription_prompt
                                },
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": audio_base64,
                                        "format": "wav"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=self.temperature
                )
                
                # Extract transcription text - handle various response formats
                message = response.choices[0].message
                
                # Store the full raw response for debugging and to preserve all data
                raw_response = response.model_dump()
                
                # Try to get content from various possible locations
                transcription_text = None
                
                if message.content:
                    transcription_text = message.content
                elif hasattr(message, 'text') and message.text:
                    transcription_text = message.text
                elif hasattr(message, 'audio') and message.audio:
                    # If audio response, try to get transcript from audio object
                    if hasattr(message.audio, 'transcript'):
                        transcription_text = message.audio.transcript
                    else:
                        logger.warning(f"Chunk {chunk_index}: Received audio response without transcript")
                        transcription_text = "[Audio response - no text transcript available]"
            
            if not transcription_text:
                # Log the full response for debugging
                logger.error(f"Chunk {chunk_index}: No transcription text found. Response: {raw_response}")
                raise ValueError(f"No transcription text found in response for chunk {chunk_index}")
            
            duration = end_time - start_time
            
            logger.info(f"Completed chunk {chunk_index} ({duration:.2f}s). Preview: {transcription_text[:50] if transcription_text else 'None'}...")
            
            return TranscriptionChunk(
                chunk_index=chunk_index,
                start_time=start_time,
                end_time=end_time,
                text=transcription_text,
                duration=duration,
                raw_response=raw_response
            )
            
        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk_index}: {e}")
            raise
    
    def _save_checkpoint(
        self,
        checkpoint_file: str,
        chunks: List[TranscriptionChunk],
        duration: float
    ):
        """
        Save checkpoint data to file.
        
        Args:
            checkpoint_file: Path to checkpoint file
            chunks: List of completed transcription chunks
            duration: Total audio duration
        """
        try:
            checkpoint_data = {
                "model": self.model,
                "duration": duration,
                "timestamp": Path(checkpoint_file).stat().st_mtime if os.path.exists(checkpoint_file) else None,
                "chunks": [
                    {
                        "index": chunk.chunk_index,
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                        "duration": chunk.duration,
                        "text": chunk.text,
                        "raw_response": chunk.raw_response
                    }
                    for chunk in chunks
                ]
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")
    
    def transcribe_file(
        self,
        audio_path: str,
        parallel: bool = True,
        output_format: str = "text",
        checkpoint_dir: Optional[str] = None,
        resume: bool = True,
        num_chunks: Optional[int] = None
    ) -> Dict:
        """
        Transcribe an audio file with optional parallel processing.
        
        Args:
            audio_path: Path to the .wav audio file
            parallel: If True, split audio and process chunks in parallel
            output_format: Output format - "text", "json", or "chunks"
            checkpoint_dir: Directory to save checkpoints (default: .transcription_checkpoints)
            resume: If True, resume from existing checkpoints
            num_chunks: If specified, process only the first N chunks (useful for testing)
            
        Returns:
            Dictionary containing transcription results:
            - text: Full transcription text
            - chunks: List of transcription chunks (if parallel=True)
            - duration: Total audio duration
            - model: Model used for transcription
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not audio_path.lower().endswith('.wav'):
            raise ValueError("Only .wav files are currently supported")
        
        logger.info(f"Starting transcription of: {audio_path}")
        logger.info(f"Parallel processing: {parallel}")
        
        # Set up checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.path.dirname(audio_path) or '.', ".transcription_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint filename based on audio file and model
        audio_basename = os.path.basename(audio_path)
        checkpoint_file = os.path.join(checkpoint_dir, f"{audio_basename}_{self.model}_checkpoint.json")
        
        duration = self._get_audio_duration(audio_path)
        logger.info(f"Audio duration: {duration:.2f} seconds")
        
        if not parallel or duration <= self.chunk_duration:
            # Process entire file as single chunk
            logger.info("Processing as single chunk")
            chunk = self._transcribe_chunk(audio_path, 0, 0.0, duration)
            
            return {
                "text": chunk.text,
                "chunks": [chunk],
                "duration": duration,
                "model": self.model,
                "parallel": False
            }
        
        # Parallel processing
        logger.info(f"Splitting audio into chunks of {self.chunk_duration}s each")
        temp_dir = os.path.join(os.path.dirname(audio_path), ".transcription_chunks")
        
        try:
            # Split audio into chunks
            audio_chunks = self._split_audio_file(audio_path, temp_dir)
            logger.info(f"Split into {len(audio_chunks)} chunks")
            
            # Limit chunks if num_chunks is specified
            if num_chunks is not None and num_chunks > 0:
                audio_chunks = audio_chunks[:num_chunks]
                logger.info(f"Limited to first {num_chunks} chunk(s) for testing")
            
            # Check for existing checkpoint
            completed_chunks = {}
            if resume and os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                        completed_chunks = {c['index']: c for c in checkpoint_data.get('chunks', [])}
                        logger.info(f"Resuming from checkpoint: {len(completed_chunks)} chunks already completed")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint: {e}")
                    completed_chunks = {}
            
            # Process chunks in parallel
            transcription_chunks = []
            chunks_to_process = []
            
            # Separate completed from pending chunks
            for i, (chunk_path, start_time, end_time) in enumerate(audio_chunks):
                if i in completed_chunks:
                    # Use cached result
                    chunk_data = completed_chunks[i]
                    transcription_chunks.append(TranscriptionChunk(
                        chunk_index=chunk_data['index'],
                        start_time=chunk_data['start_time'],
                        end_time=chunk_data['end_time'],
                        text=chunk_data['text'],
                        duration=chunk_data['duration'],
                        raw_response=chunk_data.get('raw_response')  # May not exist in old checkpoints
                    ))
                    logger.info(f"Loaded chunk {i} from checkpoint")
                else:
                    chunks_to_process.append((i, chunk_path, start_time, end_time))
            
            if chunks_to_process:
                logger.info(f"Processing {len(chunks_to_process)} remaining chunks...")
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all tasks
                    future_to_chunk = {
                        executor.submit(
                            self._transcribe_chunk,
                            chunk_path,
                            i,
                            start_time,
                            end_time
                        ): i
                        for i, chunk_path, start_time, end_time in chunks_to_process
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_chunk):
                        chunk_index = future_to_chunk[future]
                        try:
                            chunk_result = future.result()
                            transcription_chunks.append(chunk_result)
                            
                            # Save checkpoint after each successful chunk
                            self._save_checkpoint(checkpoint_file, transcription_chunks, duration)
                            logger.info(f"Checkpoint saved: {len(transcription_chunks)}/{len(audio_chunks)} chunks completed")
                            
                        except Exception as e:
                            logger.error(f"Chunk {chunk_index} failed: {e}")
                            # Save checkpoint even on failure
                            self._save_checkpoint(checkpoint_file, transcription_chunks, duration)
                            raise
            
            # Sort chunks by index to maintain order
            transcription_chunks.sort(key=lambda x: x.chunk_index)
            
            # Combine all transcriptions
            full_text = "\n\n".join(chunk.text for chunk in transcription_chunks)
            
            logger.info("Transcription completed successfully")
            
            # Clean up checkpoint file on successful completion
            if os.path.exists(checkpoint_file):
                try:
                    os.remove(checkpoint_file)
                    logger.info("Removed checkpoint file after successful completion")
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint file: {e}")
            
            return {
                "text": full_text,
                "chunks": transcription_chunks,
                "duration": duration,
                "model": self.model,
                "parallel": True,
                "num_chunks": len(transcription_chunks)
            }
            
        finally:
            # Cleanup temporary chunk files
            if os.path.exists(temp_dir):
                import shutil
                try:
                    shutil.rmtree(temp_dir)
                    logger.info("Cleaned up temporary chunk files")
                except Exception as e:
                    logger.warning(f"Could not clean up temp directory: {e}")
    
    def save_transcription(
        self,
        result: Dict,
        output_path: str,
        format: str = "text"
    ):
        """
        Save transcription results to a file.
        
        Args:
            result: Transcription result dictionary from transcribe_file()
            output_path: Path to save the transcription
            format: Output format - "text" or "json"
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            # Save as JSON with full details
            output_data = {
                "text": result["text"],
                "duration": result["duration"],
                "model": result["model"],
                "parallel": result.get("parallel", False),
                "num_chunks": result.get("num_chunks", 1),
                "chunks": [
                    {
                        "index": chunk.chunk_index,
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                        "duration": chunk.duration,
                        "text": chunk.text,
                        "raw_response": chunk.raw_response  # Include full API response
                    }
                    for chunk in result["chunks"]
                ]
            }
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved JSON transcription to: {output_path}")
        
        else:  # text format
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            
            logger.info(f"Saved text transcription to: {output_path}")


def transcribe_audio(
    audio_path: str,
    output_path: Optional[str] = None,
    parallel: bool = True,
    model: Optional[str] = None,
    chunk_duration: Optional[float] = None,
    max_workers: Optional[int] = None,
    output_format: str = "text",
    checkpoint_dir: Optional[str] = None,
    resume: bool = True
) -> Dict:
    """
    Convenience function to transcribe an audio file.
    
    Args:
        audio_path: Path to the .wav audio file
        output_path: Optional path to save transcription (if None, returns only)
        parallel: If True, use parallel processing for long files
        model: GPT-4o model to use (if None, uses env var or default)
        chunk_duration: Duration in seconds for each chunk (if None, uses env var or default)
        max_workers: Maximum parallel workers (if None, uses env var or default)
        output_format: "text" or "json"
        checkpoint_dir: Directory to save checkpoints (default: .transcription_checkpoints)
        resume: If True, resume from existing checkpoints
        
    Returns:
        Dictionary containing transcription results
    """
    transcriber = GPT4oTranscriber(
        model=model,
        chunk_duration=chunk_duration,
        max_workers=max_workers
    )
    
    result = transcriber.transcribe_file(
        audio_path=audio_path,
        parallel=parallel,
        output_format=output_format,
        checkpoint_dir=checkpoint_dir,
        resume=resume
    )
    
    if output_path:
        transcriber.save_transcription(result, output_path, format=output_format)
    
    return result
