#!/usr/bin/env python3
"""
Audio Transcription Script
Transcribe audio with speaker diarization using GPT-4o audio models.
"""

import sys
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.audio_transcription import GPT4oTranscriber


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe audio with speaker diarization using GPT-4o',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription
  python transcribe_audio.py audio.wav -o transcript.json

  # Fast parallel processing with 8 workers
  python transcribe_audio.py audio.wav -o transcript.json --max-workers 8

  # Process only first 5 minutes (testing)
  python transcribe_audio.py audio.wav -o transcript.json --num-chunks 1

  # Resume interrupted transcription (automatic)
  python transcribe_audio.py audio.wav -o transcript.json

  # Start fresh, ignore checkpoints
  python transcribe_audio.py audio.wav -o transcript.json --no-resume

Notes:
  - Requires OPENAI_API_KEY in environment or .env file
  - Automatically converts OPUS to WAV if needed
  - Supports checkpoint/resume for long files
  - Output JSON includes speaker diarization (speaker_0, speaker_1, etc.)
        """
    )

    parser.add_argument('input', help='Input audio file (.wav or .opus)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output file path (.json or .txt)')

    parser.add_argument('--format', choices=['json', 'text'], default='json',
                       help='Output format (default: json with diarization)')
    parser.add_argument('--model', default='gpt-4o-audio-preview',
                       help='GPT-4o model to use (default: gpt-4o-audio-preview)')

    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Enable parallel processing (default: True)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--chunk-duration', type=float, default=300.0,
                       help='Chunk duration in seconds for parallel mode (default: 300)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')

    parser.add_argument('--checkpoint-dir', help='Checkpoint directory (default: .transcription_checkpoints)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh, ignore checkpoints')
    parser.add_argument('--num-chunks', type=int,
                       help='Process only first N chunks (for testing)')

    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (default: 0.0 for deterministic)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Check API key
    if not os.environ.get('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment")
        print("\nPlease create a .env file with:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        print("\nOr export it in your shell:")
        print("  export OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)

    # Convert OPUS to WAV if needed
    input_path = args.input
    temp_wav = None

    if input_path.lower().endswith('.opus'):
        try:
            from pydub import AudioSegment
            print(f"Converting {input_path} to WAV format...")
            temp_wav = input_path.replace('.opus', '_temp.wav')
            audio = AudioSegment.from_file(input_path, format="opus")
            audio.export(temp_wav, format="wav")
            input_path = temp_wav
            print(f"✓ Converted to: {temp_wav}")
        except ImportError:
            print("Error: pydub required for OPUS conversion")
            print("Install: pip install pydub")
            sys.exit(1)
        except Exception as e:
            print(f"Error converting audio: {e}")
            sys.exit(1)
    elif not input_path.lower().endswith('.wav'):
        print("Error: Input must be .wav or .opus format")
        sys.exit(1)

    # Determine parallel mode
    parallel = args.parallel and not args.no_parallel

    print("=" * 70)
    print("AUDIO TRANSCRIPTION (GPT-4o with Speaker Diarization)")
    print("=" * 70)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Model:  {args.model}")
    print(f"Parallel processing: {parallel}")
    if parallel:
        print(f"  Chunk duration: {args.chunk_duration}s")
        print(f"  Workers: {args.max_workers}")
    print(f"Resume from checkpoint: {not args.no_resume}")
    print("=" * 70)

    try:
        # Initialize transcriber
        transcriber = GPT4oTranscriber(
            model=args.model,
            chunk_duration=args.chunk_duration,
            max_workers=args.max_workers,
            temperature=args.temperature
        )

        # Transcribe
        result = transcriber.transcribe_file(
            audio_path=input_path,
            parallel=parallel,
            checkpoint_dir=args.checkpoint_dir,
            resume=not args.no_resume,
            num_chunks=args.num_chunks
        )

        # Save results
        transcriber.save_transcription(
            result=result,
            output_path=args.output,
            format=args.format
        )

        print("\n" + "=" * 70)
        print("TRANSCRIPTION COMPLETE")
        print("=" * 70)
        print(f"Duration: {result['duration']:.2f} seconds")
        if result.get('parallel'):
            print(f"Chunks processed: {result.get('num_chunks', 0)}")
        print(f"Output saved to: {args.output}")

        # Show preview
        text_preview = result['text'][:300]
        if len(result['text']) > 300:
            text_preview += "..."
        print("\nTranscription preview:")
        print("-" * 70)
        print(text_preview)
        print("-" * 70)

    except Exception as e:
        print(f"\nError: Transcription failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    finally:
        # Clean up temp WAV
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
                print(f"\nCleaned up: {temp_wav}")
            except Exception as e:
                print(f"\nWarning: Could not remove temp file {temp_wav}: {e}")


if __name__ == '__main__':
    main()
