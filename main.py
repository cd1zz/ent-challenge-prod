#!/usr/bin/env python3
"""Gameplay Analysis Toolkit - Unified CLI Entry Point.

This module provides a unified command-line interface for all gameplay analysis
functionality. It routes commands to the appropriate core modules.

Usage:
    python main.py extract-hud video.mkv -o hud.json --method gpt4v
    python main.py transcribe audio.wav -o transcript.json
    python main.py detect-actions video.mkv -o actions.json
    python main.py detect-games actions.json -o games.json

For detailed help on each command:
    python main.py extract-hud --help
    python main.py transcribe --help
    python main.py detect-actions --help
    python main.py detect-games --help
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional


def get_output_path(input_path: str, suffix: str, explicit_output: Optional[str] = None) -> str:
    """Get output file path with default to output/ directory with timestamp.

    Args:
        input_path: Path to input file.
        suffix: Suffix to append to input filename (e.g., '_hud.json').
        explicit_output: Explicitly specified output path (takes priority).

    Returns:
        Output file path with timestamp (e.g., output/video_actions_20250111_143052.json).
    """
    if explicit_output:
        return explicit_output

    # Create output directory if it doesn't exist
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    # Construct default path with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_stem = Path(input_path).stem

    # Insert timestamp before file extension
    # e.g., "_actions.json" becomes "_actions_20250111_143052.json"
    suffix_parts = suffix.rsplit('.', 1)
    if len(suffix_parts) == 2:
        suffix_with_timestamp = f"{suffix_parts[0]}_{timestamp}.{suffix_parts[1]}"
    else:
        suffix_with_timestamp = f"{suffix}_{timestamp}"

    return str(output_dir / f"{input_stem}{suffix_with_timestamp}")


def ensure_output_dir(output_path: str) -> None:
    """Ensure output directory exists for the given path.

    Args:
        output_path: Full path to output file.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)


def check_output_file_exists(output_path: str) -> bool:
    """Check if output file exists and prompt user for action.

    Args:
        output_path: Path to output file.

    Returns:
        True if should proceed with writing, False if should quit.

    Raises:
        SystemExit: If user chooses to quit or use existing file.
    """
    if not Path(output_path).exists():
        return True

    print(f"\n⚠️  Output file already exists: {output_path}")
    print("\nWhat would you like to do?")
    print("  [U] Use existing file (skip processing)")
    print("  [O] Overwrite (continue processing)")
    print("  [Q] Quit (exit without processing)")

    while True:
        choice = input("\nChoice (U/O/Q): ").strip().upper()

        if choice == 'U':
            print(f"\n✓ Using existing file: {output_path}")
            print("Skipping processing.")
            raise SystemExit(0)
        elif choice == 'O':
            print(f"\n⚠️  Will overwrite: {output_path}")
            return True
        elif choice == 'Q':
            print("\n✓ Exiting without processing.")
            raise SystemExit(0)
        else:
            print("Invalid choice. Please enter U, O, or Q.")


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the gameplay analysis toolkit.

    Args:
        argv: Command-line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        description='Gameplay Analysis Toolkit - Extract data from gameplay videos',
        epilog='For detailed help: python main.py <command> --help'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        title='Available Commands',
        description='Select a command to run',
        help='Use <command> --help for more information'
    )

    # =========================================================================
    # HUD EXTRACTION COMMAND
    # =========================================================================
    extract_parser = subparsers.add_parser(
        'extract-hud',
        help='Extract HUD/UI values using GPT-4V/PaddleOCR/Tesseract',
        description='Extract HUD values using GPT-4V, PaddleOCR, or Tesseract OCR'
    )
    extract_parser.add_argument(
        'video',
        help='Path to video file'
    )
    extract_parser.add_argument(
        '-o', '--output',
        help='Output file path (.json or .csv, default: output/{video_stem}_hud_YYYYMMDD_HHMMSS.json)'
    )
    extract_parser.add_argument(
        '-m', '--method',
        choices=['gpt4v', 'paddleocr', 'ocr'],
        default='gpt4v',
        help='Extraction method (default: gpt4v)'
    )
    extract_parser.add_argument(
        '--clip',
        help='Path to CLIP actions JSON (required for gpt4v method)'
    )
    extract_parser.add_argument(
        '--transcript',
        help='Path to transcript JSON (optional, improves gpt4v sampling)'
    )
    extract_parser.add_argument(
        '--regions',
        help='Path to UI regions JSON (required for ocr/paddleocr methods)'
    )
    extract_parser.add_argument(
        '--enhanced',
        action='store_true',
        help='Use enhanced preprocessing for OCR (better for low-contrast HUDs)'
    )
    extract_parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration (for PaddleOCR)'
    )

    # GPT-4V specific options
    extract_parser.add_argument(
        '--min-interval',
        type=float,
        default=4.0,
        help='Minimum seconds between extractions (gpt4v only, default: 4.0)'
    )
    extract_parser.add_argument(
        '--max-interval',
        type=float,
        default=18.0,
        help='Maximum seconds without extraction (gpt4v only, default: 18.0)'
    )

    # OCR specific options
    extract_parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Frame sampling interval in seconds (ocr/paddleocr, default: 2.0)'
    )
    extract_parser.add_argument(
        '--det-thresh',
        type=float,
        default=0.3,
        help='Detection threshold for PaddleOCR (default: 0.3, lower = more sensitive)'
    )
    extract_parser.add_argument(
        '--rec-thresh',
        type=float,
        default=0.5,
        help='Recognition threshold for PaddleOCR (default: 0.5)'
    )

    extract_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )

    # =========================================================================
    # AUDIO TRANSCRIPTION COMMAND
    # =========================================================================
    transcribe_parser = subparsers.add_parser(
        'transcribe',
        help='Transcribe audio using GPT-4o (with speaker diarization)',
        description='Transcribe gameplay audio using GPT-4o with speaker identification'
    )
    transcribe_parser.add_argument(
        'audio',
        help='Path to audio file (.wav or .opus)'
    )
    transcribe_parser.add_argument(
        '-o', '--output',
        help='Output file path (.json or .txt, default: output/{audio_stem}_transcript_YYYYMMDD_HHMMSS.json)'
    )
    transcribe_parser.add_argument(
        '--format',
        choices=['json', 'text'],
        default='json',
        help='Output format (default: json)'
    )
    transcribe_parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Enable parallel processing (default: True)'
    )
    transcribe_parser.add_argument(
        '--no-parallel',
        action='store_false',
        dest='parallel',
        help='Disable parallel processing'
    )
    transcribe_parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum parallel workers (default: 4)'
    )
    transcribe_parser.add_argument(
        '--chunk-duration',
        type=float,
        default=300.0,
        help='Chunk duration in seconds (default: 300.0)'
    )
    transcribe_parser.add_argument(
        '--num-chunks',
        type=int,
        help='Process only first N chunks (for testing)'
    )
    transcribe_parser.add_argument(
        '--no-resume',
        action='store_false',
        dest='resume',
        help='Disable checkpoint resume (start fresh)'
    )
    transcribe_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        help='Print progress information (default: True)'
    )

    # =========================================================================
    # ACTION DETECTION COMMAND (CLIP)
    # =========================================================================
    actions_parser = subparsers.add_parser(
        'detect-actions',
        help='Classify video frames using CLIP (action detection)',
        description='Run CLIP zero-shot classification on video frames using labels from .env'
    )
    actions_parser.add_argument(
        'video',
        help='Path to video file'
    )
    actions_parser.add_argument(
        '-o', '--output',
        help='Output file path for actions JSON (default: output/{video_stem}_actions_YYYYMMDD_HHMMSS.json)'
    )
    actions_parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Frame sampling interval in seconds (default: 2.0)'
    )
    actions_parser.add_argument(
        '--max-duration',
        type=float,
        help='Process only first N seconds of video (for testing)'
    )
    actions_parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for CLIP processing (default: 16)'
    )
    actions_parser.add_argument(
        '--labels',
        help='Custom CLIP labels (pipe-delimited, replaces defaults). Example: "action1|action2|action3"'
    )
    actions_parser.add_argument(
        '--add-labels',
        help='Additional CLIP labels to append (pipe-delimited). Example: "extra1|extra2"'
    )
    actions_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress (default: True)'
    )
    actions_parser.add_argument(
        '--save-uncertain-images',
        action='store_true',
        help='Save frames where CLIP is uncertain or has low confidence (<50%%) to output/uncertain_frames/'
    )
    actions_parser.add_argument(
        '--uncertain-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for saving uncertain images (default: 0.5 or 50%%)'
    )

    # =========================================================================
    # GAME DETECTION COMMAND (Segmentation)
    # =========================================================================
    games_parser = subparsers.add_parser(
        'detect-games',
        help='Detect game boundaries from CLIP actions',
        description='Segment games by analyzing CLIP action classifications to find start/stop times'
    )
    games_parser.add_argument(
        'actions',
        help='Path to actions JSON file (output from detect-actions)'
    )
    games_parser.add_argument(
        '-o', '--output',
        help='Output file path for games JSON (default: output/{actions_stem}_games_YYYYMMDD_HHMMSS.json)'
    )
    games_parser.add_argument(
        '--min-duration',
        type=float,
        default=60.0,
        help='Minimum game duration in seconds (default: 60.0)'
    )
    games_parser.add_argument(
        '--parachute-thresh',
        type=float,
        default=0.85,
        help='Parachute confidence threshold for game start (default: 0.85)'
    )
    games_parser.add_argument(
        '--death-thresh',
        type=float,
        default=0.60,
        help='Death confidence threshold for game end (default: 0.60)'
    )
    games_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress (default: True)'
    )

    # =========================================================================
    # GAME OVER DETECTION COMMAND (PaddleOCR)
    # =========================================================================
    gameover_parser = subparsers.add_parser(
        'detect-gameover',
        help='Detect game over screens using PaddleOCR',
        description='Automatically detect game over screens and extract stats (free, no API key needed)'
    )
    gameover_parser.add_argument(
        'video',
        help='Path to video file'
    )
    gameover_parser.add_argument(
        '-o', '--output',
        help='Output file path (.json, default: output/{video_stem}_gameover_YYYYMMDD_HHMMSS.json)'
    )
    gameover_parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration'
    )
    gameover_parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Frame sampling interval in seconds (default: 2.0)'
    )
    gameover_parser.add_argument(
        '--det-thresh',
        type=float,
        default=0.3,
        help='Detection threshold (default: 0.3, lower = more sensitive)'
    )
    gameover_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        help='Print detected screens (default: True)'
    )

    # =========================================================================
    # AUDIO EVENTS DETECTION COMMAND (CLAP)
    # =========================================================================
    audio_parser = subparsers.add_parser(
        'detect-audio-events',
        help='Detect audio events in gameplay using CLAP (local, GPU-accelerated)',
        description='Classify audio events using CLAP zero-shot audio classification (runs locally on GPU)'
    )
    audio_parser.add_argument(
        'video',
        help='Path to video file'
    )
    audio_parser.add_argument(
        '-o', '--output',
        help='Output file path (.json, default: output/{video_stem}_audio_YYYYMMDD_HHMMSS.json)'
    )
    audio_parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Audio extraction interval in seconds (default: 2.0)'
    )
    audio_parser.add_argument(
        '--segment-duration',
        type=float,
        default=2.0,
        help='Duration of each audio segment in seconds (default: 2.0)'
    )
    audio_parser.add_argument(
        '--labels',
        type=str,
        help='Custom audio event labels (pipe-separated, e.g. "gunshots|footsteps|explosions")'
    )
    audio_parser.add_argument(
        '--add-labels',
        type=str,
        help='Add labels to defaults (pipe-separated)'
    )
    audio_parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Minimum confidence threshold (0.0-1.0, default: 0.0 shows all events)'
    )
    audio_parser.add_argument(
        '--max-duration',
        type=float,
        help='Maximum video duration to process in seconds'
    )
    audio_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress (default: True)'
    )

    # =========================================================================
    # SETUP REGIONS COMMAND
    # =========================================================================
    setup_parser = subparsers.add_parser(
        'setup-regions',
        help='Draw OCR region boxes on gameplay frame (interactive: click-drag boxes for health, ammo, etc.)',
        description='Define UI regions for OCR extraction (3 methods: GPT-4V auto, click-drag, manual)'
    )
    setup_parser.add_argument(
        'image',
        help='Path to reference frame image (screenshot from gameplay)'
    )
    setup_parser.add_argument(
        '-o', '--output',
        help='Output file path for regions JSON (default: output/ui_regions_YYYYMMDD_HHMMSS.json)'
    )
    setup_parser.add_argument(
        '-m', '--method',
        type=int,
        choices=[1, 2, 3],
        help='Force specific method (1=GPT-4V auto, 2=click-drag, 3=manual)'
    )

    # Parse arguments
    args = parser.parse_args(argv)

    # Show help if no command provided
    if args.command is None:
        parser.print_help()
        return 1

    # =========================================================================
    # ROUTE TO APPROPRIATE COMMAND HANDLER
    # =========================================================================

    try:
        if args.command == 'extract-hud':
            return cmd_extract_hud(args)
        elif args.command == 'transcribe':
            return cmd_transcribe(args)
        elif args.command == 'detect-actions':
            return cmd_detect_actions(args)
        elif args.command == 'detect-games':
            return cmd_detect_games(args)
        elif args.command == 'detect-gameover':
            return cmd_detect_gameover(args)
        elif args.command == 'detect-audio-events':
            return cmd_detect_audio_events(args)
        elif args.command == 'setup-regions':
            return cmd_setup_regions(args)
        else:
            print(f"Error: Unknown command '{args.command}'")
            return 1

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_extract_hud(args: argparse.Namespace) -> int:
    """Execute HUD extraction command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    from dotenv import load_dotenv
    load_dotenv()

    # Determine output path with method suffix
    method_suffix = f"_hud_{args.method}.json" if args.method != 'gpt4v' else '_hud.json'
    output_path = get_output_path(args.video, method_suffix, args.output)
    ensure_output_dir(output_path)

    # Check if output file already exists
    check_output_file_exists(output_path)

    if args.method == 'gpt4v':
        # GPT-4V method
        from src.core.hud_extraction import extract_hud_values_gpt4v

        if not args.clip:
            print("Error: --clip is required for gpt4v method")
            return 1

        result = extract_hud_values_gpt4v(
            video_path=args.video,
            clip_file=args.clip,
            transcript_file=args.transcript,
            output_file=output_path,
            min_interval=args.min_interval,
            max_interval=args.max_interval,
            verbose=args.verbose
        )

        if args.verbose:
            print(f"\n✓ Extraction complete")
            print(f"  Extracted {len(result['results'])} frames")
            print(f"  Results saved to: {output_path}")

    elif args.method == 'paddleocr':
        # PaddleOCR method - use core module
        from src.core.hud_extraction import PaddleOCRExtractor, load_regions_from_file, extract_hud_data

        if not args.regions:
            print("Error: --regions is required for paddleocr method")
            return 1

        # Load regions
        regions = load_regions_from_file(args.regions)
        if args.verbose:
            print(f"Loaded {len(regions)} UI regions from {args.regions}")

        # Initialize extractor
        extractor = PaddleOCRExtractor(
            regions=regions,
            use_gpu=args.gpu,
            show_log=args.verbose,
            det_db_thresh=getattr(args, 'det_thresh', 0.3),
            det_db_box_thresh=getattr(args, 'rec_thresh', 0.5)
        )

        # Extract HUD data
        df = extract_hud_data(
            video_path=args.video,
            extractor=extractor,
            interval_seconds=getattr(args, 'interval', 2.0),
            output_file=output_path,
            verbose=args.verbose
        )

        if args.verbose:
            print(f"\n✓ Extraction complete")
            print(f"  Extracted {len(df)} frames")
            print(f"  Results saved to: {output_path}")

    elif args.method == 'ocr':
        # Tesseract OCR method
        from src.core.hud_extraction import SuperPeopleUIExtractor

        if not args.regions:
            print("Error: --regions is required for ocr method")
            return 1

        try:
            # Initialize extractor
            extractor = SuperPeopleUIExtractor(regions_file=args.regions)

            # Extract from video
            extractor.extract_from_video(
                video_path=args.video,
                output_file=output_path,
                interval_seconds=getattr(args, 'interval', 2.0),
                enhanced=args.enhanced,
                verbose=args.verbose
            )

            if args.verbose:
                print(f"\n✓ Extraction complete")
                print(f"  Results saved to: {output_path}")

        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Execute audio transcription command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    from dotenv import load_dotenv
    from src.core.audio_transcription import transcribe_audio

    load_dotenv()

    # Determine output path
    format_ext = f".{args.format}" if hasattr(args, 'format') and args.format else '.json'
    output_path = get_output_path(args.audio, f'_transcript{format_ext}', args.output)
    ensure_output_dir(output_path)

    # Check if output file already exists
    check_output_file_exists(output_path)

    # Handle OPUS conversion if needed
    audio_path = args.audio
    if audio_path.lower().endswith('.opus'):
        import tempfile
        from pydub import AudioSegment

        if args.verbose:
            print(f"Converting OPUS to WAV...")

        # Convert OPUS to WAV
        audio = AudioSegment.from_file(audio_path, format='opus')
        wav_path = tempfile.mktemp(suffix='.wav')
        audio.export(wav_path, format='wav')
        audio_path = wav_path

        if args.verbose:
            print(f"✓ Converted to temporary WAV file")

    result = transcribe_audio(
        audio_path=audio_path,
        output_path=output_path,
        parallel=args.parallel,
        max_workers=args.max_workers if args.parallel else 1,
        chunk_duration=args.chunk_duration,
        output_format=args.format,
        resume=args.resume
    )

    if args.verbose:
        print(f"\n✓ Transcription complete")
        print(f"  Duration: {result['duration']:.1f}s")
        print(f"  Chunks: {result.get('num_chunks', 1)}")
        print(f"  Results saved to: {output_path}")

    return 0


def cmd_detect_actions(args: argparse.Namespace) -> int:
    """Execute CLIP action classification command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    from dotenv import load_dotenv
    from src.core.utils import ActionClassifier
    import os

    load_dotenv()

    # Determine output path
    output_path = get_output_path(args.video, '_actions.json', args.output)
    ensure_output_dir(output_path)

    # Check if output file already exists
    check_output_file_exists(output_path)

    # Process labels (priority: --labels > --add-labels + base > .env > DEFAULT)
    custom_labels = None
    if args.labels:
        # Replace all labels with custom ones
        custom_labels = [label.strip() for label in args.labels.split('|') if label.strip()]
        if args.verbose:
            print(f"Using {len(custom_labels)} custom labels from --labels")
    elif args.add_labels:
        # Add to existing labels (from .env or DEFAULT)
        additional_labels = [label.strip() for label in args.add_labels.split('|') if label.strip()]

        # Get base labels from .env or DEFAULT
        env_labels = os.getenv('CLIP_LABELS')
        if env_labels:
            base_labels = [label.strip() for label in env_labels.split('|') if label.strip()]
            if args.verbose:
                print(f"Using {len(base_labels)} base labels from .env")
        else:
            base_labels = ActionClassifier.DEFAULT_LABELS.copy()
            if args.verbose:
                print(f"Using {len(base_labels)} default labels")

        # Combine base + additional
        custom_labels = base_labels + additional_labels
        if args.verbose:
            print(f"Added {len(additional_labels)} custom labels, total: {len(custom_labels)}")

    if args.verbose:
        print("="*70)
        print("CLIP ACTION CLASSIFICATION")
        print("="*70)
        print(f"Video: {args.video}")
        print(f"Output: {output_path}")
        print(f"Sample interval: {args.interval}s")
        if args.max_duration:
            print(f"Max duration: {args.max_duration}s")
        if custom_labels:
            print(f"Total labels: {len(custom_labels)}")
        print("="*70)

    # Calculate max frames if duration limit specified
    max_frames = None
    if args.max_duration:
        max_frames = int(args.max_duration / args.interval)

    # Initialize classifier with custom labels if provided
    classifier = ActionClassifier(labels=custom_labels)

    # Run CLIP classification manually
    from src.core.utils import FrameExtractor
    import json
    from datetime import datetime
    import cv2

    # Setup uncertain images directory if requested
    uncertain_dir = None
    uncertain_count = 0
    if args.save_uncertain_images:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_stem = Path(args.video).stem
        uncertain_dir = Path('output') / 'uncertain_frames' / f"{video_stem}_{timestamp_str}"
        uncertain_dir.mkdir(parents=True, exist_ok=True)
        if args.verbose:
            print(f"Will save uncertain frames to: {uncertain_dir}\n")

    results = []
    batch_frames = []
    batch_timestamps = []

    with FrameExtractor(args.video) as extractor:
        if args.verbose:
            info = extractor.get_info()
            print(f"Video duration: {info['duration_formatted']}")
            print(f"Processing frames...\n")

        for timestamp, frame in extractor.extract_frames(
            interval_seconds=args.interval,
            max_frames=max_frames
        ):
            batch_frames.append(frame)
            batch_timestamps.append(timestamp)

            # Process in batches
            if len(batch_frames) >= args.batch_size:
                predictions_batch = classifier.classify_batch(batch_frames)

                for idx, (ts, preds, frame) in enumerate(zip(batch_timestamps, predictions_batch, batch_frames)):
                    top_action = classifier.get_primary_action(preds)
                    top_3 = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_confidence = top_3[0][1]

                    results.append({
                        'timestamp': ts,
                        'primary_action': top_action,
                        'top_3_predictions': [{'action': a, 'probability': p} for a, p in top_3],
                        'all_predictions': preds
                    })

                    # Save uncertain frames if requested
                    if uncertain_dir and (top_action == "uncertain" or top_confidence < args.uncertain_threshold):
                        frame_filename = f"frame_{int(ts*1000):08d}ms_conf{top_confidence*100:.1f}.jpg"
                        frame_path = uncertain_dir / frame_filename
                        cv2.imwrite(str(frame_path), frame)
                        uncertain_count += 1

                    if args.verbose:
                        print(f"[{ts:7.1f}s] {top_action} ({top_confidence*100:.1f}%)")

                batch_frames = []
                batch_timestamps = []

        # Process remaining frames
        if batch_frames:
            predictions_batch = classifier.classify_batch(batch_frames)
            for ts, preds, frame in zip(batch_timestamps, predictions_batch, batch_frames):
                top_action = classifier.get_primary_action(preds)
                top_3 = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:3]
                top_confidence = top_3[0][1]

                results.append({
                    'timestamp': ts,
                    'primary_action': top_action,
                    'top_3_predictions': [{'action': a, 'probability': p} for a, p in top_3],
                    'all_predictions': preds
                })

                # Save uncertain frames if requested
                if uncertain_dir and (top_action == "uncertain" or top_confidence < args.uncertain_threshold):
                    frame_filename = f"frame_{int(ts*1000):08d}ms_conf{top_confidence*100:.1f}.jpg"
                    frame_path = uncertain_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    uncertain_count += 1

                if args.verbose:
                    print(f"[{ts:7.1f}s] {top_action} ({top_confidence*100:.1f}%)")

    # Save results
    output_data = {
        'video_path': args.video,
        'timestamp': datetime.now().isoformat(),
        'interval_seconds': args.interval,
        'total_frames': len(results),
        'labels': classifier.labels,
        'results': results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    if args.verbose:
        print(f"\n✓ Action classification complete")
        print(f"  Classified {len(results)} frames")
        print(f"  Results saved to: {output_path}")

        # Print summary
        action_counts = {}
        for r in results:
            action = r['primary_action']
            action_counts[action] = action_counts.get(action, 0) + 1

        print("\nAction Distribution:")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(results)) * 100
            print(f"  {action}: {count} ({percentage:.1f}%)")

        # Print uncertain frames summary
        if args.save_uncertain_images and uncertain_count > 0:
            print(f"\n⚠️  Uncertain Frames:")
            print(f"  Saved {uncertain_count} uncertain/low-confidence frames")
            print(f"  Location: {uncertain_dir}")
            print(f"  Threshold: {args.uncertain_threshold*100:.0f}% confidence")

    return 0


def cmd_detect_games(args: argparse.Namespace) -> int:
    """Execute game segmentation command from CLIP actions.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    from dotenv import load_dotenv
    from src.core.game_detection import GameSegmenter
    import json

    load_dotenv()

    # Determine output path
    output_path = get_output_path(args.actions, '_games.json', args.output)
    ensure_output_dir(output_path)

    # Check if output file already exists
    check_output_file_exists(output_path)

    if args.verbose:
        print("="*70)
        print("GAME SEGMENTATION FROM CLIP ACTIONS")
        print("="*70)
        print(f"Actions file: {args.actions}")
        print(f"Output: {output_path}")
        print(f"Min duration: {args.min_duration}s")
        print(f"Parachute threshold: {args.parachute_thresh}")
        print(f"Death threshold: {args.death_thresh}")
        print("="*70 + "\n")

    # Load actions JSON
    try:
        with open(args.actions, 'r') as f:
            actions_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Actions file not found: {args.actions}")
        print("Run 'detect-actions' command first to generate actions JSON")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in actions file: {e}")
        return 1

    # Extract results array
    if 'results' not in actions_data:
        print("Error: Actions JSON missing 'results' field")
        return 1

    action_results = actions_data['results']

    if args.verbose:
        print(f"Loaded {len(action_results)} action classifications")

    # Initialize segmenter with custom thresholds
    segmenter = GameSegmenter(
        parachute_confidence_threshold=args.parachute_thresh,
        death_confidence_threshold=args.death_thresh,
        min_game_duration=args.min_duration
    )

    # Segment games
    games = segmenter.segment_games(action_results)

    if args.verbose:
        print(f"\n✓ Found {len(games)} games")
        for i, game in enumerate(games, 1):
            duration = game['end_time'] - game['start_time']
            print(f"  Game {i}: {game['start_time']:.1f}s - {game['end_time']:.1f}s (duration: {duration:.1f}s)")

    # Save results
    output_data = {
        'video_path': actions_data.get('video_path', 'unknown'),
        'actions_file': args.actions,
        'segmentation_params': {
            'min_game_duration': args.min_duration,
            'parachute_threshold': args.parachute_thresh,
            'death_threshold': args.death_thresh
        },
        'summary': {
            'total_games': len(games),
            'total_duration': sum(g['end_time'] - g['start_time'] for g in games)
        },
        'games': games
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    if args.verbose:
        print(f"\n✓ Game segmentation complete")
        print(f"  Results saved to: {output_path}")

    return 0


def cmd_detect_gameover(args: argparse.Namespace) -> int:
    """Execute game over detection command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    from src.core.game_detection import GameOverDetector, scan_video_for_game_over
    import json
    from datetime import datetime

    # Determine output path
    output_path = get_output_path(args.video, '_gameover.json', args.output)
    ensure_output_dir(output_path)

    # Check if output file already exists
    check_output_file_exists(output_path)

    # Initialize detector
    detector = GameOverDetector(
        lang=getattr(args, 'lang', 'en'),
        use_gpu=args.gpu,
        confidence_threshold=getattr(args, 'confidence', 0.6)
    )

    if args.verbose:
        print(f"Scanning {args.video} for game over screens...")

    # Scan video
    game_over_events = scan_video_for_game_over(
        video_path=args.video,
        detector=detector,
        interval_seconds=getattr(args, 'interval', 2.0),
        start_time=getattr(args, 'start_time', 0.0),
        max_duration=getattr(args, 'max_duration', None),
        verbose=args.verbose
    )

    # Save results
    output_data = {
        'video_path': args.video,
        'scan_timestamp': datetime.now().isoformat(),
        'summary': {
            'game_over_events_detected': len(game_over_events)
        },
        'events': game_over_events
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    if args.verbose:
        print(f"\n✓ Detection complete")
        print(f"  Found {len(game_over_events)} game over events")
        print(f"  Results saved to: {output_path}")

    return 0


def cmd_detect_audio_events(args: argparse.Namespace) -> int:
    """Execute CLAP audio event classification command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    from src.core.utils import AudioClassifier
    import json
    from datetime import datetime

    # Determine output path
    output_path = get_output_path(args.video, '_audio.json', args.output)
    ensure_output_dir(output_path)

    # Check if output file already exists
    check_output_file_exists(output_path)

    # Process labels (priority: --labels > --add-labels + base > DEFAULT)
    custom_labels = None
    if args.labels:
        # Replace all labels with custom ones
        custom_labels = [label.strip() for label in args.labels.split('|') if label.strip()]
        if args.verbose:
            print(f"Using {len(custom_labels)} custom labels")
    elif args.add_labels:
        # Add to default labels
        added_labels = [label.strip() for label in args.add_labels.split('|') if label.strip()]
        custom_labels = AudioClassifier.DEFAULT_LABELS + added_labels
        if args.verbose:
            print(f"Using {len(AudioClassifier.DEFAULT_LABELS)} default + {len(added_labels)} custom labels")

    # Print header
    if args.verbose:
        print("="*70)
        print("CLAP AUDIO EVENT CLASSIFICATION (Local, GPU-accelerated)")
        print("="*70)
        print(f"Video: {args.video}")
        print(f"Output: {output_path}")
        print(f"Sample interval: {args.interval}s")
        print(f"Segment duration: {args.segment_duration}s")
        if args.threshold > 0:
            print(f"Confidence threshold: {args.threshold:.2f}")
        print("="*70)

    # Initialize classifier
    classifier = AudioClassifier(labels=custom_labels)

    # Run CLAP classification
    results = classifier.classify_video_audio(
        video_path=args.video,
        interval_seconds=args.interval,
        segment_duration=args.segment_duration,
        max_duration=args.max_duration,
        verbose=args.verbose
    )

    # Apply confidence threshold filtering
    total_segments = len(results)
    if args.threshold > 0:
        results = [r for r in results if r['confidence'] >= args.threshold]
        if args.verbose and total_segments > len(results):
            print(f"\n  Filtered {total_segments - len(results)} low-confidence events (threshold: {args.threshold:.2f})")

    # Save results
    output_data = {
        'video_path': args.video,
        'timestamp': datetime.now().isoformat(),
        'interval_seconds': args.interval,
        'segment_duration': args.segment_duration,
        'confidence_threshold': args.threshold,
        'labels': classifier.labels,
        'events': results,
        'total_segments_analyzed': total_segments,
        'events_above_threshold': len(results)
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    if args.verbose:
        print(f"\n✓ Audio event classification complete")
        print(f"  Analyzed {total_segments} segments")
        if args.threshold > 0:
            print(f"  Events above threshold: {len(results)}")
        print(f"  Results saved to: {output_path}")

        # Event distribution
        event_counts = {}
        for result in results:
            event = result['primary_event']
            event_counts[event] = event_counts.get(event, 0) + 1

        print(f"\nEvent Distribution:")
        sorted_events = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
        for event, count in sorted_events[:10]:  # Top 10
            pct = (count / len(results)) * 100
            print(f"  {event}: {count} ({pct:.1f}%)")

    return 0


def cmd_setup_regions(args: argparse.Namespace) -> int:
    """Execute UI region setup command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success).
    """
    from src.core.utils import RegionSetupTool
    from dotenv import load_dotenv
    import os

    load_dotenv()

    # Determine output path (special case: default is just ui_regions.json in output/)
    if args.output:
        output_path = args.output
        ensure_output_dir(output_path)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f'ui_regions_{timestamp}.json')

    # Check if output file already exists
    check_output_file_exists(output_path)

    # Initialize tool
    tool = RegionSetupTool(args.image, output_path)

    # If method specified, use it
    if hasattr(args, 'method') and args.method:
        if args.method == 1:
            success = tool.method_1_gpt4v()
        elif args.method == 2:
            success = tool.method_2_interactive()
        elif args.method == 3:
            success = tool.method_3_manual()
        else:
            print(f"Error: Invalid method {args.method}")
            return 1

        return 0 if success else 1

    # Otherwise, show menu
    from src.core.utils.region_setup import HAVE_OPENAI

    while True:
        print("\nChoose a method:")
        print()
        print("  1. GPT-4V Automatic Detection (fastest, most accurate)")
        if not HAVE_OPENAI or not os.environ.get('OPENAI_API_KEY'):
            print("     ⚠ Requires OpenAI API key")
        print()
        print("  2. Interactive Click-and-Drag (visual, intuitive)")
        print()
        print("  3. Manual Coordinate Entry (full control)")
        print()
        print("  q. Quit")
        print()

        choice = input("Select method (1/2/3/q): ").strip()

        if choice == '1':
            success = tool.method_1_gpt4v()
            break
        elif choice == '2':
            success = tool.method_2_interactive()
            break
        elif choice == '3':
            success = tool.method_3_manual()
            break
        elif choice.lower() == 'q':
            print("Quitting")
            return 0
        else:
            print("\nInvalid choice. Try again.")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
