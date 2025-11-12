#!/usr/bin/env python3
"""
HUD Extraction Script
Extract HUD/UI values from gameplay videos using GPT-4V or OCR.
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

from core.hud_extraction import extract_hud_values_gpt4v, extract_ui_metrics_from_video


def main():
    parser = argparse.ArgumentParser(
        description='Extract HUD/UI values from gameplay video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract using GPT-4V (requires CLIP actions first)
  python extract_hud.py video.mkv --method gpt4v --clip actions.json -o hud.json

  # Extract using GPT-4V with transcript for better sampling
  python extract_hud.py video.mkv --method gpt4v --clip actions.json --transcript transcript.json -o hud.json

  # Extract using OCR (requires calibrated regions file)
  python extract_hud.py video.mkv --method ocr --regions ui_regions.json -o hud.csv

  # Extract using OCR with enhanced preprocessing for low-contrast HUDs
  python extract_hud.py video.mkv --method ocr --regions ui_regions.json --enhanced -o hud.csv
        """
    )

    parser.add_argument('video', help='Path to gameplay video file')
    parser.add_argument('--method', '-m', choices=['gpt4v', 'ocr'], required=True,
                       help='Extraction method: gpt4v (vision AI) or ocr (tesseract)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output file (.json or .csv)')

    # GPT-4V options
    gpt4v_group = parser.add_argument_group('GPT-4V options')
    gpt4v_group.add_argument('--clip', help='Path to CLIP actions JSON (required for gpt4v)')
    gpt4v_group.add_argument('--transcript', help='Path to transcript JSON (optional, improves sampling)')
    gpt4v_group.add_argument('--min-interval', type=float, default=4.0,
                            help='Minimum seconds between extractions (default: 4.0)')
    gpt4v_group.add_argument('--max-interval', type=float, default=18.0,
                            help='Maximum seconds without extraction (default: 18.0)')

    # OCR options
    ocr_group = parser.add_argument_group('OCR options')
    ocr_group.add_argument('--regions', help='Path to UI regions JSON (required for ocr)')
    ocr_group.add_argument('--interval', type=float, default=2.0,
                          help='Sample interval in seconds (default: 2.0)')
    ocr_group.add_argument('--enhanced', action='store_true',
                          help='Use enhanced preprocessing for low-contrast HUDs')
    ocr_group.add_argument('--debug-dir', help='Save preprocessing debug images to this directory')

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if args.method == 'gpt4v':
        # Check for API key
        if not os.environ.get('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY not found in environment")
            print("\nPlease create a .env file with:")
            print("  OPENAI_API_KEY=sk-your-key-here")
            print("\nOr export it in your shell:")
            print("  export OPENAI_API_KEY=sk-your-key-here")
            sys.exit(1)

        if not args.clip:
            print("Error: --clip required for gpt4v method")
            sys.exit(1)
        if not os.path.exists(args.clip):
            print(f"Error: CLIP actions file not found: {args.clip}")
            sys.exit(1)
        if args.transcript and not os.path.exists(args.transcript):
            print(f"Error: Transcript file not found: {args.transcript}")
            sys.exit(1)

        # Run GPT-4V extraction
        print("=" * 70)
        print("HUD EXTRACTION (GPT-4V Vision Model)")
        print("=" * 70)
        result = extract_hud_values_gpt4v(
            video_path=args.video,
            clip_file=args.clip,
            transcript_file=args.transcript,
            output_file=args.output,
            min_interval=args.min_interval,
            max_interval=args.max_interval,
            verbose=args.verbose
        )
        print(f"\n✓ Extracted {len(result['results'])} frames")
        print(f"  Estimated cost: ~${len(result['results']) * 0.01:.2f}")

    elif args.method == 'ocr':
        if not args.regions:
            print("Error: --regions required for ocr method")
            sys.exit(1)
        if not os.path.exists(args.regions):
            print(f"Error: Regions file not found: {args.regions}")
            sys.exit(1)

        # Run OCR extraction
        print("=" * 70)
        print("HUD EXTRACTION (Tesseract OCR)")
        print("=" * 70)
        results = extract_ui_metrics_from_video(
            video_path=args.video,
            interval_seconds=args.interval,
            output_file=args.output,
            regions_file=args.regions,
            enhanced_preprocessing=args.enhanced,
            debug_dir=args.debug_dir,
            verbose=args.verbose
        )
        print(f"\n✓ Extracted {len(results)} frames")

    print(f"  Output saved to: {args.output}")


if __name__ == '__main__':
    main()
