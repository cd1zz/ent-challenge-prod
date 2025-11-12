#!/usr/bin/env python3
"""PaddleOCR HUD Extraction Script.

Extract HUD/UI values from gameplay videos using PaddleOCR.
PaddleOCR supports 80+ languages and provides better accuracy than Tesseract
for gaming HUDs, especially for complex text layouts and low-contrast scenarios.

Installation:
    pip install paddlepaddle paddleocr opencv-python pillow

Usage:
    python extract_hud_paddleocr.py video.mkv --regions ui_regions.json -o hud.csv
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.core.hud_extraction.paddleocr_extractor import (
        PaddleOCRExtractor,
        load_regions_from_file,
        extract_hud_data
    )
except ImportError as e:
    print(f"Error importing PaddleOCR extractor: {e}")
    print("\nMake sure PaddleOCR is installed:")
    print("  pip install paddlepaddle paddleocr")
    print("\nFor GPU support:")
    print("  pip install paddlepaddle-gpu paddleocr")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Extract HUD/UI values using PaddleOCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction with regions file
  python extract_hud_paddleocr.py video.mkv --regions ui_regions.json -o hud.csv

  # With GPU acceleration (requires paddlepaddle-gpu)
  python extract_hud_paddleocr.py video.mkv --regions ui_regions.json --gpu -o hud.csv

  # Sample every 5 seconds
  python extract_hud_paddleocr.py video.mkv --regions ui_regions.json --interval 5 -o hud.csv

  # Process only first 60 seconds
  python extract_hud_paddleocr.py video.mkv --regions ui_regions.json --max-duration 60 -o hud.csv

  # With verbose output showing each extraction
  python extract_hud_paddleocr.py video.mkv --regions ui_regions.json -v -o hud.csv

  # Multi-language support (Chinese example)
  python extract_hud_paddleocr.py video.mkv --regions ui_regions.json --lang ch -o hud.csv

PaddleOCR vs Tesseract:
  + Better accuracy on gaming HUDs
  + Faster processing
  + Better with low-contrast text
  + Supports 80+ languages
  + Works with stylized fonts
  - Larger model size (~100MB)
  - GPU recommended for speed

Supported Languages:
  en, ch, fr, de, es, pt, ru, ar, hi, ja, ko, ta, te, and 70+ more
  See: https://github.com/PaddlePaddle/PaddleOCR#supported-languages

Installation:
  CPU: pip install paddlepaddle paddleocr
  GPU: pip install paddlepaddle-gpu paddleocr
        """
    )

    parser.add_argument('video', help='Path to gameplay video file')
    parser.add_argument('--regions', '-r', required=True,
                       help='Path to UI regions JSON file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output file (.csv or .json)')

    # Extraction options
    parser.add_argument('--interval', '-i', type=float, default=2.0,
                       help='Frame sampling interval in seconds (default: 2.0)')
    parser.add_argument('--max-duration', type=float,
                       help='Maximum video duration to process in seconds')

    # PaddleOCR options
    parser.add_argument('--lang', default='en',
                       help='Language code: en, ch, fr, de, es, pt, ru, etc. (default: en)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration (requires paddlepaddle-gpu)')
    parser.add_argument('--det-thresh', type=float, default=0.3,
                       help='Detection threshold - lower = more sensitive (default: 0.3)')
    parser.add_argument('--box-thresh', type=float, default=0.5,
                       help='Box threshold for text detection (default: 0.5)')
    parser.add_argument('--batch-size', type=int, default=6,
                       help='Batch size for recognition (default: 6)')

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output showing each extraction')
    parser.add_argument('--show-paddle-log', action='store_true',
                       help='Show PaddleOCR internal logs')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.regions):
        print(f"Error: Regions file not found: {args.regions}")
        sys.exit(1)

    # Load regions
    print("=" * 70)
    print("PADDLEOCR HUD EXTRACTION")
    print("=" * 70)
    print()

    regions = load_regions_from_file(args.regions)
    print(f"Loaded {len(regions)} UI regions from {args.regions}")
    for name in regions.keys():
        print(f"  - {name}")
    print()

    # Initialize PaddleOCR extractor
    extractor = PaddleOCRExtractor(
        regions=regions,
        lang=args.lang,
        use_gpu=args.gpu,
        use_angle_cls=True,
        show_log=args.show_paddle_log,
        det_db_thresh=args.det_thresh,
        det_db_box_thresh=args.box_thresh,
        rec_batch_num=args.batch_size
    )
    print()

    # Calculate max frames
    max_frames = None
    if args.max_duration:
        max_frames = int(args.max_duration / args.interval)
        print(f"Processing first {args.max_duration}s (~{max_frames} frames)")
        print()

    # Extract metrics from video
    df = extract_hud_data(
        video_path=args.video,
        extractor=extractor,
        interval_seconds=args.interval,
        max_frames=max_frames,
        output_file=args.output,
        verbose=args.verbose
    )

    print()

    # Show statistics
    print("Extraction Statistics:")
    print("-" * 70)
    for region_name in regions.keys():
        conf_col = f"{region_name}_confidence"
        if conf_col in df.columns:
            avg_conf = df[conf_col].mean()
            non_empty = (df[region_name] != "").sum()
            print(f"  {region_name}:")
            print(f"    Avg confidence: {avg_conf:.2f}")
            print(f"    Non-empty: {non_empty}/{len(df)} ({non_empty/len(df)*100:.1f}%)")

    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print()
    print("Next steps:")
    print(f"  1. Review extracted data: {args.output}")
    print(f"  2. Check confidence scores for accuracy")
    print(f"  3. Adjust --det-thresh if too sensitive/insensitive")
    print(f"  4. Use --gpu for faster processing on large videos")


if __name__ == '__main__':
    main()
