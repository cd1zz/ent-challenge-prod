#!/usr/bin/env python3
"""Game Over Detection and Stats Extraction using PaddleOCR.

Detect game over/end screens in gameplay videos and extract final stats:
- Victory/defeat/elimination detection
- Placement (rank)
- Kills, damage, survival time
- Any other visible stats

Uses PaddleOCR for both screen detection and stats extraction.

Usage:
    python detect_game_over_paddleocr.py video.mkv -o game_results.json
"""

import sys
import os
import argparse
import json
import cv2
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.core.game_detection.game_over_detector import (
        GameOverDetector,
        scan_video_for_game_over
    )
except ImportError as e:
    print(f"Error importing game over detector: {e}")
    print("\nMake sure PaddleOCR is installed:")
    print("  pip install paddlepaddle paddleocr")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Detect game over screens and extract stats using PaddleOCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan entire video for game over screens
  python detect_game_over_paddleocr.py video.mkv -o results.json

  # With GPU acceleration
  python detect_game_over_paddleocr.py video.mkv --gpu -o results.json

  # Verbose output showing all detections
  python detect_game_over_paddleocr.py video.mkv -v -o results.json

  # Sample every 5 seconds instead of 2
  python detect_game_over_paddleocr.py video.mkv --interval 5.0 -o results.json

  # Scan only last 5 minutes of video
  python detect_game_over_paddleocr.py video.mkv --start-time 900 -o results.json

Output Format:
  JSON file with all detected game over screens and extracted stats
        """
    )

    parser.add_argument('video', help='Path to gameplay video file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output JSON file')

    parser.add_argument('--interval', '-i', type=float, default=2.0,
                       help='Frame sampling interval in seconds (default: 2.0)')
    parser.add_argument('--start-time', type=float, default=0.0,
                       help='Start scanning at this timestamp (seconds)')
    parser.add_argument('--max-duration', type=float,
                       help='Maximum duration to scan (seconds)')

    parser.add_argument('--lang', default='en',
                       help='Language code for OCR (default: en)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration')
    parser.add_argument('--confidence', type=float, default=0.6,
                       help='Minimum confidence threshold (default: 0.6)')

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--save-frames', action='store_true',
                       help='Save detected game over frames as images')

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    print("=" * 70)
    print("GAME OVER DETECTION (PaddleOCR)")
    print("=" * 70)
    print()

    # Initialize detector
    detector = GameOverDetector(
        lang=args.lang,
        use_gpu=args.gpu,
        confidence_threshold=args.confidence
    )
    print()

    # Get video info
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    print(f"Video: {args.video}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Sampling interval: {args.interval}s")
    if args.start_time > 0:
        print(f"  Start time: {args.start_time}s")
    if args.max_duration:
        print(f"  Max duration: {args.max_duration}s")
    print()

    # Scan video
    print("Scanning video for game over screens...")
    print("-" * 70)

    game_over_events = scan_video_for_game_over(
        video_path=args.video,
        detector=detector,
        interval_seconds=args.interval,
        start_time=args.start_time,
        max_duration=args.max_duration,
        verbose=args.verbose
    )

    # Save frames if requested
    if args.save_frames and game_over_events:
        from src.core.game_detection.game_over_detector import extract_frames

        frame_dir = Path(args.output).parent / 'game_over_frames'
        frame_dir.mkdir(exist_ok=True)

        print(f"\nSaving {len(game_over_events)} game over frames...")
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        for event in game_over_events:
            timestamp = event['timestamp']
            frame_num = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if ret:
                frame_path = frame_dir / f"game_over_{timestamp:.1f}s.png"
                cv2.imwrite(str(frame_path), frame)
                if args.verbose:
                    print(f"  Saved: {frame_path}")

        cap.release()
        print(f"✓ Frames saved to: {frame_dir}")

    print()
    print("-" * 70)
    print(f"✓ Scan complete")
    print()

    # Save results
    output_data = {
        'video_path': args.video,
        'scan_timestamp': datetime.now().isoformat(),
        'settings': {
            'interval': args.interval,
            'start_time': args.start_time,
            'max_duration': args.max_duration,
            'language': args.lang,
            'gpu': args.gpu,
            'confidence_threshold': args.confidence
        },
        'summary': {
            'game_over_events_detected': len(game_over_events),
            'video_duration': duration
        },
        'events': game_over_events
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Results saved to: {args.output}")
    print()

    # Print summary
    print("=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    print(f"Total game over events detected: {len(game_over_events)}")
    print()

    if game_over_events:
        print("Detected events:")
        for i, event in enumerate(game_over_events, 1):
            print(f"\n{i}. Timestamp: {event['timestamp']:.1f}s")
            print(f"   Outcome: {event['outcome']} (confidence: {event['confidence']})")
            if event['stats'].get('placement'):
                print(f"   Placement: #{event['stats']['placement']}")
            if event['stats'].get('kills') is not None:
                print(f"   Kills: {event['stats']['kills']}")
            if event['stats'].get('damage'):
                print(f"   Damage: {event['stats']['damage']}")
            if event['stats'].get('survival_time'):
                print(f"   Survival Time: {event['stats']['survival_time']}")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
