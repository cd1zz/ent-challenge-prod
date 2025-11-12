#!/usr/bin/env python3
"""
Game Detection Script
Detect game start/stop boundaries and outcomes from gameplay videos.
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

from core.game_detection import segment_video_games, detect_outcomes
from core.utils import classify_video_actions


def main():
    parser = argparse.ArgumentParser(
        description='Detect game boundaries and outcomes from gameplay video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: classify actions, segment games, detect outcomes
  python detect_games.py video.mkv -o games.json --detect-outcomes

  # Just segment games (requires CLIP actions already generated)
  python detect_games.py --clip actions.json -o games.json

  # Just detect outcomes (requires CLIP actions and wants GPT-4V analysis)
  python detect_games.py video.mkv --clip actions.json --outcomes-only -o outcomes.json

Notes:
  - Game start detected by parachute drop (CLIP classification)
  - Game end detected by death/victory screens (CLIP classification)
  - Outcomes extracted using GPT-4V (placement, kills, damage)
  - Requires OPENAI_API_KEY for GPT-4V outcome detection
        """
    )

    parser.add_argument('video', nargs='?', help='Path to gameplay video file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output JSON file')

    # Input options
    parser.add_argument('--clip', help='Path to existing CLIP actions JSON (skip classification)')

    # Processing options
    parser.add_argument('--detect-outcomes', action='store_true',
                       help='Detect game outcomes using GPT-4V (requires OPENAI_API_KEY)')
    parser.add_argument('--outcomes-only', action='store_true',
                       help='Only detect outcomes (skip segmentation)')

    # CLIP options (if running classification)
    clip_group = parser.add_argument_group('CLIP classification options')
    clip_group.add_argument('--interval', type=float, default=2.0,
                           help='Frame sampling interval in seconds (default: 2.0)')
    clip_group.add_argument('--max-duration', type=float,
                           help='Maximum video duration to process in seconds')

    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Validate inputs
    if args.outcomes_only:
        if not args.video or not args.clip:
            print("Error: --outcomes-only requires both video and --clip")
            sys.exit(1)
        if not os.path.exists(args.video):
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        if not os.path.exists(args.clip):
            print(f"Error: CLIP actions not found: {args.clip}")
            sys.exit(1)
    elif args.clip:
        # Segmentation only
        if not os.path.exists(args.clip):
            print(f"Error: CLIP actions not found: {args.clip}")
            sys.exit(1)
    else:
        # Full pipeline
        if not args.video:
            print("Error: video required (unless using --clip)")
            sys.exit(1)
        if not os.path.exists(args.video):
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)

    # Step 1: Run CLIP classification if needed
    actions_file = args.clip
    if not actions_file:
        print("=" * 70)
        print("STEP 1: ACTION CLASSIFICATION (CLIP)")
        print("=" * 70)

        actions_file = args.output.replace('.json', '_actions.json')
        max_frames = None
        if args.max_duration:
            max_frames = int(args.max_duration / args.interval)

        classify_video_actions(
            args.video,
            interval_seconds=args.interval,
            max_frames=max_frames,
            output_file=actions_file,
            verbose=args.verbose
        )
        print(f"\n✓ Actions saved to: {actions_file}\n")

    # Step 2: Segment games (unless outcomes-only)
    games_file = None
    if not args.outcomes_only:
        print("=" * 70)
        print("STEP 2: GAME SEGMENTATION")
        print("=" * 70)

        games_file = args.output
        analysis = segment_video_games(
            actions_file=actions_file,
            output_file=games_file
        )
        print(f"\n✓ Segmentation saved to: {games_file}\n")

        # Show summary
        print(f"Detected {analysis['total_games']} games:")
        for game in analysis.get('games', []):
            duration_min = int(game['duration'] // 60)
            duration_sec = int(game['duration'] % 60)
            print(f"  Game {game['game_number']}: {duration_min:02d}:{duration_sec:02d} - {game['outcome']}")

    # Step 3: Detect outcomes if requested
    if args.detect_outcomes or args.outcomes_only:
        if not args.video:
            print("\nError: --detect-outcomes requires video file")
            sys.exit(1)

        if not os.environ.get('OPENAI_API_KEY'):
            print("\nError: OPENAI_API_KEY required for outcome detection")
            print("\nPlease create a .env file with:")
            print("  OPENAI_API_KEY=sk-your-key-here")
            print("\nOr export it in your shell:")
            print("  export OPENAI_API_KEY=sk-your-key-here")
            sys.exit(1)

        print("\n" + "=" * 70)
        print("STEP 3: OUTCOME DETECTION (GPT-4V)")
        print("=" * 70)

        outcomes_file = args.output if args.outcomes_only else args.output.replace('.json', '_outcomes.json')

        detect_outcomes(
            video_path=args.video,
            actions_file=actions_file,
            output_file=outcomes_file,
            verbose=args.verbose
        )
        print(f"\n✓ Outcomes saved to: {outcomes_file}")

    print("\n" + "=" * 70)
    print("✓ DETECTION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
