#!/usr/bin/env python3
"""UI Regions Setup Tool.

Interactive tool to define UI regions for OCR extraction.
Three methods available:
1. GPT-4V Automatic Detection (fastest, most accurate)
2. Interactive Click-and-Drag (visual)
3. Manual Coordinate Entry (precise)

Usage:
    python setup_regions.py reference_frame.png
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.core.utils.region_setup import RegionSetupTool, HAVE_OPENAI
except ImportError as e:
    print(f"Error importing region setup tool: {e}")
    print("\nMake sure required packages are installed:")
    print("  pip install opencv-python")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Setup UI regions for OCR extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Three methods available:

1. GPT-4V Automatic (fastest, most accurate)
   - Requires: OpenAI API key
   - Cost: ~$0.01-0.02 per image
   - Time: ~30 seconds

2. Interactive Click-and-Drag (visual, intuitive)
   - Requires: Nothing
   - Cost: Free
   - Time: ~5-10 minutes

3. Manual Entry (full control)
   - Requires: Image viewer with coords
   - Cost: Free
   - Time: ~15-20 minutes

Example:
  python setup_regions.py reference_frame.png
        """
    )

    parser.add_argument('image', help='Reference frame image')
    parser.add_argument('--output', '-o', default='ui_regions.json',
                       help='Output JSON file (default: ui_regions.json)')
    parser.add_argument('--method', '-m', type=int, choices=[1, 2, 3],
                       help='Force specific method (1=GPT-4V, 2=Interactive, 3=Manual)')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("UI REGIONS SETUP TOOL")
    print("=" * 70)
    print()
    print(f"Image: {args.image}")
    print(f"Output: {args.output}")
    print()

    tool = RegionSetupTool(args.image, args.output)

    # If method specified, use it
    if args.method:
        if args.method == 1:
            success = tool.method_1_gpt4v()
        elif args.method == 2:
            success = tool.method_2_interactive()
        else:
            success = tool.method_3_manual()

        if success:
            print()
            print("=" * 70)
            print("✓ SETUP COMPLETE")
            print("=" * 70)
            print()
            print("Next step: Test extraction")
            print(f"  python extract_hud_paddleocr.py {args.image} \\")
            print(f"    --regions {args.output} \\")
            print("    -v -o test.csv")
        sys.exit(0 if success else 1)

    # Otherwise, show menu
    while True:
        print("Choose a method:")
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
            sys.exit(0)
        else:
            print("\nInvalid choice. Try again.\n")

    if success:
        print()
        print("=" * 70)
        print("✓ SETUP COMPLETE")
        print("=" * 70)
        print()
        print("Next steps:")
        print()
        print("1. Test extraction on this frame:")
        print(f"   python extract_hud_paddleocr.py {args.image} \\")
        print(f"     --regions {args.output} \\")
        print("     -v -o test.csv")
        print()
        print("2. If results look good, run on full video:")
        print(f"   python extract_hud_paddleocr.py video.mkv \\")
        print(f"     --regions {args.output} \\")
        print("     --gpu -o hud_data.csv")
        print()


if __name__ == '__main__':
    main()
