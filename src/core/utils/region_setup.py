"""UI Regions Setup Tool.

This module provides three methods for defining UI regions for OCR extraction:
1. GPT-4V Automatic Detection - Fastest, most accurate, requires API key
2. Interactive Click-and-Drag - Visual, intuitive, no dependencies
3. Manual Coordinate Entry - Full control, requires image viewer

Regions are defined as bounding boxes (x, y, width, height) that identify
where specific HUD elements appear in gameplay videos.

Typical usage example:

    from region_setup import RegionSetupTool

    # Initialize with reference frame
    tool = RegionSetupTool('reference_frame.png', 'ui_regions.json')

    # Use GPT-4V automatic detection
    success = tool.method_1_gpt4v()

    # Or use interactive mode
    success = tool.method_2_interactive()

    # Or manual entry
    success = tool.method_3_manual()
"""

import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# Check for optional dependencies
HAVE_OPENAI = False
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    import base64
    HAVE_OPENAI = True
    load_dotenv()
except ImportError:
    pass


class RegionSetupTool:
    """Interactive UI region setup tool for OCR calibration.

    Provides three methods for defining UI regions:
    - Method 1: GPT-4V automatic detection (~$0.01, 30 seconds)
    - Method 2: Interactive click-and-drag (free, 5-10 minutes)
    - Method 3: Manual coordinate entry (free, 15-20 minutes)

    Attributes:
        image_path: Path to reference frame image.
        output_path: Path to save regions JSON file.
        image: Loaded OpenCV image (BGR format).
        regions: Dictionary of defined regions.
    """

    def __init__(self, image_path: str, output_path: str = "ui_regions.json"):
        """Initialize region setup tool.

        Args:
            image_path: Path to reference frame image (screenshot from gameplay).
            output_path: Path to save regions JSON file.

        Raises:
            ValueError: If image cannot be loaded.

        Example:
            >>> tool = RegionSetupTool('reference.png', 'ui_regions.json')
        """
        self.image_path = image_path
        self.output_path = output_path
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.regions = {}
        self.drawing = False
        self.start_point = None
        self.temp_region = None

    def method_1_gpt4v(self) -> bool:
        """Method 1: Use GPT-4V to automatically detect UI regions.

        Uses OpenAI's GPT-4V vision model to automatically identify and
        locate HUD elements in the reference frame.

        Requirements:
        - OPENAI_API_KEY environment variable or .env file
        - openai Python package

        Returns:
            True if regions were successfully detected and saved, False otherwise.

        Cost:
            ~$0.01-0.02 per image

        Example:
            >>> tool = RegionSetupTool('reference.png')
            >>> if tool.method_1_gpt4v():
            ...     print("Regions detected successfully")
        """
        if not HAVE_OPENAI:
            print("Error: OpenAI library not installed")
            print("Install: pip install openai python-dotenv")
            return False

        if not os.environ.get('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY not found")
            print("Add to .env file: OPENAI_API_KEY=sk-your-key-here")
            return False

        print()
        print("=" * 70)
        print("METHOD 1: GPT-4V AUTOMATIC DETECTION")
        print("=" * 70)
        print()
        print("Using GPT-4V to automatically detect UI regions...")
        print("Cost: ~$0.01-0.02")
        print()

        # Encode image
        with open(self.image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        # Create prompt
        prompt = """Analyze this gameplay screenshot and identify all HUD/UI elements.
For each element, provide the bounding box coordinates.

Look for:
- Health/armor bars (with numeric values)
- Ammo counters
- Kill count
- Players/teams remaining
- Weapon name
- Minimap location text
- Inventory/equipment text
- Team member status
- Any other text elements

Return ONLY valid JSON in this exact format:
{
  "element_name": {
    "x": left_edge_pixel,
    "y": top_edge_pixel,
    "width": width_in_pixels,
    "height": height_in_pixels,
    "type": "number",
    "description": "what this element shows"
  }
}

Use "number" for numeric values (health, ammo, kills).
Use "text" for text content (weapon names, locations, status).

Return ONLY the JSON, no other text."""

        try:
            client = OpenAI()

            print("Calling GPT-4V...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }],
                max_tokens=1500
            )

            result = response.choices[0].message.content

            # Extract JSON from markdown if present
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            # Parse JSON
            regions = json.loads(result)

            print(f"✓ GPT-4V detected {len(regions)} UI elements:")
            for name, region in regions.items():
                print(f"  - {name}: {region.get('description', 'N/A')}")

            # Save regions
            with open(self.output_path, 'w') as f:
                json.dump(regions, f, indent=2)

            print()
            print(f"✓ Regions saved to: {self.output_path}")

            # Visualize
            self.visualize_regions(regions)

            return True

        except Exception as e:
            print(f"Error: {e}")
            return False

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for interactive region selection.

        Internal method for OpenCV mouse event handling.

        Args:
            event: OpenCV mouse event type.
            x: Mouse X coordinate.
            y: Mouse Y coordinate.
            flags: Additional flags.
            param: User data (unused).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.image.copy()

                # Draw existing regions
                for name, region in self.regions.items():
                    rx, ry, rw, rh = region['x'], region['y'], region['width'], region['height']
                    cv2.rectangle(img_copy, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                    cv2.putText(img_copy, name, (rx, ry-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw current rectangle
                cv2.rectangle(img_copy, self.start_point, (x, y), (255, 0, 0), 2)
                cv2.imshow('Setup UI Regions', img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

            # Calculate region
            x1, y1 = self.start_point
            x2, y2 = x, y

            region_x = min(x1, x2)
            region_y = min(y1, y2)
            region_w = abs(x2 - x1)
            region_h = abs(y2 - y1)

            if region_w > 10 and region_h > 10:
                self.temp_region = {
                    'x': region_x,
                    'y': region_y,
                    'width': region_w,
                    'height': region_h
                }

                print(f"\n✓ Region selected: ({region_x}, {region_y}, {region_w}x{region_h})")
                print("Press SPACE to save this region (or ESC to discard)")

    def method_2_interactive(self) -> bool:
        """Method 2: Interactive click-and-drag region selection.

        Opens an interactive window where you can click and drag to select
        regions visually.

        Instructions:
        1. Click and drag to select a region
        2. Release mouse button
        3. Press SPACE to save region
        4. Enter region details in terminal
        5. Press 's' when done to save all regions
        6. Press 'q' to quit

        Returns:
            True if regions were saved, False otherwise.

        Example:
            >>> tool = RegionSetupTool('reference.png')
            >>> if tool.method_2_interactive():
            ...     print("Regions saved")
        """
        print()
        print("=" * 70)
        print("METHOD 2: INTERACTIVE CLICK-AND-DRAG")
        print("=" * 70)
        print()
        print("Instructions:")
        print("  1. Click and drag to select a region")
        print("  2. Release mouse button")
        print("  3. Press SPACE to save region")
        print("  4. Enter region details in terminal")
        print("  5. Press 's' when done to save all regions")
        print("  6. Press 'q' to quit")
        print()
        print("Starting interactive mode...")
        print()

        # Create window
        cv2.namedWindow('Setup UI Regions')
        cv2.setMouseCallback('Setup UI Regions', self.mouse_callback)

        img_display = self.image.copy()
        cv2.imshow('Setup UI Regions', img_display)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):  # Space - save current region
                if self.temp_region:
                    name = input("\nEnter region name (e.g., 'health', 'ammo'): ").strip()

                    if name:
                        type_input = input("Type (number/text) [number]: ").strip().lower()
                        region_type = type_input if type_input in ['number', 'text'] else 'number'

                        description = input("Description (optional): ").strip()

                        self.regions[name] = {
                            **self.temp_region,
                            'type': region_type,
                            'description': description or f"{name} value"
                        }

                        print(f"✓ Saved region '{name}'")
                        print()

                    self.temp_region = None

                    # Redraw with all regions
                    img_display = self.image.copy()
                    for rname, region in self.regions.items():
                        rx, ry, rw, rh = region['x'], region['y'], region['width'], region['height']
                        cv2.rectangle(img_display, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                        cv2.putText(img_display, rname, (rx, ry-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow('Setup UI Regions', img_display)

            elif key == ord('s'):  # Save and exit
                if self.regions:
                    with open(self.output_path, 'w') as f:
                        json.dump(self.regions, f, indent=2)
                    print(f"\n✓ Saved {len(self.regions)} regions to: {self.output_path}")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("\nNo regions defined yet. Draw some regions first.")

            elif key == ord('q'):  # Quit without saving
                print("\nQuitting without saving")
                cv2.destroyAllWindows()
                return False

            elif key == 27:  # ESC - cancel current region
                if self.temp_region:
                    print("\nRegion discarded")
                    self.temp_region = None
                    img_display = self.image.copy()
                    for rname, region in self.regions.items():
                        rx, ry, rw, rh = region['x'], region['y'], region['width'], region['height']
                        cv2.rectangle(img_display, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
                        cv2.putText(img_display, rname, (rx, ry-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imshow('Setup UI Regions', img_display)

    def method_3_manual(self) -> bool:
        """Method 3: Manual coordinate entry.

        Allows you to manually enter bounding box coordinates using an
        external image viewer to find pixel positions.

        Instructions:
        1. Open the reference image in an image viewer that shows coordinates
        2. Note the pixel coordinates for each HUD element
        3. Enter the coordinates when prompted

        Recommended viewers:
        - GIMP: Shows coords in bottom-left
        - Photoshop: Info panel
        - Paint: Bottom status bar
        - Preview (Mac): Tools → Show Inspector

        Returns:
            True if regions were saved, False otherwise.

        Example:
            >>> tool = RegionSetupTool('reference.png')
            >>> if tool.method_3_manual():
            ...     print("Regions saved")
        """
        print()
        print("=" * 70)
        print("METHOD 3: MANUAL COORDINATE ENTRY")
        print("=" * 70)
        print()
        print(f"Image size: {self.image.shape[1]}x{self.image.shape[0]} (width x height)")
        print()
        print("Open the image in an image viewer that shows pixel coordinates:")
        print(f"  - GIMP: Shows coords in bottom-left")
        print(f"  - Photoshop: Info panel")
        print(f"  - Paint: Bottom status bar")
        print(f"  - Preview (Mac): Tools → Show Inspector")
        print()
        print(f"Image: {self.image_path}")
        print()

        # Show image
        cv2.imshow('Reference Image', self.image)
        print("Displaying image in window (close when done looking)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print()
        print("Now enter region coordinates manually:")
        print()

        while True:
            print("-" * 70)
            name = input("Region name (or 'done' to finish): ").strip()

            if name.lower() == 'done':
                break

            if not name:
                continue

            try:
                x = int(input("  X (left edge): "))
                y = int(input("  Y (top edge): "))
                width = int(input("  Width: "))
                height = int(input("  Height: "))

                type_input = input("  Type (number/text) [number]: ").strip().lower()
                region_type = type_input if type_input in ['number', 'text'] else 'number'

                description = input("  Description (optional): ").strip()

                self.regions[name] = {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'type': region_type,
                    'description': description or f"{name} value"
                }

                print(f"  ✓ Added region '{name}'")

            except ValueError as e:
                print(f"  Error: Invalid input - {e}")
                continue

        if self.regions:
            with open(self.output_path, 'w') as f:
                json.dump(self.regions, f, indent=2)

            print()
            print(f"✓ Saved {len(self.regions)} regions to: {self.output_path}")

            # Visualize
            self.visualize_regions(self.regions)

            return True
        else:
            print("\nNo regions created")
            return False

    def visualize_regions(self, regions: Dict) -> None:
        """Show regions overlaid on image and save visualization.

        Args:
            regions: Dictionary of region definitions.

        Saves a visualization image with regions drawn as green rectangles
        with labels, and displays it in a window.

        Example:
            >>> tool.visualize_regions(tool.regions)
        """
        print()
        print("Generating visualization...")

        img_vis = self.image.copy()

        for name, region in regions.items():
            x, y, w, h = region['x'], region['y'], region['width'], region['height']

            # Draw rectangle
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Draw label
            cv2.putText(img_vis, name, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save visualization
        vis_path = self.output_path.replace('.json', '_visualization.png')
        cv2.imwrite(vis_path, img_vis)

        print(f"✓ Visualization saved to: {vis_path}")

        # Show visualization
        cv2.imshow('UI Regions Visualization', img_vis)
        print()
        print("Press any key to close visualization...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
