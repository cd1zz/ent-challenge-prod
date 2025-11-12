#!/usr/bin/env python3
"""
Video Slicer Utility
Lossless video slicing using ffmpeg stream copying.
"""

import subprocess
import os
from pathlib import Path
from typing import List, Tuple, Optional
import json


def get_video_duration(video_path: str) -> float:
    """
    Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'json',
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except Exception as e:
        raise RuntimeError(f"Failed to get video duration: {e}")


def slice_video(
    input_path: str,
    output_path: str,
    start_time: float,
    duration: Optional[float] = None,
    verbose: bool = True
) -> str:
    """
    Slice video using ffmpeg stream copying (lossless, fast).

    Args:
        input_path: Input video file
        output_path: Output video file
        start_time: Start time in seconds
        duration: Duration in seconds (None = to end of file)
        verbose: Print progress

    Returns:
        Path to output file
    """
    if verbose:
        print(f"Slicing video: {input_path}")
        print(f"  Start: {start_time:.1f}s")
        if duration:
            print(f"  Duration: {duration:.1f}s")
        else:
            print(f"  Duration: to end")

    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output
        '-ss', str(start_time),  # Start time
        '-i', input_path,  # Input file
    ]

    if duration:
        cmd.extend(['-t', str(duration)])  # Duration

    cmd.extend([
        '-c', 'copy',  # Stream copy (lossless, fast)
        '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
        output_path
    ])

    # Run ffmpeg
    try:
        if verbose:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True)
        else:
            result = subprocess.run(cmd, capture_output=True, check=True)

        if verbose:
            print(f"✓ Created slice: {output_path}")

        return output_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e}")


def auto_slice_video(
    input_path: str,
    chunk_duration: float,
    output_dir: Optional[str] = None,
    max_chunks: Optional[int] = None,
    verbose: bool = True
) -> List[Tuple[str, float, float]]:
    """
    Automatically slice video into chunks.

    Args:
        input_path: Input video file
        chunk_duration: Duration of each chunk in seconds
        output_dir: Output directory (default: same as input)
        max_chunks: Maximum number of chunks (None = all)
        verbose: Print progress

    Returns:
        List of (chunk_path, start_time, duration) tuples
    """
    input_path = Path(input_path)

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get total duration
    total_duration = get_video_duration(str(input_path))

    if verbose:
        print("=" * 70)
        print("AUTO-SLICE VIDEO")
        print("=" * 70)
        print(f"Input: {input_path}")
        print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
        print(f"Chunk duration: {chunk_duration:.1f}s ({chunk_duration/60:.1f}min)")

    # Calculate chunks
    num_chunks = int(total_duration / chunk_duration) + (1 if total_duration % chunk_duration > 0 else 0)

    if max_chunks and num_chunks > max_chunks:
        num_chunks = max_chunks
        if verbose:
            print(f"Limiting to {max_chunks} chunks (max_chunks specified)")

    if verbose:
        print(f"Creating {num_chunks} chunks...")
        print()

    chunks = []

    for i in range(num_chunks):
        start_time = i * chunk_duration

        # Last chunk might be shorter
        if start_time + chunk_duration > total_duration:
            duration = total_duration - start_time
        else:
            duration = chunk_duration

        # Generate output filename
        chunk_name = f"{input_path.stem}_chunk{i:03d}{input_path.suffix}"
        chunk_path = output_dir / chunk_name

        if verbose:
            print(f"Chunk {i+1}/{num_chunks}: {start_time:.1f}s - {start_time+duration:.1f}s")

        # Slice video
        slice_video(
            str(input_path),
            str(chunk_path),
            start_time,
            duration,
            verbose=False
        )

        chunks.append((str(chunk_path), start_time, duration))

        if verbose:
            print(f"  ✓ {chunk_path}")
            print()

    if verbose:
        print("=" * 70)
        print(f"✓ Created {len(chunks)} video chunks")
        print("=" * 70)

    return chunks


def merge_chunk_results(chunk_results: List[dict], output_path: str, verbose: bool = True):
    """
    Merge results from multiple chunks into single output file.

    Args:
        chunk_results: List of result dictionaries from each chunk
        output_path: Path to merged output file
        verbose: Print progress
    """
    if not chunk_results:
        raise ValueError("No chunk results to merge")

    # Determine output format from extension
    output_path = Path(output_path)

    if output_path.suffix == '.json':
        # Merge JSON results
        merged = {
            'chunks': len(chunk_results),
            'total_duration': sum(r.get('duration', 0) for r in chunk_results),
            'results': []
        }

        # Combine all results, adjusting timestamps
        time_offset = 0.0
        for i, chunk in enumerate(chunk_results):
            chunk_start = chunk.get('start_time', time_offset)

            if 'results' in chunk:
                for item in chunk['results']:
                    # Adjust timestamp
                    if 'timestamp' in item:
                        item['timestamp'] += chunk_start
                    merged['results'].append(item)

            time_offset += chunk.get('duration', 0)

        # Write merged JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(merged, f, indent=2)

        if verbose:
            print(f"✓ Merged {len(chunk_results)} chunks → {output_path}")
            print(f"  Total results: {len(merged['results'])}")

    elif output_path.suffix == '.csv':
        # Merge CSV results
        import pandas as pd

        dfs = []
        time_offset = 0.0

        for i, chunk in enumerate(chunk_results):
            if 'csv_path' in chunk:
                df = pd.read_csv(chunk['csv_path'])

                # Adjust timestamps
                if 'timestamp' in df.columns:
                    df['timestamp'] += chunk.get('start_time', time_offset)

                dfs.append(df)
                time_offset += chunk.get('duration', 0)

        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(output_path, index=False)

            if verbose:
                print(f"✓ Merged {len(dfs)} chunks → {output_path}")
                print(f"  Total rows: {len(merged_df)}")

    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) < 3:
        print("Usage: python video_slicer.py <input.mkv> <chunk_duration_seconds>")
        sys.exit(1)

    input_video = sys.argv[1]
    chunk_duration = float(sys.argv[2])

    chunks = auto_slice_video(input_video, chunk_duration)
    print(f"\nCreated {len(chunks)} chunks:")
    for path, start, duration in chunks:
        print(f"  {path} ({start:.1f}s - {start+duration:.1f}s)")
