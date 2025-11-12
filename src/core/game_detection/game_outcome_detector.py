"""
Game Outcome Detection using GPT-4V
Analyzes frames before new game starts to determine previous game outcome.
"""

import json
import subprocess
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class GameOutcomeDetector:
    """
    Detects game outcomes by analyzing frames before parachute drops.
    """

    def __init__(self, model: str = "gpt-4o"):
        """Initialize detector with OpenAI model."""
        self.model = model
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def extract_frame(self, video_path: str, timestamp: float, output_path: str):
        """Extract single frame at timestamp using ffmpeg."""
        cmd = [
            'ffmpeg', '-y', '-ss', str(timestamp), '-i', video_path,
            '-vframes', '1', '-f', 'image2', output_path
        ]
        subprocess.run(cmd, capture_output=True, check=False)

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_outcome_frame(self, frame_path: str) -> Dict:
        """
        Analyze a single frame to detect game outcome and stats.

        Returns dict with:
        - outcome: "eliminated", "victory", "unknown"
        - placement: team placement if visible
        - kills: kill count if visible
        - confidence: high/medium/low
        - raw_text: any extracted text
        """
        prompt = """Analyze this Super People battle royale gameplay frame to determine if it shows a game outcome screen.

Look for ELIMINATION screens:
- "Your Team has been eliminated"
- Red/dark overlay with elimination message
- Player death/spectating indicators

Look for VICTORY screens:
- "Winner Winner Chicken Dinner"
- Victory/win messages
- "#1" placement or "1st place"
- Bright/celebratory visuals

Look for PLACEMENT and STATISTICS:
- Team placement (e.g., "#5", "5th place", "Teams: 14")
- Kills count
- Damage dealt
- Survival time
- Any post-game stats

Return JSON with this EXACT format:
{
  "outcome": "eliminated" or "victory" or "unknown",
  "confidence": "high" or "medium" or "low",
  "placement": number or null,
  "kills": number or null,
  "damage": number or null,
  "survival_time": "MM:SS" or null,
  "raw_text": "any visible text from the screen",
  "reasoning": "brief explanation of what you see"
}

IMPORTANT:
- Mark "eliminated" ONLY if you see clear elimination/death screen
- Mark "victory" ONLY if you see clear victory/win screen
- Mark "unknown" for gameplay, loading, menus, or unclear frames
- Extract ALL visible numbers and stats"""

        # Encode image
        base64_image = self.encode_image(frame_path)

        # Call GPT-4V
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        # Parse response
        result = response.choices[0].message.content

        # Extract JSON
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        elif "```" in result:
            result = result.split("```")[1].split("```")[0].strip()

        return json.loads(result)

    def find_game_boundaries(self, actions_file: str, min_parachute_duration: float = 20.0) -> List[Tuple[float, float, float, float]]:
        """
        Find game boundaries by detecting parachute sequences and victory/outcome screens.

        Game flow: Lobby → Helicopter → Parachute → Gameplay → Post-game stats → Lobby → ...

        Args:
            actions_file: Path to CLIP actions JSON
            min_parachute_duration: Minimum duration (seconds) for valid parachute sequence (default: 20.0)

        Returns list of (parachute_start_time, parachute_end_time, outcome_window_start, outcome_window_end) tuples.
        - parachute_start_time: When helicopter/parachute sequence begins (game start)
        - parachute_end_time: When parachute landing ends
        - outcome_window_start: Start of window to search for outcome screen (after landing ends)
        - outcome_window_end: End of window to search for outcome screen (before next helicopter)
        """
        with open(actions_file) as f:
            data = json.load(f)

        results = data.get('results', [])
        if not results:
            return []

        # First pass: detect all parachute sequences with their durations
        # Look for helicopter → parachute sequence as a stronger indicator
        parachute_sequences = []
        in_helicopter_or_parachute = False
        sequence_start = None

        for i, result in enumerate(results):
            timestamp = result.get('timestamp', 0)

            # Check both primary action and all_predictions
            primary_action = result.get('primary_action', '').lower()
            all_preds = result.get('all_predictions', {})
            parachute_prob = all_preds.get('player parachuting and landing', 0.0)
            helicopter_prob = all_preds.get('helicopters visible on screen before parachute drop', 0.0)

            # Consider it part of drop sequence if helicopter OR parachuting
            is_drop_sequence = ('parachut' in primary_action or parachute_prob > 0.5 or
                               'helicopter' in primary_action or helicopter_prob > 0.5)

            if is_drop_sequence and not in_helicopter_or_parachute:
                # Start of drop sequence (helicopter or direct parachute)
                in_helicopter_or_parachute = True
                sequence_start = timestamp

            elif not is_drop_sequence and in_helicopter_or_parachute:
                # End of drop sequence
                sequence_end = timestamp
                duration = sequence_end - sequence_start
                parachute_sequences.append((sequence_start, sequence_end, duration))
                in_helicopter_or_parachute = False
                sequence_start = None

        # Handle case where video ends during drop sequence
        if in_helicopter_or_parachute and sequence_start is not None:
            sequence_end = results[-1].get('timestamp', 0)
            duration = sequence_end - sequence_start
            parachute_sequences.append((sequence_start, sequence_end, duration))

        # Second pass: filter out short sequences and find outcome windows
        # Key insight: Outcome screens appear BEFORE each parachute, showing results of the PREVIOUS game
        # Flow: Game N ends → Post-game stats/outcome → Lobby → Helicopter → Parachute → Game N+1 starts
        #
        # So for each parachute sequence, we look 10-60 seconds BEFORE it starts to find the outcome
        # of the game that just finished.

        valid_parachute_sequences = [(start, end, dur) for start, end, dur in parachute_sequences
                                     if dur >= min_parachute_duration]

        boundaries = []
        for i, (parachute_start, parachute_end, duration) in enumerate(valid_parachute_sequences):
            # For this parachute drop, look BEFORE it for the previous game's outcome
            # Outcome screen typically appears 10-60 seconds before helicopter/parachute
            outcome_window_start = max(0, parachute_start - 60)
            outcome_window_end = parachute_start  # Stop at helicopter/parachute start

            # Avoid overlap with previous game's parachute landing
            if i > 0:
                prev_parachute_end = valid_parachute_sequences[i - 1][1]
                outcome_window_start = max(outcome_window_start, prev_parachute_end)

            boundaries.append((parachute_start, parachute_end, outcome_window_start, outcome_window_end))

        return boundaries

    def detect_game_outcomes(self,
                            video_path: str,
                            actions_file: str,
                            output_file: str,
                            temp_dir: Optional[str] = None,
                            verbose: bool = True) -> List[Dict]:
        """
        Detect outcomes for all games in video.

        Args:
            video_path: Path to video file
            actions_file: Path to CLIP actions JSON
            output_file: Where to save results
            temp_dir: Directory for temporary frames
            verbose: Print progress

        Returns:
            List of game outcome dictionaries
        """
        if verbose:
            print("=" * 70)
            print("GAME OUTCOME DETECTION (GPT-4V)")
            print("=" * 70)

        # Create temp directory
        if temp_dir is None:
            temp_dir = Path(output_file).parent / 'outcome_frames'
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Load existing frame analysis cache if available
        cache_file = Path(output_file).parent / f"{Path(output_file).stem}_frame_cache.json"
        frame_cache = {}
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    frame_cache = json.load(f)
                if verbose:
                    print(f"\n✓ Loaded {len(frame_cache)} cached frame analyses from {cache_file.name}")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Could not load cache: {e}")

        # Find game boundaries
        if verbose:
            print(f"\nFinding parachute sequences in {actions_file}...")

        # Get all sequences for stats
        with open(actions_file) as f:
            data = json.load(f)
        results = data.get('results', [])

        # Count total drop sequences (helicopter + parachute)
        parachute_sequences = []
        in_drop_sequence = False
        drop_start = None
        for result in results:
            timestamp = result.get('timestamp', 0)
            primary_action = result.get('primary_action', '').lower()
            all_preds = result.get('all_predictions', {})
            parachute_prob = all_preds.get('player parachuting and landing', 0.0)
            helicopter_prob = all_preds.get('helicopters visible on screen before parachute drop', 0.0)
            is_drop = ('parachut' in primary_action or parachute_prob > 0.5 or
                      'helicopter' in primary_action or helicopter_prob > 0.5)

            if is_drop and not in_drop_sequence:
                in_drop_sequence = True
                drop_start = timestamp
            elif not is_drop and in_drop_sequence:
                parachute_sequences.append(timestamp - drop_start)
                in_drop_sequence = False
                drop_start = None

        boundaries = self.find_game_boundaries(actions_file)

        if verbose:
            print(f"✓ Found {len(parachute_sequences)} total parachute sequences")
            print(f"✓ Filtered to {len(boundaries)} valid games (duration >= 20s)")

        # Load existing results if available (for resuming)
        game_outcomes = []
        if Path(output_file).exists():
            try:
                with open(output_file) as f:
                    existing_data = json.load(f)
                game_outcomes = existing_data.get('games', [])
                if verbose and game_outcomes:
                    print(f"\n✓ Loaded {len(game_outcomes)} existing game results (resuming)")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Could not load existing results: {e}")

        # Track which games have been processed
        processed_games = {g['game_number'] for g in game_outcomes}

        # Analyze each game
        for i, (parachute_start, parachute_end, outcome_start, outcome_end) in enumerate(boundaries):
            game_num = i + 1

            # Skip if already processed
            if game_num in processed_games:
                if verbose:
                    print(f"\n[Game {game_num}/{len(boundaries)}] ✓ Already processed (skipping)")
                continue
            if verbose:
                print(f"\n[Game {i+1}/{len(boundaries)}] Game: {parachute_start:.1f}s - {parachute_end:.1f}s")
                print(f"  Searching for outcome screen: {outcome_start:.1f}s - {outcome_end:.1f}s")

            # Extract frames at 1-second intervals in the outcome window
            frames_analyzed = []
            outcome_detected = None
            new_analyses = 0
            used_cache_message = False

            # Process frames in OUTCOME window (after game ends, before next game)
            for t in range(int(outcome_start), int(outcome_end), 1):
                frame_path = temp_dir / f"frame_{t}s.png"
                cache_key = f"frame_{t}s"

                # Check cache first
                if cache_key in frame_cache:
                    analysis = frame_cache[cache_key]
                    if verbose and not used_cache_message:
                        print(f"  Using cached analyses...")
                        used_cache_message = True
                else:
                    # Extract frame if needed
                    if not frame_path.exists():
                        self.extract_frame(video_path, t, str(frame_path))

                    try:
                        analysis = self.analyze_outcome_frame(str(frame_path))
                        frame_cache[cache_key] = analysis  # Cache the result
                        new_analyses += 1

                        # Show what GPT-4V detected for each new analysis
                        if verbose:
                            status = "✓" if analysis['outcome'] in ['eliminated', 'victory'] else "○"
                            print(f"  {status} [Frame {t}s] {analysis['outcome']} (conf: {analysis['confidence']})")
                    except Exception as e:
                        if verbose:
                            print(f"  ⚠️  Error analyzing frame at {t}s: {e}")
                        continue

                frames_analyzed.append({
                    'timestamp': t,
                    'analysis': analysis
                })

                # Check if we found an outcome - EARLY STOP if found
                if analysis['outcome'] in ['eliminated', 'victory'] and analysis['confidence'] in ['high', 'medium']:
                    if not outcome_detected:
                        outcome_detected = analysis
                        outcome_detected['timestamp'] = t
                        if verbose:
                            print(f"  🎯 FOUND: {analysis['outcome'].upper()} at {t}s")
                            remaining = int(outcome_end) - t - 1
                            if remaining > 0:
                                print(f"     Stopping early (saved {remaining} API calls)")
                        break  # EARLY STOP - no need to analyze more frames

            if verbose and new_analyses > 0:
                print(f"  Made {new_analyses} new GPT-4V API calls for this game")

            # Show summary of what was found
            if verbose and outcome_detected:
                print(f"  📊 Outcome Summary:")
                print(f"    Result: {outcome_detected['outcome'].upper()}")
                if outcome_detected.get('placement'):
                    print(f"    Placement: #{outcome_detected['placement']}")
                if outcome_detected.get('kills'):
                    print(f"    Kills: {outcome_detected['kills']}")
                if outcome_detected.get('damage'):
                    print(f"    Damage: {outcome_detected['damage']}")
                if outcome_detected.get('raw_text'):
                    preview = outcome_detected['raw_text'][:100]
                    print(f"    Extracted Text: \"{preview}...\"")
            elif verbose:
                print(f"  ⚠️  No clear outcome detected in {len(frames_analyzed)} frames analyzed")

            # Store result
            game_result = {
                'game_number': i + 1,
                'parachute_start': parachute_start,
                'parachute_end': parachute_end,
                'outcome_window': (outcome_start, outcome_end),
                'frames_analyzed': len(frames_analyzed),
                'outcome': outcome_detected if outcome_detected else {
                    'outcome': 'unknown',
                    'confidence': 'low',
                    'reasoning': 'No clear outcome screen detected'
                },
                'all_frames': frames_analyzed
            }

            game_outcomes.append(game_result)

            # Save progress incrementally after each game
            temp_output = {
                'video_path': video_path,
                'total_games': len(game_outcomes),
                'games': game_outcomes,
                'summary': {
                    'victories': sum(1 for g in game_outcomes if g['outcome']['outcome'] == 'victory'),
                    'eliminations': sum(1 for g in game_outcomes if g['outcome']['outcome'] == 'eliminated'),
                    'unknown': sum(1 for g in game_outcomes if g['outcome']['outcome'] == 'unknown')
                }
            }
            with open(output_file, 'w') as f:
                json.dump(temp_output, f, indent=2)

            # Also save frame cache after each game
            with open(cache_file, 'w') as f:
                json.dump(frame_cache, f, indent=2)

            if verbose:
                print(f"  ✓ Progress saved ({len(game_outcomes)}/{len(boundaries)} games completed)")

        # Save frame cache for future runs
        with open(cache_file, 'w') as f:
            json.dump(frame_cache, f, indent=2)
        if verbose:
            print(f"\n✓ Saved {len(frame_cache)} frame analyses to cache")

        # Save results
        output_data = {
            'video_path': video_path,
            'total_games': len(game_outcomes),
            'games': game_outcomes,
            'summary': {
                'victories': sum(1 for g in game_outcomes if g['outcome']['outcome'] == 'victory'),
                'eliminations': sum(1 for g in game_outcomes if g['outcome']['outcome'] == 'eliminated'),
                'unknown': sum(1 for g in game_outcomes if g['outcome']['outcome'] == 'unknown')
            }
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        if verbose:
            print(f"\n{'='*70}")
            print("OUTCOME DETECTION COMPLETE")
            print(f"{'='*70}")
            print(f"\nGames analyzed: {len(game_outcomes)}")
            print(f"  Victories: {output_data['summary']['victories']}")
            print(f"  Eliminations: {output_data['summary']['eliminations']}")
            print(f"  Unknown: {output_data['summary']['unknown']}")
            print(f"\n✓ Results saved to: {output_file}")

        return game_outcomes


def detect_outcomes(video_path: str,
                    actions_file: str,
                    output_file: str,
                    verbose: bool = True) -> List[Dict]:
    """
    Main function to detect game outcomes.

    Args:
        video_path: Path to video file
        actions_file: Path to CLIP actions JSON
        output_file: Where to save results
        verbose: Print progress

    Returns:
        List of game outcome dictionaries
    """
    detector = GameOutcomeDetector()
    return detector.detect_game_outcomes(
        video_path=video_path,
        actions_file=actions_file,
        output_file=output_file,
        verbose=verbose
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 4:
        video = sys.argv[1]
        actions = sys.argv[2]
        output = sys.argv[3]
        detect_outcomes(video, actions, output)
    else:
        print("Usage: python game_outcome_detector.py <video.mkv> <actions.json> <output.json>")
