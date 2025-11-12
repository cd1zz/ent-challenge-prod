# Scripts Directory

This directory contains all CLI entry point scripts for the Gameplay Analysis Toolkit.

## Available Scripts

### Core Extraction Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `extract_hud.py` | Extract HUD/UI data | `python scripts/extract_hud.py video.mkv -o hud.json` |
| `extract_hud_paddleocr.py` | Extract HUD using PaddleOCR | `python scripts/extract_hud_paddleocr.py video.mkv -o hud.csv` |
| `transcribe_audio.py` | Transcribe audio with diarization | `python scripts/transcribe_audio.py audio.wav -o transcript.json` |
| `detect_games.py` | Detect game boundaries | `python scripts/detect_games.py video.mkv -o games.json` |
| `detect_game_over_paddleocr.py` | Detect game over screens | `python scripts/detect_game_over_paddleocr.py video.mkv -o results.json` |

### Utility Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_regions.py` | Interactive UI region setup | `python scripts/setup_regions.py video.mkv -o regions.json` |

## Usage

### Option 1: Use main.py (Recommended)

The unified CLI entry point in the root directory:

```bash
# From root directory
python main.py extract-hud video.mkv -o hud.json --method gpt4v
python main.py transcribe audio.wav -o transcript.json

# Game detection (two-step process)
python main.py detect-actions video.mkv -o actions.json  # Step 1: CLIP classification
python main.py detect-games actions.json -o games.json    # Step 2: Game segmentation
```

### Option 2: Call Scripts Directly

```bash
# From root directory
python scripts/extract_hud.py video.mkv -o hud.json --method gpt4v
python scripts/transcribe_audio.py audio.wav -o transcript.json
python scripts/detect_games.py video.mkv -o games.json
```

### Option 3: Run from Scripts Directory

```bash
cd scripts
python extract_hud.py ../video.mkv -o ../hud.json --method gpt4v
python transcribe_audio.py ../audio.wav -o ../transcript.json
python detect_games.py ../video.mkv -o ../games.json
```

## Help

Each script has detailed help information:

```bash
python scripts/extract_hud.py --help
python scripts/transcribe_audio.py --help
python scripts/detect_games.py --help
```

## See Also

- [Main README](../README.md) - Full documentation
- [CLAUDE.md](../CLAUDE.md) - Developer notes
- [Core Modules](../src/core/) - Python package imports
