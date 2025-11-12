# Gameplay Analysis Toolkit

A streamlined Python toolkit for extracting structured data from gameplay videos using AI models (GPT-4V, GPT-4o, CLIP, CLAP, OCR).

## Core Features

This toolkit provides **four core functionalities**:

1. **HUD Extraction** - Extract game UI/HUD data using GPT-4V vision or OCR
2. **Audio Transcription** - Transcribe audio with speaker diarization using GPT-4o
3. **Audio Event Detection** - Detect gameplay audio events using CLAP (gunshots, footsteps, etc.)
4. **Game Detection** - Automatically detect game start/stop boundaries and outcomes

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [HUD Extraction](#1-extract-hud-values)
  - [Audio Transcription](#2-transcribe-audio-with-speaker-diarization)
  - [Audio Event Detection](#3-detect-audio-events)
  - [Game Detection](#4-detect-game-boundaries-and-outcomes)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Python API](#python-package-usage)

## Installation

### System Requirements

- **Python:** 3.8 or higher
- **Operating System:** Windows, macOS, or Linux
- **RAM:** 8GB minimum (16GB recommended for video processing)
- **GPU:** Optional, but recommended for faster CLIP and PaddleOCR processing

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr python3-pip
```

#### macOS
```bash
brew install ffmpeg tesseract python3
```

#### Windows
1. Download and install [FFmpeg](https://ffmpeg.org/download.html)
2. Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
3. Add both to your system PATH

### Python Dependencies

#### Standard Installation (CPU)

For most users - works on any system:

```bash
# Clone the repository
git clone <your-repo-url>
cd ent_challenge_version_2

# Install Python dependencies
pip install openai python-dotenv opencv-python pillow pytesseract torch pydub paddlepaddle paddleocr
pip install git+https://github.com/openai/CLIP.git
pip install laion-clap librosa soundfile

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

#### GPU-Accelerated Installation

For users with NVIDIA GPU and CUDA:

```bash
# Clone the repository
git clone <your-repo-url>
cd ent_challenge_version_2

# Check CUDA version
nvidia-smi  # Look for CUDA Version

# Install GPU dependencies
pip install openai python-dotenv opencv-python pillow pytesseract torch pydub paddlepaddle-gpu paddleocr
pip install git+https://github.com/openai/CLIP.git
pip install laion-clap librosa soundfile

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=sk-your-key-here
```

All scripts automatically load environment variables from `.env` using python-dotenv.

### Verify Installation

```bash
# Test basic functionality
python -c "from src.core.hud_extraction import extract_hud_values_gpt4v; print('✓ HUD extraction')"
python -c "from src.core.audio_transcription import transcribe_audio; print('✓ Audio transcription')"
python -c "from src.core.game_detection import segment_video_games; print('✓ Game detection')"

# Test CLI
python main.py --help
```

## Quick Start

### Using main.py (Recommended)

```bash
# Extract HUD data
python main.py extract-hud video.mkv -o hud.json --method gpt4v --clip actions.json

# Transcribe audio
python main.py transcribe audio.wav -o transcript.json

# Detect audio events
python main.py detect-audio-events video.mkv -o audio_events.json

# Detect games (two-step process)
python main.py detect-actions video.mkv -o actions.json  # Step 1: CLIP classification
python main.py detect-games actions.json -o games.json    # Step 2: Game segmentation
```

### Complete Workflow Example

```bash
# Step 1: Transcribe audio
python main.py transcribe gameplay.wav -o transcript.json --max-workers 8

# Step 2: Detect actions with CLIP
python main.py detect-actions gameplay.mkv -o actions.json

# Step 3: Segment games from actions
python main.py detect-games actions.json -o games.json

# Step 4: Extract HUD data
python main.py extract-hud gameplay.mkv -o hud.json \
  --method gpt4v \
  --clip actions.json \
  --transcript transcript.json
```

**Result:**
- `transcript.json` - Who said what and when
- `games.json` - Game boundaries with win/loss/placement/kills
- `hud.json` - Frame-by-frame HUD values (health, ammo, etc.)

## Usage Guide

### 1. Extract HUD Values

Extract HUD/UI data from gameplay videos using three different methods:

#### Method 1: GPT-4V (Highest Accuracy - Recommended)

```bash
# Using main.py
python main.py extract-hud video.mkv -o hud.json \
  --method gpt4v \
  --clip actions.json \
  --transcript transcript.json

# Or call script directly
python scripts/extract_hud.py video.mkv --method gpt4v \
  --clip actions.json \
  --transcript transcript.json \
  -o hud_values.json
```

**Pros:**
- 98-99% accuracy
- No manual calibration needed
- Intelligent sampling (40-60 frames per 10min vs 300+)
- Cost: ~$0.40-0.50 per 10 minutes

**Extracted Data:**
- Player health, armor, ammo
- Weapon and equipment info
- Compass heading
- Game status (kills, teams remaining, soldiers)
- Team health bars

#### Method 2: PaddleOCR (Fast, Free, High Accuracy)

```bash
# Using main.py
python main.py extract-hud video.mkv -o hud.csv \
  --method paddleocr \
  --regions docs/examples/ui_regions.json \
  --gpu

# Or call script directly
python scripts/extract_hud_paddleocr.py video.mkv \
  --regions docs/examples/ui_regions.json \
  --gpu \
  -o hud_metrics.csv
```

**Pros:**
- 95-98% accuracy
- Fast with GPU acceleration
- Free and open source
- Works well with stylized fonts and low-contrast text

**Setup Required:**
1. Define UI regions once using the setup tool
2. Run extraction with regions file

See [UI Region Setup](#ui-region-setup) section below.

#### Method 3: Tesseract OCR (Fastest, Requires Calibration)

```bash
# Using main.py
python main.py extract-hud video.mkv -o hud.csv \
  --method ocr \
  --regions docs/examples/ui_regions.json \
  --enhanced

# Or call script directly
python scripts/extract_hud.py video.mkv --method ocr \
  --regions docs/examples/ui_regions.json \
  --enhanced \
  -o hud_metrics.csv
```

**Pros:**
- Very fast (no API calls)
- Free
- Good for specific, stable HUD elements

**Cons:**
- Requires manual region calibration per game
- Less accurate on transparent/low-contrast HUDs

### 2. Transcribe Audio with Speaker Diarization

```bash
# Using main.py (recommended)
python main.py transcribe audio.wav -o transcript.json
python main.py transcribe audio.wav -o transcript.json --max-workers 8

# Or call script directly
python scripts/transcribe_audio.py audio.wav -o transcript.json
python scripts/transcribe_audio.py audio.wav -o transcript.json --max-workers 8

# Supports OPUS files (auto-converts to WAV)
python main.py transcribe audio.opus -o transcript.json

# Process only first chunk for testing
python scripts/transcribe_audio.py audio.wav -o test.json --num-chunks 1
```

**Output Format (JSON):**
```json
{
  "chunks": [
    {
      "speaker": "speaker_0",
      "text": "Enemy behind the building on the left",
      "start_time": 45.2,
      "end_time": 47.8,
      "timestamp": 45.2
    }
  ]
}
```

**Features:**
- Speaker diarization (identifies different speakers)
- Parallel processing for faster results
- Automatic checkpoint/resume for interrupted transcriptions
- Cost: ~$0.10-0.20 per minute of audio

### 3. Detect Audio Events

Detect gameplay audio events using CLAP (Contrastive Language-Audio Pretraining) zero-shot classification. Runs locally on GPU for fast, free audio analysis.

```bash
# Basic usage (uses default labels)
python main.py detect-audio-events video.mkv

# Specify output path
python main.py detect-audio-events video.mkv -o audio_events.json

# Adjust sampling interval and segment duration
python main.py detect-audio-events video.mkv \
  --interval 2.0 \
  --segment-duration 2.0

# Custom audio event labels (replace defaults)
python main.py detect-audio-events video.mkv \
  --labels "gunshots|footsteps|explosions|voice chat"

# Add extra labels to defaults
python main.py detect-audio-events video.mkv \
  --add-labels "reload sounds|grenade throws"

# Test on first 5 minutes
python main.py detect-audio-events video.mkv --max-duration 300
```

**Output Format (JSON):**
```json
{
  "events": [
    {
      "timestamp": 12.5,
      "segment_duration": 2.0,
      "primary_event": "gunshots and weapon fire",
      "confidence": 0.89,
      "all_predictions": {
        "gunshots and weapon fire": 0.89,
        "automatic rifle firing": 0.72,
        "footsteps on various surfaces": 0.31
      }
    }
  ],
  "summary": {
    "total_segments": 150,
    "top_events": {
      "gunshots and weapon fire": 45,
      "footsteps on various surfaces": 38,
      "ambient game environment": 25
    }
  }
}
```

**Default Audio Event Labels:**
- Weapon sounds (gunshots, rifle fire, shotgun blast, etc.)
- Movement sounds (footsteps, running, jumping, landing)
- Combat sounds (explosions, melee attacks, shield break)
- Environment sounds (doors, vehicles, ambient noise)
- Communication (voice chat, pings, callouts)

**Features:**
- Zero-shot classification (no training required)
- GPU-accelerated (RTX 4070+)
- Free (runs locally, no API costs)
- Customizable event labels
- Automatic audio extraction from video using ffmpeg
- Event distribution summary with top 10 events

**Technical Details:**
- Uses LAION-CLAP model
- Extracts 2-second audio segments at 2-second intervals (default)
- Processes on CUDA GPU for speed
- Outputs timestamped events with confidence scores

### 4. Detect Game Boundaries and Outcomes

#### Option A: Using PaddleOCR (Recommended - No API Cost)

Automatically detects game over screens and extracts stats:

```bash
# Using main.py
python main.py detect-gameover video.mkv -o game_results.json --gpu

# Or call script directly
python scripts/detect_game_over_paddleocr.py video.mkv -o game_results.json
python scripts/detect_game_over_paddleocr.py video.mkv --gpu -o game_results.json
python scripts/detect_game_over_paddleocr.py video.mkv -v -o game_results.json
```

**Automatically detects:**
- Victory/defeat/elimination screens
- Placement/rank (#1, #5, etc.)
- Kills, damage, survival time
- Any visible stats

**Pros:**
- Free (no API costs)
- Fast with GPU
- No manual configuration
- Extracts all visible stats automatically

#### Option B: Using CLIP for Game Segmentation

```bash
# Step 1: Classify video frames with CLIP (uses labels from .env)
python main.py detect-actions video.mkv

# Step 2: Segment games from action classifications
python main.py detect-games output/video_actions.json

# Advanced: Custom labels (replace defaults)
python main.py detect-actions video.mkv \
  --labels "player running|player jumping|player shooting|player dead"

# Advanced: Add extra labels (append to defaults)
python main.py detect-actions video.mkv \
  --add-labels "player swimming|player climbing"

# Advanced: Customize segmentation thresholds
python main.py detect-games output/video_actions.json \
  --parachute-thresh 0.85 \
  --death-thresh 0.60 \
  --min-duration 60.0
```

**How it works:**
1. **detect-actions**: CLIP classifies frames into action categories
   - Labels from: `--labels` (custom) > `--add-labels` (append) > `.env` > defaults
   - "player parachuting and landing" = game start indicator
   - "player dead or spectating" = game end indicator
2. **detect-games**: Analyzes action classifications to find game boundaries
   - Uses state machine with lookahead confirmation
   - Groups frames into individual matches with start/end times

## UI Region Setup

For OCR methods (PaddleOCR and Tesseract), you need to define UI regions where HUD elements appear.

### Quick Setup

```bash
# Using main.py
python main.py setup-regions video.mkv -o ui_regions.json

# Or call script directly
python scripts/setup_regions.py video.mkv -o ui_regions.json
```

This launches an interactive window where you can:
1. Click and drag to define regions
2. Label each region (health, ammo, etc.)
3. Test OCR accuracy in real-time
4. Save configuration to JSON

### Region File Format

UI regions are stored in JSON format:

```json
{
  "health": {
    "x": 50,
    "y": 950,
    "width": 100,
    "height": 40,
    "type": "number",
    "description": "Player health value"
  },
  "ammo": {
    "x": 1750,
    "y": 950,
    "width": 150,
    "height": 50,
    "type": "text",
    "description": "Current weapon ammo"
  }
}
```

### Three Methods to Create Regions

1. **GPT-4V Automatic Detection** (fastest, most accurate)
   - Extract a reference frame: `ffmpeg -ss 30 -i video.mkv -vframes 1 frame.png`
   - Use GPT-4V to automatically annotate UI elements
   - Cost: ~$0.01 per image (one-time setup)

2. **Interactive Annotation Tool** (visual, intuitive)
   - Use the included setup script with visual interface
   - Click and drag to select regions
   - Test OCR results in real-time

3. **Manual Definition** (full control)
   - Extract a reference frame
   - Use an image viewer to identify coordinates
   - Create JSON file manually

For detailed instructions, see the included UI Regions Setup Guide.

## Configuration

### Environment Variables

Create a `.env` file for configuration:

```bash
# OpenAI API Key (required for GPT-4V/GPT-4o features)
OPENAI_API_KEY=sk-your-key-here

# GPT-4o Transcription Settings (optional)
GPT4O_TRANSCRIBE_MODEL=gpt-4o-audio-preview
GPT4O_CHUNK_DURATION=300.0
GPT4O_MAX_WORKERS=4
GPT4O_TEMPERATURE=0.0

# CLIP Settings (optional)
CLIP_LABELS=player looting items|player in combat|player parachuting and landing|player dead or spectating
CLIP_MODEL=ViT-B/32
CLIP_DEVICE=cuda

# CLAP Settings (optional)
CLAP_LABELS=gunshots and weapon fire|footsteps on various surfaces|explosions and blast sounds|voice chat and callouts

# PaddleOCR Settings (optional)
PADDLEOCR_USE_GPU=true
PADDLEOCR_LANG=en
```

### CLIP Action Labels

Customize labels for game action detection:

**Priority System:**
1. `--labels` flag (highest priority) - Replaces all labels
2. `--add-labels` flag - Adds to labels from .env or defaults
3. `CLIP_LABELS` environment variable (in .env)
4. `DEFAULT_LABELS` (hardcoded fallback)

**Examples:**

```bash
# Use default labels (from .env or defaults)
python main.py detect-actions gameplay.mkv

# Replace all labels with custom ones
python main.py detect-actions gameplay.mkv \
  --labels "player running|player jumping|player shooting|player dead"

# Add extra labels to existing ones
python main.py detect-actions gameplay.mkv \
  --add-labels "player swimming|player climbing"
```

### CLAP Audio Event Labels

Customize labels for audio event detection using the same priority system as CLIP:

**Priority System:**
1. `--labels` flag (highest priority) - Replaces all labels
2. `--add-labels` flag - Adds to labels from .env or defaults
3. `CLAP_LABELS` environment variable (in .env)
4. `DEFAULT_LABELS` (hardcoded fallback)

**Examples:**

```bash
# Use default labels (from .env or defaults)
python main.py detect-audio-events gameplay.mkv

# Replace all labels with custom ones
python main.py detect-audio-events gameplay.mkv \
  --labels "gunshots|footsteps|explosions|voice chat"

# Add extra labels to existing ones
python main.py detect-audio-events gameplay.mkv \
  --add-labels "reload sounds|grenade throws"
```

## Advanced Usage

### Quick Testing (First 5 Minutes)

```bash
# Using main.py
python main.py transcribe audio.wav -o test.json --num-chunks 1
python main.py detect-games video.mkv -o test.json --max-duration 300
python main.py extract-hud video.mkv -o test.json \
  --method gpt4v --clip test_actions.json

# Or use scripts directly
python scripts/transcribe_audio.py audio.wav -o test_transcript.json --num-chunks 1
python scripts/detect_games.py video.mkv -o test_games.json --max-duration 300
python scripts/extract_hud.py video.mkv --method gpt4v \
  --clip test_games_actions.json -o test_hud.json
```

### Script Help

Get detailed help for any command:
```bash
# Using main.py
python main.py extract-hud --help
python main.py transcribe --help
python main.py detect-games --help

# Or from individual scripts
python scripts/extract_hud.py --help
python scripts/transcribe_audio.py --help
python scripts/detect_games.py --help
```

### PaddleOCR Tuning

Adjust sensitivity for better OCR results:

```bash
# More sensitive (detects more text, may include noise)
python scripts/extract_hud_paddleocr.py video.mkv \
  --regions ui_regions.json \
  --det-thresh 0.2 \
  -o hud.csv

# Less sensitive (more strict)
python scripts/extract_hud_paddleocr.py video.mkv \
  --regions ui_regions.json \
  --det-thresh 0.5 \
  -o hud.csv
```

**Parameters:**
- `--det-thresh` (0.1-0.7): Detection sensitivity (default: 0.3)
- `--box-thresh` (0.3-0.8): Bounding box strictness (default: 0.5)
- `--batch-size`: Processing batch size (default: 6)
- `--gpu`: Enable GPU acceleration
- `--lang`: Language model (en, ch, japan, korean, etc.)

### Multi-Language Support

PaddleOCR supports 80+ languages:

```bash
# Chinese
python scripts/extract_hud_paddleocr.py video.mkv --lang ch -o hud.csv

# Japanese
python scripts/extract_hud_paddleocr.py video.mkv --lang japan -o hud.csv

# Korean
python scripts/extract_hud_paddleocr.py video.mkv --lang korean -o hud.csv
```

### Game Detection Customization

```bash
# Adjust sampling rate
python main.py detect-actions video.mkv --interval 1.0  # Sample every 1 second

# Adjust segmentation thresholds
python main.py detect-games actions.json \
  --parachute-thresh 0.90 \
  --death-thresh 0.50 \
  --min-duration 120.0
```

## Troubleshooting

### "OPENAI_API_KEY not found"

Create a `.env` file with your API key:
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### "ffmpeg not found"

Install ffmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from ffmpeg.org and add to PATH
```

### "Could not load CLIP model"

Install PyTorch and CLIP:
```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
```

### OCR returns empty/wrong values

1. Use `--enhanced` flag for low-contrast HUDs
2. Recalibrate UI regions using `setup_regions.py`
3. Try PaddleOCR method (more robust than Tesseract)
4. Use GPT-4V method for best accuracy

### PaddleOCR installation issues

```bash
# For CPU version
pip install paddlepaddle

# For GPU version (CUDA required)
pip install paddlepaddle-gpu

# Verify
python -c "import paddle; print(paddle.device.cuda.device_count())"
```

### PaddleOCR GPU not working

```bash
# Check CUDA
nvidia-smi

# Uninstall CPU version
pip uninstall paddlepaddle

# Install GPU version
pip install paddlepaddle-gpu

# Verify
python -c "import paddle; print(paddle.device.cuda.device_count())"
```

### Low OCR accuracy

**For PaddleOCR:**
1. Lower detection threshold: `--det-thresh 0.2`
2. Enable verbose mode to debug: `-v`
3. Check region definitions - are they too small/large?
4. Try different language model if non-English: `--lang ch`

**For Tesseract:**
1. Use `--enhanced` flag
2. Recalibrate regions with more padding
3. Switch to PaddleOCR for better results

### Game detection missing games

1. Check action labels match your game
2. Lower confidence thresholds
3. Use verbose mode to see classifications
4. Adjust sampling interval

## Python Package Usage

You can also import and use the modules directly in your Python code:

```python
# HUD Extraction
from src.core.hud_extraction import extract_hud_values_gpt4v

results = extract_hud_values_gpt4v(
    video_path="gameplay.mp4",
    clip_file="actions.json",
    transcript_file="transcript.json",
    output_file="hud.json"
)

# Audio Transcription
from src.core.audio_transcription import GPT4oTranscriber

transcriber = GPT4oTranscriber()
result = transcriber.transcribe_file(
    audio_path="gameplay.wav",
    parallel=True,
    output_format="json"
)

# Game Detection
from src.core.game_detection import segment_video_games

analysis = segment_video_games(
    actions_file="actions.json",
    output_file="games.json"
)
```

## API Requirements

### OpenAI API Key

Most features require an OpenAI API key. Set it up using a `.env` file (see [Configuration](#configuration)).

### Models Used

- **GPT-4o** - Vision model for HUD extraction and outcome detection
- **GPT-4o Audio** - Audio transcription with diarization
- **CLIP (ViT-B/32)** - Zero-shot visual classification for action detection (local, GPU)
- **CLAP (LAION)** - Zero-shot audio classification for event detection (local, GPU)

### Cost Estimates

| Feature | Cost (per 10 min video) | Speed |
|---------|------------------------|-------|
| HUD (GPT-4V) | ~$0.40-0.50 | Moderate |
| HUD (PaddleOCR) | Free | Fast |
| HUD (Tesseract) | Free | Very Fast |
| Transcription | ~$1.00-2.00 | Moderate (parallel) |
| Audio Events (CLAP) | Free | Fast (GPU) |
| Game Detection (CLIP only) | Free | Fast |
| Game Detection (with outcomes) | ~$0.05-0.10 | Moderate |

## Directory Structure

```
├── main.py                       # Unified CLI entry point (RECOMMENDED)
├── README.md                     # Main documentation
├── CLAUDE.md                     # Developer/AI assistant notes
│
├── scripts/                      # CLI entry point scripts
│   ├── extract_hud.py           # HUD extraction
│   ├── extract_hud_paddleocr.py # PaddleOCR HUD extraction
│   ├── transcribe_audio.py      # Audio transcription
│   ├── detect_games.py          # Game detection
│   ├── detect_game_over_paddleocr.py  # Game over detection
│   ├── setup_regions.py         # UI region setup tool
│   └── README.md                # Scripts documentation
│
├── src/core/                     # Core Python modules
│   ├── hud_extraction/          # GPT-4V and OCR HUD extraction
│   ├── audio_transcription/     # GPT-4o audio transcription
│   ├── game_detection/          # Game segmentation and outcomes
│   └── utils/                   # Supporting utilities
│
├── src/archive/                  # Archived/legacy code
│
├── docs/                         # Additional documentation
│   └── examples/                # Example configuration files
│
├── output/                       # Default output directory
└── reference_images/             # Reference images for documentation
```

## Contributing

This is a streamlined extraction toolkit focused on three core functions. The behavioral analysis components have been archived but can be restored if needed.

## License

MIT License

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Run scripts with `--help` flag for detailed usage
3. Review the included documentation files
4. Verify API keys are configured correctly in `.env`
