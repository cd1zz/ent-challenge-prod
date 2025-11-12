# CLAUDE.md - Developer Notes for AI Assistants

This file contains information for Claude Code (or other AI assistants) working on this codebase.

## Table of Contents

- [Codebase Overview](#codebase-overview)
- [Architecture](#architecture)
- [Code Style](#code-style)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Recent Changes](#recent-changes)
- [Common Tasks](#common-tasks)
- [Notes for AI Assistants](#notes-for-ai-assistants)

## Codebase Overview

This is a Python toolkit for extracting structured data from gameplay videos using AI models. The codebase has been reorganized to focus on three core extraction functions.

### Key Metrics

- **Total Lines:** ~7,000 lines (3,000 core + 4,000 archived)
- **Core Modules:** 12 files in `src/core/`
- **Entry Scripts:** 6 scripts in `scripts/` + unified `main.py` CLI
- **Language:** Python 3.8+
- **Test Coverage:** 61 passing tests, 5 skipped (GPU/API tests)

## Architecture

### Core Modules (`src/core/`)

The codebase is organized into three functional areas:

#### 1. HUD Extraction (`src/core/hud_extraction/`)
- `gpt4v_hud_extractor.py` (268 lines) - GPT-4V vision-based extraction
- `ui_ocr_extractor.py` (280 lines) - Tesseract OCR-based extraction
- `intelligent_sampler.py` (280 lines) - Cost-optimized frame sampling
- `hud_config.py` (120 lines) - Centralized HUD field definitions

#### 2. Audio Transcription (`src/core/audio_transcription/`)
- `gpt4o_transcribe.py` (604 lines) - GPT-4o audio with speaker diarization
- Supports parallel processing and checkpoint/resume

#### 3. Game Detection (`src/core/game_detection/`)
- `game_segmenter.py` (281 lines) - Detect game boundaries via CLIP
- `game_outcome_detector.py` (496 lines) - Extract placement/kills/damage

#### 4. Utilities (`src/core/utils/`)
- `action_classifier.py` (299 lines) - CLIP zero-shot visual classification
- `audio_classifier.py` (280 lines) - CLAP zero-shot audio classification
- `frame_extractor.py` (180 lines) - Video frame extraction
- `high_quality_frame_extractor.py` (91 lines) - Scene change detection
- `video_slicer.py` (80 lines) - Video splitting utilities
- `region_setup.py` (150 lines) - Interactive UI region annotation

### Entry Point Scripts

| Script | Purpose | Lines | Key Function |
|--------|---------|-------|--------------|
| `main.py` | Unified CLI (recommended) | 900 | All 6 commands |
| `scripts/extract_hud.py` | HUD extraction CLI | 145 | GPT-4V/OCR/PaddleOCR |
| `scripts/extract_hud_paddleocr.py` | PaddleOCR HUD extraction | 220 | Free, high-accuracy OCR |
| `scripts/transcribe_audio.py` | Audio transcription CLI | 168 | GPT-4o audio with diarization |
| `scripts/detect_games.py` | Game detection CLI | 157 | CLIP + segmentation |
| `scripts/detect_game_over_paddleocr.py` | Game over detection | 190 | OCR-based outcome extraction |
| `scripts/setup_regions.py` | UI region configuration | 150 | Interactive region annotation |

### Unified CLI (`main.py`)

The main.py provides a unified interface for all commands:

```bash
python main.py <command> [options]
```

**Commands:**
1. `extract-hud` - HUD extraction (GPT-4V/OCR/PaddleOCR)
2. `transcribe` - Audio transcription with diarization (GPT-4o)
3. `detect-actions` - CLIP visual frame classification (local, GPU)
4. `detect-audio-events` - CLAP audio event classification (local, GPU)
5. `detect-games` - Game boundary segmentation
6. `detect-gameover` - Game over screen detection
7. `setup-regions` - Interactive UI region setup

**Key Features:**
- Default output directory: `output/`
- Automatic directory creation
- Consistent argument parsing
- Helper functions for path management

### Archived Code (`src/archive/`)

Contains ~4,000 lines of behavioral analysis code that is not part of the core functionality:
- `behavioral_analysis/` - Player psychology profiling
- `deprecated/` - Unreliable audio_analyzer module
- `old_scripts/` - Legacy pipeline scripts

**Important:** When working on core functionality, ignore the archived code unless explicitly asked.

## Code Style

### Documentation Standards

The codebase follows **PEP 257** docstring conventions with **Google-style** formatting:

```python
def function_name(arg1: str, arg2: int = 0) -> Dict[str, Any]:
    """Brief one-line summary.

    Longer description explaining the function's purpose and behavior.
    Can span multiple lines with details about the algorithm or approach.

    Args:
        arg1: Description of first argument. Can span multiple lines
            with indentation for continued description.
        arg2: Description of second argument with default value.
            Default is 0.

    Returns:
        Dictionary containing result data with keys:
        - 'field1': Description of field1
        - 'field2': Description of field2

    Raises:
        ValueError: If arg1 is empty.
        FileNotFoundError: If referenced file does not exist.

    Example:
        >>> result = function_name("test", arg2=5)
        >>> print(result['field1'])
        some value

    Note:
        Additional notes about edge cases or important behavior.
    """
    pass
```

### Type Hints

All functions use **PEP 484** type hints:

```python
from typing import Dict, List, Optional, Tuple

def process_data(
    input_path: str,
    options: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict], Dict]:
    """Process data with type-safe signatures."""
    pass
```

### Import Organization

Imports are organized in three groups (PEP 8):

```python
"""Module docstring."""

# Standard library
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Third-party packages
import cv2
import numpy as np
from openai import OpenAI

# Local imports
from .config import settings
from ..utils import helper_function
```

### Module Structure

1. Module-level docstring with usage example
2. Imports (organized as above)
3. Constants (UPPER_CASE)
4. Classes with comprehensive docstrings
5. Module-level functions
6. Convenience/wrapper functions at the end

## Common Patterns

### Environment Variables

All scripts use `python-dotenv` for configuration:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env file automatically

api_key = os.getenv('OPENAI_API_KEY')
```

### Progress Output

Use verbose flags for user-facing progress:

```python
if verbose:
    print(f"Processing {len(items)} items...")
    for i, item in enumerate(items):
        print(f"[{i+1:3d}/{len(items):3d}] {item.name} ✓")
```

### Error Handling

Provide informative error messages:

```python
try:
    result = process_file(path)
except FileNotFoundError:
    print(f"Error: File not found: {path}")
    print("Please check the file path and try again.")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    raise
```

### File I/O

Use pathlib and context managers:

```python
from pathlib import Path

def save_results(data: Dict, output_path: str) -> None:
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
```

### Default Output Paths

All commands in `main.py` use default output paths:

```python
def get_output_path(input_path: str, suffix: str, explicit_output: Optional[str] = None) -> str:
    """Get output file path with default to output/ directory."""
    if explicit_output:
        return explicit_output

    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)

    input_stem = Path(input_path).stem
    return str(output_dir / f"{input_stem}{suffix}")
```

**Default paths:**
- `detect-actions video.mkv` → `output/video_actions.json`
- `detect-audio-events video.mkv` → `output/video_audio.json`
- `detect-games actions.json` → `output/actions_games.json`
- `extract-hud video.mkv` → `output/video_hud.json`
- `transcribe audio.wav` → `output/audio_transcript.json`

## Development Workflow

### Adding New Features

1. **Identify the module** - Which functional area (HUD/audio/game detection)?
2. **Check for dependencies** - What existing utilities can be reused?
3. **Write documentation first** - Module docstring, function signatures
4. **Implement with type hints** - Full PEP 484 annotations
5. **Add examples** - Usage examples in docstrings
6. **Test with verbose output** - Ensure user-facing messages are clear

### Refactoring Guidelines

1. **Don't break existing scripts** - Maintain backward compatibility
2. **Update docstrings** - Keep documentation in sync with code
3. **Preserve archived code** - Don't modify `src/archive/`
4. **Add type hints** - Improve type coverage
5. **Extract common patterns** - Move repeated code to utilities

### Code Review Checklist

- [ ] Module-level docstring with usage example
- [ ] All functions have comprehensive docstrings
- [ ] Complete type hints on all signatures
- [ ] Imports organized (stdlib, third-party, local)
- [ ] No hardcoded paths or API keys
- [ ] Error messages are user-friendly
- [ ] Examples in docstrings work correctly
- [ ] Verbose output for long-running operations

## Dependencies

### Core Dependencies

```
openai>=1.0.0          # GPT-4V, GPT-4o API
python-dotenv>=1.0.0   # .env file support
opencv-python>=4.8.0   # Video processing
pillow>=10.0.0         # Image handling
pytesseract>=0.3.10    # OCR (Tesseract)
paddlepaddle>=2.5.0    # PaddleOCR
paddleocr>=2.7.0       # PaddleOCR
torch>=2.0.0           # PyTorch for CLIP/CLAP (GPU version recommended)
clip @ git+https://github.com/openai/CLIP.git  # CLIP visual model
laion-clap>=1.1.7      # CLAP audio model
pydub>=0.25.1          # Audio format conversion
librosa>=0.10.0        # Audio processing for CLAP
soundfile>=0.12.0      # Audio file I/O for CLAP
```

### System Dependencies

- `ffmpeg` - Video/audio processing and extraction
- `tesseract-ocr` - OCR engine
- CUDA (optional) - GPU acceleration for CLIP/CLAP/PaddleOCR
  - CLIP: Visual action classification
  - CLAP: Audio event classification
  - PaddleOCR: Text recognition

## Testing

### Test Suite Overview

Comprehensive pytest test suite with **61 passing tests** and **5 skipped tests**.

```
✓ 61 tests passed
⊘ 5 tests skipped (GPU/API/integration tests)
✗ 0 tests failed
```

### Test Coverage

**Unit Tests (`tests/unit/`):**
- `test_action_classifier.py` - 9 tests (CLIP classification)
- `test_game_segmenter.py` - 11 tests (game boundary detection)
- `test_intelligent_sampler.py` - 16 tests (frame sampling)
- `test_main.py` - 22 tests (CLI functionality)

**Integration Tests (`tests/integration/`):**
- `test_workflow.py` - 8 tests (end-to-end workflows)

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific category
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests only
pytest -m "not slow"        # Skip slow tests
pytest -m "not api"         # Skip API-dependent tests

# Verbose output
pytest -v                   # Verbose test names
pytest -vv                  # Very verbose
pytest -s                   # Show print statements

# Parallel execution
pip install pytest-xdist
pytest -n auto              # Use all CPUs
```

### Manual Testing

Each script has a `--help` flag with examples:

```bash
python main.py extract-hud --help
python main.py transcribe --help
python main.py detect-games --help
```

### Quick Test Workflow

```bash
# 1. Test on first 5 minutes
python main.py detect-games video.mkv --max-duration 300

# 2. Test transcription on first chunk
python main.py transcribe audio.wav --num-chunks 1

# 3. Test HUD extraction with small sample
python main.py extract-hud video.mkv --method gpt4v \
  --clip test_actions.json
```

### Fixed Issues (from recent testing)

1. **Import Error:** Made `TranscriptAnalyzer` import optional in `intelligent_sampler.py`
2. **Empty Predictions:** Added check for empty predictions in `ActionClassifier.get_primary_action()`
3. **Parameter Names:** Fixed `GameSegmenter` parameter naming (constructor vs attributes)

## Recent Changes

### 2024 Reorganization

**Major restructuring completed:**
- Split monolithic `src/media_analyzer/` into focused `src/core/` modules
- Archived ~4,000 lines of behavioral analysis code
- Created unified `main.py` CLI entry point
- Implemented default output directory system
- Split `detect-games` into two commands (`detect-actions` + `detect-games`)
- Standardized documentation with Google-style docstrings
- Added comprehensive type hints throughout core modules

### Unified CLI (main.py)

Created single entry point for all functionality:
- 6 commands: extract-hud, transcribe, detect-actions, detect-games, detect-gameover, setup-regions
- Default output directory: `output/`
- Automatic path management
- Consistent argument parsing

### Command Separation

Split game detection into two steps for better modularity:

**Before:**
```bash
python main.py detect-games video.mkv -o games.json
```

**After:**
```bash
python main.py detect-actions video.mkv -o actions.json  # Step 1: CLIP
python main.py detect-games actions.json -o games.json    # Step 2: Segmentation
```

**Benefits:**
1. Separation of concerns (CLIP vs segmentation)
2. Reusability (reuse actions for different thresholds)
3. Performance (skip expensive CLIP when tuning)
4. Configuration flexibility (labels in .env, thresholds per-run)

### Custom CLIP Labels

Added three ways to customize action labels:

1. **Replace all labels:** `--labels "label1|label2|label3"`
2. **Add extra labels:** `--add-labels "label4|label5"`
3. **Use defaults:** From `.env` CLIP_LABELS or hardcoded

**Priority:** `--labels` > `--add-labels` > `CLIP_LABELS` (env) > `DEFAULT_LABELS`

### Default Output Directory

All commands now save to `output/` by default:
- Created helper functions: `get_output_path()`, `ensure_output_dir()`
- Output path optional (no longer required)
- Automatic directory creation
- Predictable naming: `{input_stem}_{suffix}.json`

### Environment Variable Migration

Migrated to python-dotenv for all configuration:
- All scripts load `.env` automatically
- Created `.env.example` template
- Added to `.gitignore` for security
- Enhanced error messages for missing API keys

### Documentation Improvements

- Module-level docstrings with usage examples
- Complete Args/Returns/Raises sections
- Type hints on all public functions
- Docstring examples that demonstrate actual usage
- Comprehensive testing documentation

### Audio Event Detection (CLAP Integration)

Added CLAP (Contrastive Language-Audio Pretraining) for zero-shot audio event classification:

**New Module:** `src/core/utils/audio_classifier.py` (280 lines)
- AudioClassifier class for CLAP-based audio classification
- 30+ default gameplay audio event labels
- Automatic audio extraction from video using ffmpeg
- GPU acceleration support (CUDA)
- Custom label support via CLI flags or .env

**New Command:** `detect-audio-events`
```bash
python main.py detect-audio-events video.mkv --interval 2.0 --segment-duration 2.0
```

**Features:**
- Extracts audio segments from video at regular intervals
- Classifies each segment using CLAP zero-shot learning
- Supports custom labels (--labels, --add-labels, CLAP_LABELS env var)
- GPU-accelerated for fast processing (RTX 4070+)
- Outputs timestamped events with confidence scores
- Event distribution summary (top 10 events)

**Technical Details:**
- Uses laion-clap library (v1.1.7+)
- Requires librosa and soundfile for audio processing
- Extracts WAV audio segments via ffmpeg subprocess
- Mirrors ActionClassifier architecture for consistency
- Default 2-second segments at 2-second intervals

## Common Tasks

### Adding a New HUD Field

1. Update `src/core/hud_extraction/hud_config.py`:
   ```python
   def get_gpt4v_field_descriptions() -> Dict[str, str]:
       return {
           # ... existing fields ...
           "new_field": "Description for GPT-4V prompt"
       }
   ```

2. Update prompt in `gpt4v_hud_extractor.py` (if needed)

3. Test extraction on sample video

### Adding a New Game Action Label

1. Update `src/core/utils/action_classifier.py`:
   ```python
   DEFAULT_LABELS = [
       # ... existing labels ...
       "new action description",
   ]
   ```

2. Or add to `.env`:
   ```bash
   CLIP_LABELS=existing labels|new action description
   ```

3. Test CLIP classification on sample frames

4. Update game segmentation logic if needed

### Adding Audio Event Labels

CLAP audio event classification supports the same label customization as CLIP:

1. Update `src/core/utils/audio_classifier.py`:
   ```python
   DEFAULT_LABELS = [
       # ... existing labels ...
       "new audio event description",
   ]
   ```

2. Or add to `.env`:
   ```bash
   CLAP_LABELS=existing labels|new audio event
   ```

3. Or use command-line flags:
   ```bash
   # Replace all labels
   python main.py detect-audio-events video.mkv --labels "gunshots|footsteps|explosions"

   # Add to default labels
   python main.py detect-audio-events video.mkv --add-labels "new event 1|new event 2"
   ```

**Label Priority:** `--labels` > `--add-labels` > `CLAP_LABELS` (env) > `DEFAULT_LABELS`

### Optimizing API Costs

The intelligent sampler (`src/core/hud_extraction/intelligent_sampler.py`) controls frame sampling:

```python
sampler = IntelligentSampler(
    min_interval=4.0,   # Min seconds between extractions
    max_interval=18.0,  # Max seconds without extraction
)
```

Adjust these values to balance cost vs coverage.

### Adding a New Command to main.py

1. Create argument parser:
   ```python
   parser_newcmd = subparsers.add_parser('new-command', help='Description')
   parser_newcmd.add_argument('input', help='Input file')
   parser_newcmd.add_argument('-o', '--output', help='Output path (default: output/...)')
   parser_newcmd.set_defaults(func=cmd_new_command)
   ```

2. Create command handler:
   ```python
   def cmd_new_command(args: argparse.Namespace) -> int:
       """Execute new-command."""
       output_path = get_output_path(args.input, '_suffix.json', args.output)
       ensure_output_dir(output_path)

       # Implementation here

       return 0
   ```

3. Update help text and README

### Updating Test Suite

1. Add test to appropriate file in `tests/unit/` or `tests/integration/`
2. Use fixtures from `conftest.py` where possible
3. Mark appropriately: `@pytest.mark.unit`, `@pytest.mark.slow`, etc.
4. Run tests: `pytest tests/unit/test_your_file.py -v`

## Notes for AI Assistants

When working on this codebase:

### General Guidelines

1. **Follow established patterns** - Look at existing modules for style
2. **Maintain documentation quality** - Don't skip docstrings or type hints
3. **Preserve archived code** - Don't modify `src/archive/` unless asked
4. **Test user-facing changes** - Run scripts manually to verify
5. **Ask before major refactoring** - Discuss structural changes first

### Good Examples to Follow

- `src/core/utils/action_classifier.py` - Well-documented class
- `src/core/game_detection/game_segmenter.py` - Good function documentation
- `src/core/hud_extraction/gpt4v_hud_extractor.py` - Comprehensive module docs
- `main.py` - Clean CLI implementation with helper functions

### Anti-Patterns to Avoid

- ❌ Missing type hints on public functions
- ❌ Minimal docstrings ("Process data" without details)
- ❌ Hardcoded paths or API keys
- ❌ No error handling on file I/O or API calls
- ❌ Breaking changes to existing script interfaces
- ❌ Modifying archived code without explicit request

### Code Organization Rules

1. **Core functionality** goes in `src/core/`
2. **CLI scripts** go in `scripts/` (individual) or `main.py` (unified)
3. **Documentation** goes in `docs/` (detailed guides) or README.md (main)
4. **Tests** go in `tests/unit/` or `tests/integration/`
5. **Archived code** stays in `src/archive/` untouched

### Environment Variables

All configuration should use `.env` via python-dotenv:

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Access variables
api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('GPT4O_TRANSCRIBE_MODEL', 'gpt-4o-audio-preview')  # with default
```

Never hardcode:
- API keys
- File paths (use pathlib)
- Model names (use env vars with defaults)

### Output Path Management

Use the helper functions in `main.py`:

```python
# Get output path with defaults
output_path = get_output_path(args.input, '_suffix.json', args.output)

# Ensure directory exists
ensure_output_dir(output_path)
```

Default pattern: `output/{input_stem}_{suffix}.json`

### CLIP Label Configuration

Understand the priority system:
1. Command-line `--labels` (replaces all)
2. Command-line `--add-labels` (appends to base)
3. Environment `CLIP_LABELS` (from .env)
4. Hardcoded `DEFAULT_LABELS`

When suggesting label changes, recommend appropriate level based on use case:
- Game-specific: Use `--labels` flag
- Project-wide: Use `.env` CLIP_LABELS
- Universal defaults: Update `DEFAULT_LABELS` in code

### CLAP Label Configuration

CLAP audio event classification uses the same priority system as CLIP:
1. Command-line `--labels` (replaces all defaults)
2. Command-line `--add-labels` (appends to defaults)
3. Environment `CLAP_LABELS` (from .env)
4. Hardcoded `DEFAULT_LABELS` in `audio_classifier.py`

**Default Audio Event Labels:**
- Weapon sounds (gunshots, rifle fire, shotgun blast, etc.)
- Movement sounds (footsteps, running, jumping, landing)
- Combat sounds (explosions, melee attacks, shield break)
- Environment sounds (doors, vehicles, ambient noise)
- Communication (voice chat, pings, callouts)

**Example Usage:**
```bash
# Use only custom labels
python main.py detect-audio-events video.mkv --labels "gunshots|footsteps|explosions"

# Add to defaults
python main.py detect-audio-events video.mkv --add-labels "specific weapon type|unique sound"

# Set project-wide in .env
CLAP_LABELS=gunshots|footsteps|explosions|voice chat|reload sounds
```

## Future Improvements

### Planned Enhancements

1. ✅ ~~Unified CLI entry point~~ - **COMPLETED** (main.py)
2. **Requirements File** - Generate `requirements.txt` with exact versions
3. **Unit Tests** - Expand test coverage beyond current 61 tests
4. **Package Structure** - Make installable with `pip install -e .`
5. **Configuration System** - Unified config management beyond .env
6. **Docker Support** - Containerized deployment

### Code Quality Improvements

- [ ] Add mypy type checking
- [ ] Add black code formatting
- [ ] Add flake8 linting
- [ ] Expand pytest test suite (target 80%+ coverage)
- [ ] Add CI/CD pipeline
- [ ] Generate API documentation with Sphinx
- [ ] Create requirements.txt with pinned versions

### Feature Requests

- [ ] Support for more video formats
- [ ] Real-time processing mode
- [ ] Web UI for configuration
- [ ] Batch processing multiple videos
- [ ] Export to different formats (CSV, Parquet, SQLite)
- [ ] Cloud storage integration (S3, GCS)

## Project History

### Original Structure
- Monolithic `src/media_analyzer/` with behavioral analysis
- 6-stage pipeline (`mine_behaviors.py`)
- ~7,000 lines across 31 files
- Complex behavioral profiling features

### 2024 Reorganization
- Focused on 3 core extraction functions
- Split into `src/core/` (3,000 lines, 12 files)
- Archived behavioral analysis (4,000 lines)
- Created clean entry point scripts
- 57% code reduction, 61% fewer files

### Current State
- Clean, focused toolkit
- Unified CLI + individual scripts
- Comprehensive documentation
- 61 passing tests
- Production-ready core features

## Support Resources

### For Users
- `README.md` - Complete user guide
- `python main.py --help` - Command help
- `python main.py <command> --help` - Specific command help
- `scripts/*.py --help` - Individual script help

### For Developers
- This file (CLAUDE.md) - Development guide
- `tests/README.md` - Testing documentation
- Code examples in docstrings
- Inline comments for complex logic

### For AI Assistants
- Follow patterns in good example files
- Check test suite for expected behavior
- Review recent changes section for context
- Consult anti-patterns list before suggesting changes
