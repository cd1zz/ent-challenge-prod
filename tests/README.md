# Test Suite

Comprehensive test suite for the Gameplay Analysis Toolkit.

## Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests for individual modules
│   ├── test_action_classifier.py
│   ├── test_game_segmenter.py
│   ├── test_intelligent_sampler.py
│   └── test_main.py
├── integration/             # Integration tests across modules
│   └── test_workflow.py
└── fixtures/                # Test data files
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip tests requiring API keys
pytest -m "not api"
```

### Run Specific Test Files

```bash
# Single file
pytest tests/unit/test_action_classifier.py

# Specific test class
pytest tests/unit/test_action_classifier.py::TestActionClassifier

# Specific test function
pytest tests/unit/test_action_classifier.py::TestActionClassifier::test_init_default_labels
```

### Verbose Output

```bash
pytest -v                    # Verbose test names
pytest -vv                   # Very verbose with more details
pytest -s                    # Show print statements
```

### Coverage Report

```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# View HTML report
# Open htmlcov/index.html in browser
```

## Test Markers

Tests are organized with markers for selective execution:

- `@pytest.mark.unit` - Unit tests (fast, no external dependencies)
- `@pytest.mark.integration` - Integration tests (multiple modules)
- `@pytest.mark.slow` - Tests taking >1 second
- `@pytest.mark.gpu` - Tests requiring GPU/CUDA
- `@pytest.mark.api` - Tests requiring API keys
- `@pytest.mark.video` - Tests requiring video files

## Writing Tests

### Test File Naming

- Unit tests: `test_<module_name>.py`
- Integration tests: `test_<workflow_name>.py`
- All test files must start with `test_`

### Test Function Naming

```python
def test_<feature_being_tested>():
    """Describe what this test validates."""
    # Arrange
    # Act
    # Assert
```

### Using Fixtures

```python
def test_with_sample_data(sample_actions_data):
    """Test using shared fixture."""
    assert 'results' in sample_actions_data
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6)
])
def test_multiplication(input, expected):
    assert input * 2 == expected
```

### Mocking External Dependencies

```python
from unittest.mock import patch, MagicMock

@patch('module.external_function')
def test_with_mock(mock_function):
    mock_function.return_value = "mocked result"
    # Test code
```

## Fixtures

Common fixtures are defined in `conftest.py`:

- `project_root_path` - Project root directory
- `test_data_dir` - Test fixtures directory
- `temp_output_dir` - Temporary output directory
- `sample_frame` - Sample video frame (numpy array)
- `sample_pil_image` - Sample PIL Image
- `sample_actions_data` - Sample CLIP actions data
- `sample_games_data` - Sample game segmentation data
- `sample_transcript_data` - Sample transcript data
- `sample_ui_regions` - Sample UI regions configuration

## Continuous Integration

To run tests in CI/CD pipeline:

```bash
# Install test dependencies
pip install -r requirements.txt pytest pytest-cov

# Run tests with coverage
pytest --cov=src --cov-report=xml --cov-report=term

# Check coverage threshold
pytest --cov=src --cov-fail-under=80
```

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install package in development mode
pip install -e .
```

### API Key Errors

Skip API-dependent tests:

```bash
pytest -m "not api"
```

### GPU Errors

Skip GPU-dependent tests:

```bash
pytest -m "not gpu"
```

## Best Practices

1. **Fast Tests**: Unit tests should complete in milliseconds
2. **Isolated Tests**: Each test should be independent
3. **Clear Names**: Test names should describe what they test
4. **One Assertion**: Prefer single concept per test
5. **Use Fixtures**: Reuse common test data
6. **Mock External**: Mock API calls, file I/O when possible
7. **Test Edge Cases**: Include boundary conditions
8. **Document Complex Tests**: Add docstrings for complex logic

## Example Test

```python
import pytest
from src.core.utils import ActionClassifier

@pytest.mark.unit
class TestActionClassifier:
    """Test suite for ActionClassifier."""

    def test_init_with_custom_labels(self, sample_clip_labels):
        """Test initialization with custom labels."""
        # Arrange
        labels = sample_clip_labels

        # Act
        classifier = ActionClassifier(labels=labels)

        # Assert
        assert classifier.labels == labels
        assert classifier.num_labels == len(labels)
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest markers](https://docs.pytest.org/en/stable/example/markers.html)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
