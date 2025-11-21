# MindGuard Backend Test Suite

This directory contains comprehensive unit and integration tests for the MindGuard backend using pytest.

## Test Structure

Tests mirror the project structure:
```
backend/
├── app/
│   ├── services/
│   │   └── recommendation_service.py
│   └── routes/
│       └── ...
└── tests/
    ├── test_recommendation_service.py
    └── test_*.py (mirroring app structure)
```

## Running Tests

### Install Dependencies

First, ensure all test dependencies are installed:

```bash
cd backend
pip install -r requirements.txt
```

### Run All Tests

```bash
# From backend directory
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=app --cov-report=html
```

### Run Specific Tests

```bash
# Run tests for a specific module
pytest tests/test_recommendation_service.py

# Run tests matching a pattern
pytest -k "test_get_personalized_recommendations"

# Run tests with a specific marker
pytest -m unit
pytest -m integration
```

### Run Tests in Parallel

```bash
# Install pytest-xdist first: pip install pytest-xdist
pytest -n auto
```

### View Coverage Report

```bash
# Generate HTML coverage report
pytest --cov=app --cov-report=html

# Open the report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Integration tests (may require external services)
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.model` - ML model tests
- `@pytest.mark.db` - Database tests

## Test Conventions

1. **Test Naming**: Tests use descriptive names like `test_get_personalized_recommendations_high_risk`
2. **Fixtures**: Shared fixtures are in `conftest.py`
3. **Mocking**: External APIs and services are mocked for deterministic tests
4. **Independence**: Tests run independently and don't require internet access
5. **Edge Cases**: Tests include edge cases (empty data, invalid inputs, etc.)

## Writing New Tests

When creating a new module or function:

1. Create a test file in `tests/` mirroring the module path
2. Use pytest fixtures from `conftest.py` when appropriate
3. Mock external dependencies (APIs, databases, ML models)
4. Include:
   - Happy path tests
   - Edge case tests
   - Error handling tests
   - Integration tests when meaningful

### Example Test Structure

```python
import pytest
from app.services.my_service import MyService

@pytest.mark.unit
class TestMyService:
    """Test suite for MyService"""
    
    @pytest.fixture
    def service(self):
        return MyService()
    
    def test_basic_functionality(self, service):
        """Test basic functionality"""
        result = service.do_something()
        assert result is not None
    
    def test_edge_case_empty_input(self, service):
        """Test with empty input"""
        result = service.do_something("")
        assert result == expected_value
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```bash
# CI-friendly command (no interactive output)
pytest --cov=app --cov-report=xml --junitxml=test-results.xml
```

## Troubleshooting

### Tests Failing Due to Missing Dependencies

```bash
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

### Database Connection Issues

Tests use mocked database connections by default. If you need real database tests:

```bash
# Set test database environment variable
export TEST_DATABASE_NAME=mindguard_test_db
pytest -m db
```

### Import Errors

Ensure you're running tests from the backend directory:

```bash
cd backend
pytest
```

