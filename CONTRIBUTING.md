# Contributing to Flipswitch Python SDK

Thank you for your interest in contributing to the Flipswitch Python SDK!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/python-sdk.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -e ".[dev]"`
6. Create a feature branch: `git checkout -b feature/your-feature`
7. Make your changes
8. Run tests: `pytest`
9. Commit your changes: `git commit -m "Add your feature"`
10. Push to the branch: `git push origin feature/your-feature`
11. Create a Pull Request

## Development Setup

Requirements:
- Python 3.9+

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run demo
python examples/demo.py <your-api-key>
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for public APIs
- Write docstrings for public functions and classes

## Pull Request Guidelines

- Keep changes focused and atomic
- Write clear commit messages
- Include tests for new functionality
- Update documentation as needed
- Ensure all tests pass

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
