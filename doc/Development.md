# Development Guide

This guide covers development practices, code quality tools, CI/CD pipeline, and
security best practices for stable-delusion.

## Installation

### From Source (For Development)

```bash
$ poetry install
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
$ poetry run pytest tests/ -v

# Run specific test categories
$ poetry run pytest tests/unit/ -v
$ poetry run pytest tests/integration/ -v
```

## Code Quality Tools

This project includes comprehensive code quality and security tools:

### Linting and Code Style

```bash
# Check code style with flake8
$ poetry run flake8 stable_delusion tests

# Run pylint for comprehensive code analysis
$ poetry run pylint stable_delusion/ tests/

# Run mypy for static type checking
$ poetry run mypy stable_delusion/
```

### Security Analysis

```bash
# Run security analysis with bandit
$ poetry run bandit -r stable_delusion/

# Bandit configuration excludes test files automatically
# See .bandit file for configuration details
```

## Pre-commit Workflow

Before committing code, run all quality checks:

```bash
# 1. Run tests
$ poetry run pytest

# 2. Check code style
$ poetry run flake8 stable_delusion tests

# 3. Run pylint analysis
$ poetry run pylint stable_delusion/ tests/

# 4. Run type checking
$ poetry run mypy stable_delusion/

# 5. Run security analysis
$ poetry run bandit -r stable_delusion/
```

## Configuration Files

- `.pylintrc` - Pylint configuration for code quality standards
- `.flake8` - Flake8 configuration for PEP 8 compliance
- `.bandit` - Bandit security scanner configuration
- `.gitlab-ci.yml` - CI/CD pipeline configuration
- `CLAUDE.md` - Development guidelines for AI assistance

## CI/CD Pipeline

The GitLab CI/CD pipeline automatically runs on every push and includes:

1. **Setup**: Project structure validation and dependency installation
2. **Tests**: Complete test suite execution (375 tests)
3. **Code Quality**:
   - Flake8 style checking
   - Pylint comprehensive analysis
   - MyPy static type checking
   - Bandit security scanning
4. **Deploy** (main branch only):
   - Automatic version bumping (patch version increment)
   - Build and publish to TestPyPI
5. **Verify** (main branch only):
   - Install package from TestPyPI
   - Run smoke tests to verify package integrity
6. **Publish** (main branch only):
   - Publish to production PyPI after successful TestPyPI verification

All quality gates must pass for the pipeline to succeed, ensuring consistent code
quality and security.

### Required CI/CD Variables

For deployment stages, the following CI/CD variables must be configured in
**Settings → CI/CD → Variables**:

- `CI_PUSH_TOKEN`: Project Access Token with `write_repository` scope (Maintainer role)
  - Required for automatic version bumping
  - Create at: Settings → Access Tokens
- `TESTPYPI_TOKEN`: API token from https://test.pypi.org/manage/account/token/
  - Required for TestPyPI deployment
- `PYPI_TOKEN`: API token from https://pypi.org/manage/account/token/
  - Required for production PyPI deployment

All tokens should be configured as **protected** and **masked** variables.

## Security Best Practices

- **Flask Debug Mode**: Controlled via `FLASK_DEBUG` environment variable (disabled by default)
- **Secret Management**: API keys stored in environment variables, never hardcoded
- **Test Isolation**: Security scanning excludes test files with mock credentials
- **Dependency Management**: Regular dependency updates via Poetry
- **Automated Security Scanning**: Bandit security analysis runs on every push via GitLab CI/CD

## Changelog Management

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
format. When adding new features or making changes:

1. Add your changes to the `[Unreleased]` section in `CHANGELOG.md`
2. Use the appropriate category:
   - **Added** for new features
   - **Changed** for changes in existing functionality
   - **Deprecated** for soon-to-be removed features
   - **Removed** for now removed features
   - **Fixed** for any bug fixes
   - **Security** for vulnerability fixes

The CI/CD pipeline automatically converts the `[Unreleased]` section to a
versioned release when deploying to main branch.

## Architecture Documentation

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).

## API Documentation

For API usage examples and endpoint documentation, see [API_DEMO.md](API_DEMO.md).