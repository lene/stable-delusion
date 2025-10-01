# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2025-10-01

### Fixed
- TestPyPI and PyPI deployment authentication with proper token configuration
- Repository URL configuration in deployment jobs

## [0.1.1] - 2025-10-01

### Added
- CI_PUSH_TOKEN support for automatic version bumping
- Cache key based on poetry.lock for reliable dependency caching

### Fixed
- CI pipeline cache issues causing "Command not found: pytest" errors
- Cross-device link error in output file handling using atomic file operations
- Removed redundant GitHub mirror job (now handled by GitLab's built-in feature)

### Changed
- Poetry install now uses --sync flag to ensure venv matches lock file exactly

## [0.1.0] - 2025-09-30

### Added
- AI-powered image generation using Google Gemini 2.5 Flash Image Preview
- Multi-image support for reference-based generation
- Automatic image upscaling (2x or 4x) using Google Vertex AI
- AWS S3 storage backend support alongside local filesystem storage
- RESTful Flask API with multipart/form-data endpoints
- OpenAPI 3.0.3 specification available at `/openapi.json`
- Command-line interface with comprehensive parameters
- Health check endpoint (`/health`) for service monitoring
- Generation metadata tracking with deduplication support
- Comprehensive error logging with safety filter diagnostics
- Environment-based configuration via `.env` files
- Flexible output directory and filename customization
- Coloredlogs with quiet/debug CLI flags
- Structured logging setup

### Changed
- Complete project renaming from nano-api to stable-delusion
- Renamed `--output` parameter to `--output-filename` for clarity
- Architecture simplification with shared utilities
- Removed unnecessary abstraction layers

### Security
- Flask debug mode disabled by default in production
- Environment variable-based secret management
- Input validation for all API endpoints
- Secure file upload handling with content type verification

### Testing
- 375 comprehensive tests covering unit and integration scenarios
- S3 integration tests for cloud storage backend
- Metadata repository tests for deduplication logic

### Infrastructure
- GitLab CI/CD pipeline with automated testing and quality checks
- Automated code quality enforcement (Flake8, Pylint, MyPy, Bandit)
- Automatic version bumping on main branch merges
- TestPyPI deployment and verification pipeline
- Production PyPI publishing after successful verification
- Poetry dependency management
- Type annotations throughout codebase

[Unreleased]: https://gitlab.com/lilacashes/stable-delusion/compare/v0.1.2...HEAD
[0.1.2]: https://gitlab.com/lilacashes/stable-delusion/compare/v0.1.1...v0.1.2
[0.1.1]: https://gitlab.com/lilacashes/stable-delusion/compare/v0.1.0...v0.1.1
[0.1.0]: https://gitlab.com/lilacashes/stable-delusion/releases/tag/v0.1.0