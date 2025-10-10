# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5] - 2025-10-10

### Changed
- Major code refactoring to one-class-per-file architecture for improved maintainability
- Split `config.py` into separate files in `config/` directory (Config, ConfigManager, constants)
- Split `exceptions.py` into separate files in `exceptions/` directory (one file per exception type)
- Split `models/client_config.py` into separate files in `models/client_config/` directory
- Split `models/responses.py` into separate files in `models/responses/` directory
- Split `repositories/interfaces.py` into separate files in `repositories/interfaces/` directory
- Split `services/interfaces.py` into separate files in `services/interfaces/` directory
- Reorganized client code with cleaner separation of concerns

### Added
- Token usage tracking and reporting system for API calls
- `TokenUsageTracker` service for monitoring API token consumption
- `token_stats.py` module for detailed token usage statistics
- Token usage models (`TokenUsage`, `TokenUsageEntry`, `TokenUsageStats`)
- Utility scripts for codebase analysis and maintenance
- Enhanced OpenAPI specification with 176 new lines of documentation

### Fixed
- Test imports updated to work with new file structure
- Code quality issues resolved after file reorganization

### Performance
- Improved code modularity and reusability through better separation of concerns
- Reduced file size and complexity for easier navigation and maintenance

## [0.1.4] - 2025-10-04

### Added
- Automatic image size optimization for files exceeding 7MB
- JPEG conversion with progressive quality reduction (95% → 5% in 5% decrements)
- Support for optimizing PNG, WebP, grayscale, and RGBA images
- Temporary file management with automatic cleanup after upload
- 17 comprehensive tests for image optimization (all programmatically generated)

### Changed
- Image uploads to Gemini API now automatically optimized if >7MB
- Image uploads to Seedream API now automatically optimized if >7MB
- Images converted to optimized JPEG format when size reduction needed

### Performance
- Ensures compatibility with API size limits (Gemini: 7MB, Seedream: 10MB)
- Reduced upload failures due to file size constraints

## [0.1.3] - 2025-10-03

### Added
- SHA-256 hash-based duplicate detection for S3 file uploads
- Hash caching for O(1) duplicate lookups (reduced from O(N) per file)
- S3 deduplication script (`scripts/deduplicate_s3_images.py`)
- SHA-256 metadata backfill script for existing S3 files
- Automatic git tag creation in CI pipeline after successful PyPI deployment
- TestPyPI version validation in `validate_version.py` to prevent duplicate uploads

### Fixed
- Seedream service now uses HTTPS URLs instead of s3:// for API compatibility
- Version validation now checks if version already exists on TestPyPI
- Duplicate uploads to S3 now properly skip and return existing file URL
- CI pipeline no longer fails when tag is created (uses `[skip ci]`)

### Changed
- Refactored long functions to improve code readability (avg 9.0 lines/function)
- S3 duplicate detection uses shared `build_s3_hash_cache()` utility
- Reduced duplicate code across S3 repositories

### Performance
- Optimized S3 duplicate checking from O(N*M) to O(N+M) API calls using hash caching

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

[Unreleased]: https://gitlab.com/lilacashes/stable-delusion/compare/v0.1.5...HEAD
[0.1.5]: https://gitlab.com/lilacashes/stable-delusion/compare/v0.1.4...v0.1.5
[0.1.4]: https://gitlab.com/lilacashes/stable-delusion/compare/v0.1.3...v0.1.4
[0.1.3]: https://gitlab.com/lilacashes/stable-delusion/compare/v0.1.2...v0.1.3
[0.1.2]: https://gitlab.com/lilacashes/stable-delusion/compare/v0.1.1...v0.1.2
[0.1.1]: https://gitlab.com/lilacashes/stable-delusion/compare/v0.1.0...v0.1.1
[0.1.0]: https://gitlab.com/lilacashes/stable-delusion/releases/tag/v0.1.0