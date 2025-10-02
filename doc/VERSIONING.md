# Version Management Guide

This project uses manual version bumping with automated validation.

## How to Update the Version

When preparing a merge request, follow these steps:

### 1. Update Version in `pyproject.toml`

Edit the version field using semantic versioning (MAJOR.MINOR.PATCH):

```toml
[tool.poetry]
name = "stable-delusion"
version = "0.2.0"  # Update this
```

**Semantic Versioning Guidelines:**
- **MAJOR** (X.0.0): Breaking changes, incompatible API changes
- **MINOR** (0.X.0): New features, backward-compatible
- **PATCH** (0.0.X): Bug fixes, backward-compatible

### 2. Update CHANGELOG.md

Add an entry for your version with the current date:

```markdown
## [Unreleased]

## [0.2.0] - 2025-10-02

### Added
- New feature description

### Changed
- Updated feature description

### Fixed
- Bug fix description
```

Add the version link at the bottom:

```markdown
[Unreleased]: https://gitlab.com/lilacashes/stable-delusion/compare/v0.2.0...HEAD
[0.2.0]: https://gitlab.com/lilacashes/stable-delusion/releases/tag/v0.2.0
[0.1.0]: https://gitlab.com/lilacashes/stable-delusion/releases/tag/v0.1.0
```

### 3. Validate Locally (Optional)

Before pushing, you can validate your changes locally:

```bash
python3 scripts/validate_version.py
```

This will check:
- ✓ Version format follows semantic versioning
- ✓ Version is higher than the previous version
- ✓ Version is documented in CHANGELOG.md

### 4. Create Merge Request

Push your changes and create a merge request. The CI pipeline will automatically:
- Run all tests
- Perform code quality checks
- **Validate your version update**

If validation fails, the CI will tell you exactly what needs to be fixed.

## CI Validation

The `validate_version` job runs on all merge requests and checks:

1. **Version Format**: Must follow `X.Y.Z` format (semantic versioning)
2. **Version Increment**: Must be higher than the latest git tag
3. **Changelog Entry**: Must have entry with format `## [X.Y.Z] - YYYY-MM-DD`

## Example Workflow

```bash
# 1. Update version in pyproject.toml
vim pyproject.toml  # Change version from 0.1.0 to 0.2.0

# 2. Update CHANGELOG.md
vim CHANGELOG.md    # Add ## [0.2.0] - 2025-10-02 section

# 3. Validate locally
python3 scripts/validate_version.py

# 4. Commit and push
git add pyproject.toml CHANGELOG.md
git commit -m "Bump version to 0.2.0"
git push origin feature-branch

# 5. Create merge request
# CI will validate version automatically
```

## Troubleshooting

### "Version must be increased!"

Your new version must be higher than the previous one. Check the latest tag:

```bash
git describe --tags --abbrev=0
```

### "Version X.Y.Z not found in CHANGELOG.md"

Add an entry to CHANGELOG.md with this exact format:

```markdown
## [X.Y.Z] - YYYY-MM-DD
```

### "Version does not follow semantic versioning"

Use the format `MAJOR.MINOR.PATCH` (e.g., `0.2.0`, `1.0.0`, `1.2.3`)

## Deployment

Once your MR is merged to `main`, the CI pipeline will:
1. Build the package
2. Deploy to TestPyPI for verification
3. Deploy to production PyPI if TestPyPI verification passes

The version you set will be used for the deployment.
