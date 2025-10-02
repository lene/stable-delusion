# Scripts

## validate_version.py

Validates version updates for merge requests.

### Usage

```bash
python3 scripts/validate_version.py
```

### What it checks

1. **Version Format**: Ensures version follows semantic versioning (X.Y.Z)
2. **Version Increment**: Verifies new version is higher than the latest git tag
3. **Changelog Entry**: Confirms version is documented in CHANGELOG.md with proper format

### Exit Codes

- `0`: All validations passed
- `1`: One or more validations failed

### Example Output

```
🔍 Validating version update...

📦 Current version: 0.2.0
📌 Previous version: 0.1.0

✓ Checking version format...
  ✅ Version format is valid

✓ Checking version increment...
  ✅ Version increased from 0.1.0 to 0.2.0

✓ Checking CHANGELOG.md...
  ✅ Version documented in CHANGELOG.md

🎉 All version validations passed!
```

### CI Integration

This script runs automatically on all merge requests in the `validate_version` job.

See [VERSIONING.md](../doc/VERSIONING.md) for the complete version management workflow.
