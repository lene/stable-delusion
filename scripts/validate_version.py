#!/usr/bin/env python3
"""
Validate that version has been updated correctly for a merge request.

Checks:
1. Version in pyproject.toml is higher than the latest git tag
2. Version appears in CHANGELOG.md with a release date
3. Version format follows semantic versioning
"""

import re
import sys
import subprocess
from pathlib import Path


def get_current_version():
    """Get version from pyproject.toml."""
    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        print("❌ pyproject.toml not found")
        sys.exit(1)

    content = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        print("❌ Version not found in pyproject.toml")
        sys.exit(1)

    return match.group(1)


def get_latest_git_tag():
    """Get the latest version tag from git."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            tag = result.stdout.strip()
            # Remove 'v' prefix if present
            return tag[1:] if tag.startswith("v") else tag
        # No tags exist yet
        return "0.0.0"
    except (subprocess.SubprocessError, OSError) as e:
        print(f"⚠️  Warning: Could not get git tags: {e}")
        return "0.0.0"


def parse_version(version_string):
    """Parse semantic version string into tuple of integers."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-(.+))?$", version_string)
    if not match:
        return None
    major, minor, patch = match.groups()[:3]
    return (int(major), int(minor), int(patch))


def validate_version_format(version):
    """Validate version follows semantic versioning."""
    if not re.match(r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?$", version):
        print(f"❌ Version '{version}' does not follow semantic versioning (X.Y.Z)")
        return False
    return True


def validate_version_increased(current_version, previous_version):
    """Validate that current version is higher than previous."""
    current = parse_version(current_version)
    previous = parse_version(previous_version)

    if not current:
        print(f"❌ Invalid current version format: {current_version}")
        return False

    if not previous:
        print(f"⚠️  Could not parse previous version: {previous_version}")
        return True

    if current <= previous:
        print(
            f"❌ Version must be increased! Current: {current_version}, "
            f"Previous: {previous_version}"
        )
        return False

    return True


def validate_changelog(version):
    """Validate that version is documented in CHANGELOG.md."""
    changelog = Path("CHANGELOG.md")
    if not changelog.exists():
        print("⚠️  CHANGELOG.md not found - skipping changelog validation")
        return True

    content = changelog.read_text(encoding="utf-8")

    # Check for version entry with date: ## [X.Y.Z] - YYYY-MM-DD
    version_pattern = rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}"
    if not re.search(version_pattern, content, re.MULTILINE):
        print(f"❌ Version {version} not found in CHANGELOG.md")
        print(f"   Expected format: ## [{version}] - YYYY-MM-DD")
        print("   Please add a changelog entry for this version")
        return False

    # Check for version link at the bottom
    link_pattern = rf"^\[{re.escape(version)}\]:\s+https://"
    if not re.search(link_pattern, content, re.MULTILINE):
        print(f"⚠️  Version link for {version} not found in CHANGELOG.md")
        print(
            f"   Add: [{version}]: "
            f"https://gitlab.com/lilacashes/stable-delusion/releases/tag/v{version}"
        )

    return True


def main():
    """Run all validation checks."""
    print("🔍 Validating version update...\n")

    # Get versions
    current_version = get_current_version()
    previous_version = get_latest_git_tag()

    print(f"📦 Current version: {current_version}")
    print(f"📌 Previous version: {previous_version}\n")

    # Run validations
    checks_passed = True

    # 1. Check version format
    print("✓ Checking version format...")
    if not validate_version_format(current_version):
        checks_passed = False
    else:
        print("  ✅ Version format is valid\n")

    # 2. Check version increased
    print("✓ Checking version increment...")
    if not validate_version_increased(current_version, previous_version):
        checks_passed = False
    else:
        print(f"  ✅ Version increased from {previous_version} to {current_version}\n")

    # 3. Check changelog
    print("✓ Checking CHANGELOG.md...")
    if not validate_changelog(current_version):
        checks_passed = False
    else:
        print("  ✅ Version documented in CHANGELOG.md\n")

    # Final result
    if checks_passed:
        print("🎉 All version validations passed!")
        sys.exit(0)
    else:
        print("\n❌ Version validation failed!")
        print("\nPlease ensure:")
        print("  1. Version in pyproject.toml follows semantic versioning")
        print("  2. Version is higher than the previous version")
        print("  3. Version is documented in CHANGELOG.md with format:")
        print(f"     ## [{current_version}] - YYYY-MM-DD")
        sys.exit(1)


if __name__ == "__main__":
    main()
