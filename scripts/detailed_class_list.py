#!/usr/bin/env python3
"""Show detailed list of classes in multi-class files."""

from pathlib import Path
import re

def get_classes(file_path):
    """Get class names from a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Find all class definitions
        pattern = r'^class\s+(\w+)'
        matches = re.findall(pattern, content, re.MULTILINE)
        return matches
    except:
        return []

# Find all Python files in stable_delusion
files = list(Path('stable_delusion').rglob('*.py'))

# Filter files with multiple classes
multiclass_files = []
for file in files:
    classes = get_classes(file)
    if len(classes) > 1:
        multiclass_files.append((len(classes), str(file), classes))

# Sort by class count (descending)
multiclass_files.sort(reverse=True)

print(f"\n{'='*80}")
print(f"DETAILED LIST: Source Files with Multiple Classes")
print(f"{'='*80}\n")

for count, filepath, classes in multiclass_files:
    print(f"📁 {filepath}")
    print(f"   {count} classes: {', '.join(classes)}")
    print()

print(f"{'='*80}")
print(f"Summary:")
print(f"  Files to refactor: {len(multiclass_files)}")
print(f"  Total classes:     {sum(c for c, _, _ in multiclass_files)}")
print(f"  Classes after:     {sum(c for c, _, _ in multiclass_files)} individual files")
print(f"  New files needed:  {sum(c for c, _, _ in multiclass_files) - len(multiclass_files)}")
print(f"{'='*80}\n")
