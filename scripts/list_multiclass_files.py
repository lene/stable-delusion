#!/usr/bin/env python3
"""List source files containing more than 1 class."""

from pathlib import Path

def count_classes(file_path):
    """Count classes in a file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return sum(1 for line in lines if line.strip().startswith('class '))
    except:
        return 0

# Find all Python files in stable_delusion
files = list(Path('stable_delusion').rglob('*.py'))

# Filter and sort by class count
multiclass_files = []
for file in files:
    class_count = count_classes(file)
    if class_count > 1:
        multiclass_files.append((class_count, str(file)))

# Sort by class count (descending)
multiclass_files.sort(reverse=True)

print(f"\nSource files with multiple classes ({len(multiclass_files)} files):\n")
print(f"{'Classes':<10} {'File'}")
print("-" * 80)

for count, filepath in multiclass_files:
    print(f"{count:<10} {filepath}")

print(f"\nTotal: {len(multiclass_files)} files with multiple classes")
print(f"Total classes to refactor: {sum(c for c, _ in multiclass_files)}")
