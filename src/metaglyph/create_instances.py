#!/usr/bin/env python3
"""Duplicate existing task instances for statistical robustness.

This is a legacy helper for task directories that already contain
instance_001.* files. For normal dataset generation, run Stage 1 with
`uv run metaglyph --stage 1`.
"""

import shutil
from pathlib import Path

TASK_FAMILIES = [
    "1_selection_classification",
    "2_structured_extraction",
    "3_constraint_composition",
    "4_conditional_transformation",
]

EXTENSIONS = [".input", ".gold", ".constraints", ".meta"]

def create_instances(tasks_dir: Path, num_instances: int = 50):
    """Duplicate instance_001 to create instance_002 through instance_N."""
    errors = []

    for family in TASK_FAMILIES:
        family_dir = tasks_dir / family
        if not family_dir.exists():
            errors.append(f"{family}: directory not found at {family_dir}")
            continue

        # Check if instance_001 exists
        source_files = {ext: family_dir / f"instance_001{ext}" for ext in EXTENSIONS}
        missing = [ext for ext, path in source_files.items() if not path.exists()]

        if missing:
            errors.append(f"{family}: missing instance_001 files for extensions {missing}")
            continue

        # Create instances 002 through N
        created = 0
        for i in range(2, num_instances + 1):
            instance_id = f"instance_{i:03d}"

            for ext in EXTENSIONS:
                source = source_files[ext]
                dest = family_dir / f"{instance_id}{ext}"

                if not dest.exists():
                    if ext == ".meta":
                        # Update instance_id in meta file
                        with open(source, 'r', encoding='utf-8') as f:
                            content = f.read()
                        # Replace instance_001 with new instance_id
                        content = content.replace("instance_001", instance_id)
                        with open(dest, 'w', encoding='utf-8') as f:
                            f.write(content)
                    else:
                        shutil.copy(source, dest)
                    created += 1

        print(f"{family}: created {created} files ({num_instances - 1} instances)")

    if errors:
        error_text = "\n".join(f"- {error}" for error in errors)
        raise FileNotFoundError(
            "Cannot duplicate task instances because required source artifacts are missing.\n"
            f"{error_text}\n\n"
            "Generate tasks first with: uv run metaglyph --stage 1\n"
            "Then run this helper only if you have legacy instance_001.* files to duplicate."
        )


def fix_existing_meta_files(tasks_dir: Path, num_instances: int = 50):
    """Fix instance_id in existing .meta files."""
    fixed = 0

    for family in TASK_FAMILIES:
        family_dir = tasks_dir / family
        if not family_dir.exists():
            continue

        for i in range(2, num_instances + 1):
            instance_id = f"instance_{i:03d}"
            meta_path = family_dir / f"{instance_id}.meta"

            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check if it still has wrong instance_id
                if "instance_001" in content:
                    content = content.replace("instance_001", instance_id)
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixed += 1

    return fixed

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Duplicate existing task instances for experiments")
    parser.add_argument("--instances", type=int, default=50, help="Total number of instances (default: 50)")
    parser.add_argument("--tasks-dir", type=str, default="tasks", help="Tasks directory (default: tasks)")
    parser.add_argument("--fix", action="store_true", help="Fix existing .meta files with wrong instance_id")

    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)

    if args.fix:
        fixed = fix_existing_meta_files(tasks_dir, args.instances)
        print(f"Fixed {fixed} .meta files")
    else:
        try:
            create_instances(tasks_dir, args.instances)
        except FileNotFoundError as exc:
            raise SystemExit(f"error: {exc}") from exc

    print("\nDone! Continue with: uv run metaglyph --stage 2-6")


if __name__ == "__main__":
    main()
