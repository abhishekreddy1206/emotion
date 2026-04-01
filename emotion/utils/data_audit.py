#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset deduplication and audit tool.

Scans all image directories, hashes images to find duplicates,
reports class-name inconsistencies, and outputs per-directory stats.

Usage:
    python -m utils.data_audit           # Dry run (report only)
    python -m utils.data_audit --merge   # Merge unique images into data/
"""

import os
import sys
import hashlib
import shutil
import argparse
from collections import defaultdict
from pathlib import Path


# Directories to scan (relative to emotion/ project root)
SCAN_DIRS = {
    "data": {"train": "data/train", "validation": "data/validation"},
    "CK+48/data": {"train": "CK+48/data/train", "validation": "CK+48/data/validation"},
    "CK+48/images": {"flat": "CK+48/images"},
    "images": {"train": "images/train", "validation": "images/validation"},
    "aligned": {"train": "aligned/train", "test": "aligned/test"},
}

# Canonical class name mapping (original -> canonical lowercase)
CLASS_NAME_MAP = {
    "anger": "angry",
    "Anger": "angry",
    "angry": "angry",
    "Angry": "angry",
    "fear": "fear",
    "Fear": "fear",
    "happy": "happy",
    "Happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "Neutral": "neutral",
    "sad": "sad",
    "Sad": "sad",
    "sadness": "sad",
    "Sadness": "sad",
    "surprise": "surprise",
    "Surprise": "surprise",
    "disgust": "disgust",
    "Disgust": "disgust",
    # Numeric labels from aligned/ dataset
    "1": "angry",
    "2": "disgust",
    "3": "fear",
    "4": "happy",
    "5": "neutral",
    "6": "sad",
    "7": "surprise",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


def hash_file(filepath):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_directory(base_dir):
    """Scan a directory tree for images, returning {hash: [(path, class_name), ...]}."""
    hash_to_files = defaultdict(list)
    stats = defaultdict(int)

    if not os.path.isdir(base_dir):
        return hash_to_files, stats

    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue

            filepath = os.path.join(root, fname)
            # Class name is the immediate parent directory
            class_dir = os.path.basename(root)
            canonical = CLASS_NAME_MAP.get(class_dir, class_dir.lower())

            file_hash = hash_file(filepath)
            hash_to_files[file_hash].append((filepath, canonical))
            stats[canonical] += 1

    return hash_to_files, stats


def run_audit(project_root, merge=False):
    """Run the full audit across all dataset directories."""
    print("=" * 70)
    print("DATASET AUDIT REPORT")
    print("=" * 70)

    all_hashes = defaultdict(list)  # hash -> [(path, class, source_dir)]
    dir_stats = {}

    for dir_name, splits in SCAN_DIRS.items():
        print(f"\nScanning: {dir_name}/")
        dir_total = 0
        for split_name, split_path in splits.items():
            full_path = os.path.join(project_root, split_path)
            hashes, stats = scan_directory(full_path)

            for h, entries in hashes.items():
                for path, cls in entries:
                    all_hashes[h].append((path, cls, dir_name))

            split_total = sum(stats.values())
            dir_total += split_total
            print(f"  {split_name}: {split_total} images across {len(stats)} classes")
            for cls, count in sorted(stats.items()):
                print(f"    {cls}: {count}")

        dir_stats[dir_name] = dir_total
        print(f"  Total: {dir_total}")

    # Find duplicates
    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS")
    print("=" * 70)

    total_images = sum(dir_stats.values())
    unique_hashes = len(all_hashes)
    duplicate_count = total_images - unique_hashes

    print(f"\nTotal images across all directories: {total_images}")
    print(f"Unique images (by hash): {unique_hashes}")
    print(f"Duplicate images: {duplicate_count}")

    # Cross-directory duplicates
    cross_dupes = 0
    for h, entries in all_hashes.items():
        dirs = set(e[2] for e in entries)
        if len(dirs) > 1:
            cross_dupes += 1

    print(f"Images duplicated across directories: {cross_dupes}")

    # Class name inconsistencies
    print("\n" + "=" * 70)
    print("CLASS NAME MAPPING")
    print("=" * 70)

    all_classes = set()
    for h, entries in all_hashes.items():
        for _, cls, _ in entries:
            all_classes.add(cls)

    print(f"\nCanonical classes found: {sorted(all_classes)}")

    # Per-class unique count
    class_unique = defaultdict(int)
    for h, entries in all_hashes.items():
        # Take the first class label as canonical (they should all agree)
        cls = entries[0][1]
        class_unique[cls] += 1

    print("\nUnique images per class:")
    for cls, count in sorted(class_unique.items()):
        print(f"  {cls}: {count}")

    if merge:
        print("\n" + "=" * 70)
        print("MERGING INTO data/")
        print("=" * 70)
        merge_into_canonical(project_root, all_hashes)

    return all_hashes, dir_stats


def merge_into_canonical(project_root, all_hashes):
    """Merge all unique images into data/train/ and data/validation/ directories."""
    canonical_dir = os.path.join(project_root, "data")
    train_dir = os.path.join(canonical_dir, "train")
    val_dir = os.path.join(canonical_dir, "validation")

    merged_count = 0
    skipped_count = 0

    for file_hash, entries in all_hashes.items():
        src_path, cls, src_dir = entries[0]

        # Check if this image is already in data/
        already_in_data = any(e[2] == "data" for e in entries)

        if not already_in_data:
            # Copy to train/ by default (80/20 split can be done later)
            dest_class_dir = os.path.join(train_dir, cls)
            os.makedirs(dest_class_dir, exist_ok=True)

            fname = os.path.basename(src_path)
            dest_path = os.path.join(dest_class_dir, f"{src_dir.replace('/', '_')}_{fname}")

            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                merged_count += 1
            else:
                skipped_count += 1

    print(f"\nMerged {merged_count} new unique images into data/")
    print(f"Skipped {skipped_count} (already existed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and deduplicate emotion datasets")
    parser.add_argument("--merge", action="store_true",
                        help="Merge unique images into data/ directory")
    args = parser.parse_args()

    # Assume script is run from the emotion/ directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(project_root) != "emotion":
        project_root = os.path.dirname(os.path.abspath(__file__))

    run_audit(project_root, merge=args.merge)
