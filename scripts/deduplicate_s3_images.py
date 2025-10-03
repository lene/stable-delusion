#!/usr/bin/env python3
"""Deduplicate S3 images by SHA-256 hash, keeping only the oldest copy."""

import os
import sys
from collections import defaultdict
from datetime import datetime
import boto3
from botocore.exceptions import ClientError


def get_s3_client():
    return boto3.client("s3")


def list_all_objects(s3_client, bucket):
    objects = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if not key.endswith("/"):
                    objects.append({
                        "Key": key,
                        "LastModified": obj["LastModified"],
                        "Size": obj["Size"]
                    })
    return objects


def get_object_metadata(s3_client, bucket, key):
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return response.get("Metadata", {})
    except ClientError as e:
        print(f"⚠️  Error getting metadata for {key}: {e}")
        return {}


def group_by_hash(s3_client, bucket, objects):
    hash_groups = defaultdict(list)
    no_hash_count = 0

    print("📊 Analyzing files...")
    for i, obj in enumerate(objects, 1):
        if i % 20 == 0 or i == len(objects):
            print(f"   Processing {i}/{len(objects)}...", end="\r")

        metadata = get_object_metadata(s3_client, bucket, obj["Key"])
        sha256 = metadata.get("sha256")

        if sha256:
            hash_groups[sha256].append(obj)
        else:
            no_hash_count += 1

    print(f"\n   ✅ Analyzed {len(objects)} files")
    if no_hash_count > 0:
        print(f"   ⚠️  {no_hash_count} files without SHA-256 metadata (skipped)")

    return hash_groups


def find_duplicates(hash_groups):
    duplicates_to_delete = []

    for sha256, objects in hash_groups.items():
        if len(objects) > 1:
            # Sort by LastModified (oldest first)
            objects.sort(key=lambda x: x["LastModified"])
            oldest = objects[0]
            duplicates = objects[1:]

            duplicates_to_delete.append({
                "hash": sha256,
                "keep": oldest,
                "delete": duplicates
            })

    return duplicates_to_delete


def calculate_size_savings(duplicates):
    total_bytes = 0
    for dup in duplicates:
        for obj in dup["delete"]:
            total_bytes += obj["Size"]
    return total_bytes


def format_size(bytes_size):
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"


def print_duplicate_report(duplicates):
    print("\n" + "=" * 80)
    print("📋 DUPLICATE FILES REPORT")
    print("=" * 80)

    if not duplicates:
        print("✅ No duplicates found!")
        return

    total_to_delete = sum(len(dup["delete"]) for dup in duplicates)
    total_size = calculate_size_savings(duplicates)

    print(f"\n🔍 Found {len(duplicates)} sets of duplicates")
    print(f"📦 Total files to delete: {total_to_delete}")
    print(f"💾 Space to reclaim: {format_size(total_size)}")
    print("\n" + "-" * 80)

    # Show first 10 duplicate sets
    for i, dup in enumerate(duplicates[:10], 1):
        print(f"\n[{i}] Duplicate set ({len(dup['delete'])} duplicates):")
        print(f"    Hash: {dup['hash'][:16]}...")
        print(f"    ✅ KEEP:   {dup['keep']['Key']}")
        print(f"              (modified: {dup['keep']['LastModified'].strftime('%Y-%m-%d %H:%M:%S')})")
        for obj in dup['delete']:
            print(f"    ❌ DELETE: {obj['Key']}")
            print(f"              (modified: {obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')})")

    if len(duplicates) > 10:
        remaining = len(duplicates) - 10
        remaining_files = sum(len(dup["delete"]) for dup in duplicates[10:])
        print(f"\n... and {remaining} more duplicate sets ({remaining_files} files)")

    print("\n" + "=" * 80)


def delete_duplicates(s3_client, bucket, duplicates):
    deleted_count = 0
    failed_count = 0

    print("\n🗑️  Deleting duplicate files...")

    for dup in duplicates:
        for obj in dup["delete"]:
            try:
                s3_client.delete_object(Bucket=bucket, Key=obj["Key"])
                deleted_count += 1
                if deleted_count % 10 == 0:
                    print(f"   Deleted {deleted_count} files...", end="\r")
            except ClientError as e:
                print(f"\n❌ Error deleting {obj['Key']}: {e}")
                failed_count += 1

    print(f"\n   ✅ Deleted {deleted_count} duplicate files")
    if failed_count > 0:
        print(f"   ⚠️  Failed to delete {failed_count} files")

    return deleted_count, failed_count


def main():
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        print("Error: AWS_S3_BUCKET environment variable not set")
        sys.exit(1)

    print(f"🔄 Starting S3 deduplication for: s3://{bucket}")
    print()

    s3 = get_s3_client()

    # Step 1: List all objects
    print("📦 Listing all S3 objects...")
    objects = list_all_objects(s3, bucket)
    print(f"   Found {len(objects)} files")

    # Step 2: Group by hash
    hash_groups = group_by_hash(s3, bucket, objects)

    # Step 3: Find duplicates
    print("\n🔍 Finding duplicates...")
    duplicates = find_duplicates(hash_groups)

    # Step 4: Show report
    print_duplicate_report(duplicates)

    if not duplicates:
        return

    # Step 5: Confirm deletion
    skip_confirm = os.getenv("SKIP_CONFIRM")
    if not skip_confirm:
        print("\n⚠️  This will permanently delete duplicate files!")
        response = input("Proceed with deletion? (yes/no) ")
        if response.lower() not in ["yes", "y"]:
            print("Aborted.")
            sys.exit(0)

    # Step 6: Delete duplicates
    deleted, failed = delete_duplicates(s3, bucket, duplicates)

    # Step 7: Final report
    remaining = len(objects) - deleted
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    print(f"📁 Original files:     {len(objects)}")
    print(f"🗑️  Duplicates deleted: {deleted}")
    print(f"📄 Files remaining:    {remaining}")
    if failed > 0:
        print(f"⚠️  Failed deletions:   {failed}")
    print(f"💾 Space reclaimed:    {format_size(calculate_size_savings(duplicates))}")
    print("\n✨ Deduplication complete!")


if __name__ == "__main__":
    main()
