#!/usr/bin/env python3
"""Backfill SHA-256 metadata for existing S3 objects."""

import hashlib
import os
import sys
import boto3
from botocore.exceptions import ClientError


def calculate_sha256(content: bytes) -> str:
    hash_sha256 = hashlib.sha256()
    hash_sha256.update(content)
    return hash_sha256.hexdigest()


def get_s3_client():
    return boto3.client("s3")


def list_s3_objects(s3_client, bucket, prefix=""):
    objects = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if not key.endswith("/"):
                    objects.append(key)
    return objects


def backfill_metadata(s3_client, bucket, key):
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        existing_metadata = head.get("Metadata", {})
        if "sha256" in existing_metadata:
            return "skipped"
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read()
        file_hash = calculate_sha256(content)
        existing_metadata["sha256"] = file_hash
        copy_source = {"Bucket": bucket, "Key": key}
        s3_client.copy_object(
            CopySource=copy_source,
            Bucket=bucket,
            Key=key,
            Metadata=existing_metadata,
            MetadataDirective="REPLACE",
            ContentType=head.get("ContentType", "application/octet-stream"),
        )
        return "updated"
    except ClientError as e:
        print(f"❌ Error processing {key}: {e}")
        return "error"


def main():
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        print("Error: AWS_S3_BUCKET environment variable not set")
        sys.exit(1)
    print(f"🔄 Starting SHA-256 metadata backfill for: s3://{bucket}")
    print()
    print("This will add SHA-256 hash metadata to all files without it.")
    print()
    skip_confirm = os.getenv("SKIP_CONFIRM")
    if not skip_confirm:
        response = input("Continue? (y/n) ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)
    s3 = get_s3_client()
    updated = 0
    skipped = 0
    errors = 0
    print()
    print("📦 Processing S3 objects...")
    print("━" * 50)
    all_objects = list_s3_objects(s3, bucket)
    total = len(all_objects)
    for i, key in enumerate(all_objects, 1):
        result = backfill_metadata(s3, bucket, key)
        if result == "updated":
            updated += 1
            print(f"✅ [{i}/{total}] Updated: {key}")
        elif result == "skipped":
            skipped += 1
            print(f"⏭️  [{i}/{total}] Skipped (already has SHA-256): {key}")
        else:
            errors += 1
    print()
    print("📊 Summary")
    print("━" * 50)
    print(f"✅ Files updated:  {updated}")
    print(f"⏭️  Files skipped:  {skipped}")
    if errors > 0:
        print(f"⚠️  Errors encountered: {errors}")
    print()
    print("✨ Backfill complete!")


if __name__ == "__main__":
    main()
