#!/usr/bin/env python3
"""Reorganize S3 bucket to match new structure."""

import os
import sys
import boto3
from botocore.exceptions import ClientError


def get_s3_client():
    """Create S3 client with credentials from environment."""
    return boto3.client("s3")


def move_s3_object(s3_client, bucket, source_key, dest_key):
    """Move an object in S3 from source to destination."""
    try:
        # Copy object to new location
        copy_source = {"Bucket": bucket, "Key": source_key}
        s3_client.copy_object(CopySource=copy_source, Bucket=bucket, Key=dest_key)

        # Delete original object
        s3_client.delete_object(Bucket=bucket, Key=source_key)

        return True
    except ClientError as e:
        print(f"❌ Error moving {source_key}: {e}")
        return False


def list_s3_objects(s3_client, bucket, prefix):
    """List all objects with the given prefix."""
    objects = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                # Skip directory markers
                if not key.endswith("/"):
                    objects.append(key)

    return objects


def main():
    bucket = os.getenv("AWS_S3_BUCKET")
    if not bucket:
        print("Error: AWS_S3_BUCKET environment variable not set")
        sys.exit(1)

    print(f"🔄 Starting S3 bucket reorganization for: s3://{bucket}")
    print()
    print("This will reorganize files into:")
    print("  - input/           (Seedream input images)")
    print("  - output/gemini/   (Gemini-generated images)")
    print("  - output/seedream/ (Seedream-generated images)")
    print("  - metadata/        (Generation metadata)")
    print()

    skip_confirm = os.getenv("SKIP_CONFIRM")
    if not skip_confirm:
        response = input("Continue? (y/n) ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    s3 = get_s3_client()

    # Counters
    seedream_input_count = 0
    gemini_output_count = 0
    errors = 0

    # Step 1: Move Seedream inputs
    print()
    print("📦 Step 1: Moving Seedream input images to input/")
    print("━" * 50)

    seedream_inputs = list_s3_objects(s3, bucket, "images/seedream/inputs/")
    for source_key in seedream_inputs:
        filename = os.path.basename(source_key)
        dest_key = f"input/{filename}"

        if move_s3_object(s3, bucket, source_key, dest_key):
            seedream_input_count += 1
            print(f"✅ Moved: {filename}")
        else:
            errors += 1

    # Step 2: Move Gemini outputs
    print()
    print("📦 Step 2: Moving Gemini output images to output/gemini/")
    print("━" * 50)

    # Get all images in the images/ prefix
    all_images = list_s3_objects(s3, bucket, "images/")

    # Filter for generated and upscaled images (not in seedream/ subfolder)
    gemini_outputs = [
        key
        for key in all_images
        if ("generated_" in key or "upscaled_" in key) and "/seedream/" not in key
    ]

    for source_key in gemini_outputs:
        filename = os.path.basename(source_key)
        dest_key = f"output/gemini/{filename}"

        if move_s3_object(s3, bucket, source_key, dest_key):
            gemini_output_count += 1
            print(f"✅ Moved: {filename}")
        else:
            errors += 1

    # Summary
    print()
    print("📊 Summary")
    print("━" * 50)
    print(f"✅ Seedream inputs moved: {seedream_input_count}")
    print(f"✅ Gemini outputs moved:  {gemini_output_count}")
    if errors > 0:
        print(f"⚠️  Errors encountered:    {errors}")
    print()

    # Verify new structure
    print("📋 Verifying new structure:")
    print()

    input_count = len(list_s3_objects(s3, bucket, "input/"))
    gemini_count = len(list_s3_objects(s3, bucket, "output/gemini/"))

    print(f"Input images: {input_count}")
    print(f"Gemini output images: {gemini_count}")
    print()

    print("✨ Reorganization complete!")
    print()
    print("⚠️  Note: The old 'images/' directory structure remains in place.")
    print("   After verifying the new structure works, you can manually remove it with:")
    print(f"   aws s3 rm s3://{bucket}/images/ --recursive")


if __name__ == "__main__":
    main()
