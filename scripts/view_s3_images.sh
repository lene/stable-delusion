#!/bin/bash
# View all images in S3 bucket using xv

set -e

# Get S3 bucket from environment or use default
S3_BUCKET="${AWS_S3_BUCKET:-}"

if [ -z "$S3_BUCKET" ]; then
    echo "Error: AWS_S3_BUCKET environment variable not set"
    echo "Usage: AWS_S3_BUCKET=your-bucket-name $0 [folder]"
    echo ""
    echo "Available folders:"
    echo "  input/           - Input images"
    echo "  output/gemini/   - Gemini-generated images"
    echo "  output/seedream/ - Seedream-generated images"
    echo "  metadata/        - Generation metadata"
    exit 1
fi

# Optional folder within bucket (default: view all outputs)
FOLDER="${1:-output/}"

echo "📦 Viewing images from S3: s3://${S3_BUCKET}/${FOLDER}"
echo ""

# Create temporary directory for downloads
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Bulk download all images from S3
echo "⬇️  Downloading images from s3://${S3_BUCKET}/${FOLDER}..."
echo ""

if aws s3 cp "s3://${S3_BUCKET}/${FOLDER}" "$TEMP_DIR/" --recursive \
    --exclude "*" \
    --include "*.png" \
    --include "*.jpg" \
    --include "*.jpeg" \
    --include "*.gif" \
    --include "*.bmp" \
    --include "*.webp" \
    --include "*.PNG" \
    --include "*.JPG" \
    --include "*.JPEG" \
    --include "*.GIF" \
    --include "*.BMP" \
    --include "*.WEBP"; then
    echo ""
    echo "✅ Download complete!"
else
    echo ""
    echo "❌ Download failed"
    exit 1
fi

echo ""

# Check if any files were downloaded
if [ -z "$(ls -A "$TEMP_DIR")" ]; then
    echo "❌ No images were successfully downloaded"
    exit 1
fi

# Display all images with xv
echo "🖼️  Opening images with xv..."
echo "   (Use arrow keys to navigate, 'q' to quit)"
echo ""

xv -nolim "$TEMP_DIR"/* 2>/dev/null || {
    echo "⚠️  xv exited with error or was closed"
}

echo ""
echo "✨ Done! Cleaning up temporary files..."
# Trap will handle cleanup automatically
