#!/bin/bash
# Reorganize S3 bucket structure to match new schema:
# - input/ for all input images
# - output/gemini/ for Gemini-generated images
# - output/seedream/ for Seedream-generated images
# - metadata/ for generation metadata

set -e

# Get S3 bucket from environment or use default
S3_BUCKET="${AWS_S3_BUCKET:-}"

if [ -z "$S3_BUCKET" ]; then
    echo "Error: AWS_S3_BUCKET environment variable not set"
    echo "Usage: AWS_S3_BUCKET=your-bucket-name $0"
    exit 1
fi

echo "🔄 Starting S3 bucket reorganization for: s3://${S3_BUCKET}"
echo ""
echo "This will reorganize files into:"
echo "  - input/           (Seedream input images)"
echo "  - output/gemini/   (Gemini-generated images)"
echo "  - output/seedream/ (Seedream-generated images)"
echo "  - metadata/        (Generation metadata)"
echo ""

# Skip confirmation if SKIP_CONFIRM is set
if [ -z "$SKIP_CONFIRM" ]; then
    read -p "Continue? (y/n) " -n 1 -r </dev/tty
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Track counts
SEEDREAM_INPUT_COUNT=0
GEMINI_OUTPUT_COUNT=0
ERRORS=0

echo ""
echo "📦 Step 1: Moving Seedream input images to input/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Move all files from images/seedream/inputs/ to input/
# Create temp file to avoid stdin conflicts
TEMP_FILE=$(mktemp)
aws s3 ls s3://${S3_BUCKET}/images/seedream/inputs/ --recursive | awk '{print $4}' > "$TEMP_FILE"

while IFS= read -r file; do
    if [ -z "$file" ]; then
        continue
    fi

    # Extract just the filename from the full S3 key
    filename=$(basename "$file")

    if aws s3 mv "s3://${S3_BUCKET}/${file}" "s3://${S3_BUCKET}/input/${filename}" 2>&1 | grep -q "move:"; then
        ((SEEDREAM_INPUT_COUNT++))
        echo "✅ Moved: ${filename}"
    else
        ((ERRORS++))
        echo "❌ Failed to move: ${filename}"
    fi
done < "$TEMP_FILE"
rm "$TEMP_FILE"

echo ""
echo "📦 Step 2: Moving Gemini output images to output/gemini/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Move generated images from images/ root
# Create temp file to avoid stdin conflicts
TEMP_FILE=$(mktemp)
aws s3 ls s3://${S3_BUCKET}/images/ --recursive | grep -E "(generated_|upscaled_)" | awk '{print $4}' > "$TEMP_FILE"

while IFS= read -r file; do
    if [ -z "$file" ]; then
        continue
    fi

    filename=$(basename "$file")

    if aws s3 mv "s3://${S3_BUCKET}/${file}" "s3://${S3_BUCKET}/output/gemini/${filename}" 2>&1 | grep -q "move:"; then
        ((GEMINI_OUTPUT_COUNT++))
        echo "✅ Moved: ${filename}"
    else
        ((ERRORS++))
        echo "❌ Failed to move: ${filename}"
    fi
done < "$TEMP_FILE"
rm "$TEMP_FILE"

echo ""
echo "📊 Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Seedream inputs moved: ${SEEDREAM_INPUT_COUNT}"
echo "✅ Gemini outputs moved:  ${GEMINI_OUTPUT_COUNT}"
if [ $ERRORS -gt 0 ]; then
    echo "⚠️  Errors encountered:    ${ERRORS}"
fi
echo ""

# Verify new structure
echo "📋 Verifying new structure:"
echo ""
echo "Input images:"
aws s3 ls s3://${S3_BUCKET}/input/ --recursive | wc -l | xargs echo "  Count:"
echo ""
echo "Gemini output images:"
aws s3 ls s3://${S3_BUCKET}/output/gemini/ --recursive | wc -l | xargs echo "  Count:"
echo ""

echo "✨ Reorganization complete!"
echo ""
echo "⚠️  Note: The old 'images/' directory structure remains in place."
echo "   After verifying the new structure works, you can manually remove it with:"
echo "   aws s3 rm s3://${S3_BUCKET}/images/ --recursive"
