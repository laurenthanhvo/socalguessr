#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/make_submission_zip.sh <checkpoint_path>"
  exit 1
fi

CHECKPOINT_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMISSION_DIR="$REPO_DIR/submission"
ZIP_PATH="$REPO_DIR/submission.zip"

rm -rf "$SUBMISSION_DIR"
mkdir -p "$SUBMISSION_DIR"

cp "$REPO_DIR/predict.py" "$SUBMISSION_DIR/predict.py"
cp "$CHECKPOINT_PATH" "$SUBMISSION_DIR/$(basename "$CHECKPOINT_PATH")"

rm -f "$ZIP_PATH"
(
  cd "$SUBMISSION_DIR"
  zip -r "$ZIP_PATH" .
)

echo "Created $ZIP_PATH"
