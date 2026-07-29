#!/bin/bash
# Package nnUNet_results as a tarball for Grand Challenge's separate
# "Models" upload (Algorithm -> Models -> Upload). Grand Challenge
# extracts this to /opt/ml/model/ at runtime, matching the
# nnUNet_results env var set in the Dockerfile -- keeps the container
# image itself well under the platform's size guidance.
#
# Usage:
#   bash package_model_weights.sh

set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

echo "Checking model weights before packaging..."
bash "$SCRIPTPATH/check_weights.sh"

OUT_FILE="$SCRIPTPATH/nnUNet_results.tar.gz"

echo ""
echo "Packaging $SCRIPTPATH/nnUNet_results -> $OUT_FILE"
tar -czf "$OUT_FILE" -C "$SCRIPTPATH" nnUNet_results

SIZE_MB=$(du -m "$OUT_FILE" | cut -f1)
echo ""
echo "Done. $OUT_FILE (${SIZE_MB}MB)"
echo "Upload this file on your Algorithm's Models page on Grand Challenge."
