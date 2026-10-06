#!/usr/bin/env bash
# Download the latest version of the prediction dataset from Zenodo.
# Concept DOI: 10.5281/zenodo.21342806 (always resolves to the newest version;
# version 2.0 = 10.5281/zenodo.23196883 was published 2026-10-06).
set -euo pipefail

CONCEPT_API="https://zenodo.org/api/records/21342806"
DEST_DIR="data"
mkdir -p "$DEST_DIR"

for FILE in SE_full_predictions.csv.gz SE_split.gpkg; do
  echo "Resolving $FILE ..."
  URL=$(curl -sL "$CONCEPT_API" | jq -r --arg f "$FILE" '.files[] | select(.key==$f) | .links.self')
  if [ -z "$URL" ] || [ "$URL" = "null" ]; then
    echo "Could not find $FILE in the record" >&2
    exit 1
  fi
  echo "Downloading $FILE ..."
  curl -L --fail --progress-bar "$URL" -o "$DEST_DIR/$FILE"
done

echo "Done in $DEST_DIR/"
