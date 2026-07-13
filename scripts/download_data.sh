#!/usr/bin/env bash
# DOI: 10.5281/zenodo.21342807
ZENODO_URL="https://zenodo.org/records/21342807/files"
DEST_DIR="data"
mkdir -p "$DEST_DIR"
echo "Downloading SE_full_predictions.csv.gz..."
wget -q --show-progress "$ZENODO_URL/SE_full_predictions.csv.gz" -O "$DEST_DIR/SE_full_predictions.csv.gz"
echo "Downloading SE_split.gpkg..."
wget -q --show-progress "$ZENODO_URL/SE_split.gpkg" -O "$DEST_DIR/SE_split.gpkg"
echo "Done in $DEST_DIR/"
