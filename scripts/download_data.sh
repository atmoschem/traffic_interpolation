#!/usr/bin/env bash
ZENODO_URL="https://zenodo.org/records/XXXXXX/files"
DEST_DIR="data"
mkdir -p "$DEST_DIR"
echo "Downloading SE_full_predictions.csv.gz..."
wget -q --show-progress "$ZENODO_URL/SE_full_predictions.csv.gz" -O "$DEST_DIR/SE_full_predictions.csv.gz"
echo "Downloading SE_split.gpkg..."
wget -q --show-progress "$ZENODO_URL/SE_split.gpkg" -O "$DEST_DIR/SE_split.gpkg"
echo "Done in $DEST_DIR/"
