#!/usr/bin/env bash
set -euo pipefail
ACTION="${1:-help}"; FILE="${2:-}"
MAX_BYTES=$((45 * 1024 * 1024))
case "$ACTION" in
  split)
    local f="$1"; base=$(basename "$f" .gz); base=$(basename "$base" .csv)
    dir="$(dirname "$f")/${base}_chunks"; mkdir -p "$dir"
    if [[ "$f" == *.gz ]]; then
      header=$(zcat "$f" | head -1); total=$(($(zcat "$f" | wc -l) - 1))
      parts=$(( (total * $(stat -c%s "$f") / total / MAX_BYTES) + 1 ))
      zcat "$f" | tail -n +2 | split -l $(( total / parts + 1 )) \
        --filter="(echo \"$header\"; cat) | gzip > \\$FILE.gz" - "$dir/${base}_part_"
    fi
    echo "Created $(ls "$dir"/*.gz 2>/dev/null | wc -l) chunks in $dir/"
    ;;
  join)
    dir="$1"; out="${dir%_chunks}.csv.gz"
    for part in $(ls "$dir"/*.gz | sort); do zcat "$part"; done | gzip > "$out"
    echo "Done: $out"
    ;;
  *) echo "Usage: $0 {split|join} <file_or_dir>"; exit 1 ;;
esac
