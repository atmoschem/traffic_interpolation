#!/usr/bin/env python3
"""
Publish a new version of the Zenodo dataset record 21342807.

Creates a new version of the record, replaces the listed files, updates the
version note, and publishes.

Usage:
    export ZENODO_TOKEN=...        # or write it to ~/.zenodo_token
    python scripts/upload_zenodo.py --file ../xgboost_june_2026/SE_full_predictions.csv.gz \
        --version "v2 (2026-10-06)" \
        --note "Adds hourly XGBoost speed columns (speed_xgb_weekday_h*, speed_xgb_weekend_h*)."

    # prepare the draft without publishing:
    python scripts/upload_zenodo.py --file ... --no-publish

Token scopes needed: deposit:write and deposit:actions.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

API = "https://zenodo.org/api"
RECORD_ID = 21342807  # latest published version; the concept is 21342806


def get_token() -> str:
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        f = Path.home() / ".zenodo_token"
        if f.exists():
            token = f.read_text().strip()
    if not token:
        sys.exit("No Zenodo token: set ZENODO_TOKEN or write it to ~/.zenodo_token "
                 "(create one at https://zenodo.org/account/settings/applications/tokens/new/ "
                 "with scopes deposit:write and deposit:actions)")
    return token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", required=True,
                    help="file to upload into the new version (repeatable)")
    ap.add_argument("--record", type=int, default=RECORD_ID)
    ap.add_argument("--version", default=None, help="metadata version string")
    ap.add_argument("--note", default=None, help="extra sentence added to the record notes")
    ap.add_argument("--no-publish", action="store_true", help="leave the draft unpublished")
    args = ap.parse_args()

    headers = {"Authorization": f"Bearer {get_token()}"}

    # 0. check credentials
    r = requests.get(f"{API}/deposit/depositions", params={"size": 1}, headers=headers, timeout=60)
    if r.status_code != 200:
        sys.exit(f"Token rejected by Zenodo ({r.status_code}): {r.text[:300]}")

    # 1. create the new version (idempotent: returns the existing draft if there is one)
    r = requests.post(f"{API}/deposit/depositions/{args.record}/actions/newversion",
                      headers=headers, timeout=120)
    if r.status_code not in (201, 403):
        sys.exit(f"newversion failed ({r.status_code}): {r.text[:300]}")
    links = r.json().get("links", {})
    draft_url = links.get("latest_draft")
    if not draft_url:
        sys.exit(f"no draft link in response: {r.text[:300]}")
    draft = requests.get(draft_url, headers=headers, timeout=60).json()
    draft_id = draft["id"]
    bucket = draft["links"]["bucket"]
    print(f"draft version id: {draft_id} | bucket: {bucket}")

    # 2. remove draft files that will be replaced
    wanted = {Path(f).name for f in args.file}
    for f in draft.get("files", []):
        if f["filename"] in wanted:
            d = requests.delete(f"{draft_url}/files/{f['id']}", headers=headers, timeout=60)
            print(f"  removed old {f['filename']} ({f['filesize']} bytes): {d.status_code}")

    # 3. upload the new files (bucket API, supports large files)
    for path in args.file:
        p = Path(path)
        if not p.exists():
            sys.exit(f"missing file: {p}")
        with p.open("rb") as fh:
            r = requests.put(f"{bucket}/{p.name}", data=fh, headers=headers, timeout=7200)
        if r.status_code not in (200, 201):
            sys.exit(f"upload failed for {p.name} ({r.status_code}): {r.text[:300]}")
        print(f"  uploaded {p.name}: {p.stat().st_size / 1e6:.1f} MB")

    # 4. metadata: keep everything, update version/notes/description if asked
    meta = draft["metadata"]
    if args.version:
        meta["version"] = args.version
    if args.note:
        meta["notes"] = (meta.get("notes", "") + " " + args.note).strip()
    r = requests.put(draft_url, data=json.dumps({"metadata": meta}), headers=headers, timeout=120)
    if r.status_code != 200:
        sys.exit(f"metadata update failed ({r.status_code}): {r.text[:500]}")
    print("  metadata updated")

    # 5. publish
    if args.no_publish:
        print(f"\ndraft ready (not published): {draft['links']['html']}")
        return
    r = requests.post(f"{draft_url}/actions/publish", headers=headers, timeout=300)
    if r.status_code != 202:
        sys.exit(f"publish failed ({r.status_code}): {r.text[:300]}")
    rec = r.json()
    print(f"\npublished: {rec.get('doi_url', rec.get('links', {}).get('html', ''))}")
    print(f"  files: {[f['filename'] for f in rec.get('files', [])]}")


if __name__ == "__main__":
    main()
