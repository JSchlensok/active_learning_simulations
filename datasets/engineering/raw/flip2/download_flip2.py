"""Download protein fitness landscape splits from FLIP2 (https://flip.protein.properties).

FLIP2 publishes one gzipped CSV per (dataset, split) as plain static files on GitHub Pages, plus a
per-dataset ``README.md`` carrying the upstream source, license and citation. There is no API, so
this script discovers what exists by parsing the ``assets/splits/<dataset>/<split>.csv.gz`` links
out of the landing page — new datasets and splits are picked up without touching this file.

Downloads mirror the upstream layout (``trpb/one_to_many.csv.gz``) and stay gzip-compressed, since
``pandas.read_csv`` reads ``.gz`` transparently. Every file is recorded in ``metadata.json`` with
its size and sha256, alongside the provenance block parsed from the dataset README.

Usage (from this directory):

    uv run python download_flip2.py                          # nucb + trpb, all their splits
    uv run python download_flip2.py --list                    # every dataset/split FLIP2 offers
    uv run python download_flip2.py --datasets amylase ired   # fetch others
    uv run python download_flip2.py --force                   # re-download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE_URL = "https://flip.protein.properties"
SPLITS_PATH = "assets/splits"
TIMEOUT_SECONDS = 120

DEFAULT_DATASETS = ["nucb", "trpb"]

HERE = Path(__file__).parent
METADATA_PATH = HERE / "metadata.json"

# Matches the download links on the landing page, e.g. assets/splits/trpb/one_to_many.csv.gz
SPLIT_LINK_PATTERN = re.compile(rf"{SPLITS_PATH}/([a-z0-9_]+)/([A-Za-z0-9_]+)\.csv\.gz")


def _fetch(url: str, accept: str = "*/*") -> bytes:
    request = urllib.request.Request(url, headers={"Accept": accept})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def discover_splits() -> dict[str, list[str]]:
    """Parse the FLIP2 landing page and return {dataset: [split, ...]}.

    Only the FLIP2 datasets link their CSVs directly; the legacy FLIP datasets (gb1, meltome, scl,
    aav, bind, cas, secondary_structure) link a directory instead and are therefore not discovered.
    """
    page = _fetch(f"{BASE_URL}/", accept="text/html").decode("utf-8", errors="replace")
    splits: dict[str, list[str]] = defaultdict(list)
    for dataset, split in SPLIT_LINK_PATTERN.findall(page):
        if split not in splits[dataset]:
            splits[dataset].append(split)
    if not splits:
        raise RuntimeError(
            "Found no split download links on the FLIP2 landing page; layout changed?"
        )
    return dict(splits)


def fetch_readme(dataset: str) -> str | None:
    """Return the upstream dataset README, which holds the source URL, license and citation."""
    try:
        return _fetch(
            f"{BASE_URL}/{SPLITS_PATH}/{dataset}/README.md", accept="text/markdown"
        ).decode("utf-8")
    except urllib.error.HTTPError:
        return None


def parse_provenance(readme: str) -> dict[str, str]:
    """Pull the structured fields out of a dataset README into metadata.json."""
    provenance: dict[str, str] = {}
    for field, key in (
        ("Dataset at", "source"),
        ("License", "license"),
        ("Provenance", "provenance"),
    ):
        match = re.search(rf"^{field}:?\s*(.+)$", readme, flags=re.MULTILINE)
        if match:
            provenance[key] = match.group(1).strip()
    # The citation is a multi-line blockquote after "Attributed to:".
    citation = re.search(r"Attributed to:\s*\n+((?:>.*\n?)+)", readme)
    if citation:
        text = " ".join(
            line.lstrip("> ").strip() for line in citation.group(1).splitlines()
        )
        provenance["citation"] = re.sub(r"\s+", " ", text).strip()
    return provenance


def download(url: str, target_path: Path, magic: bytes | None = None) -> bytes:
    content = _fetch(url)
    if not content:
        raise RuntimeError(f"{url}: server returned an empty body.")
    if magic and not content.startswith(magic):
        raise RuntimeError(
            f"{url}: expected {magic!r} magic bytes, got {content[:16]!r}"
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(content)
    return content


def _record(content: bytes, target_path: Path, url: str) -> dict[str, object]:
    return {
        "path": str(target_path.relative_to(HERE)),
        "url": url,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"FLIP2 dataset ids (default: {' '.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Restrict to these split names (default: every split of each dataset)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all discoverable datasets/splits and exit",
    )
    args = parser.parse_args()

    try:
        available = discover_splits()
    except (urllib.error.URLError, RuntimeError) as error:
        print(f"Could not read the FLIP2 landing page: {error}", file=sys.stderr)
        return 1

    if args.list:
        print(
            f"{len(available)} FLIP2 datasets ({sum(len(s) for s in available.values())} splits):"
        )
        for dataset, splits in sorted(available.items()):
            print(f"  {dataset:<10} {', '.join(sorted(splits))}")
        return 0

    unknown = [dataset for dataset in args.datasets if dataset not in available]
    if unknown:
        print(
            f"Unknown dataset id(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(available))}",
            file=sys.stderr,
        )
        return 1

    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    failures: list[str] = []

    for dataset in args.datasets:
        entry = metadata.setdefault(dataset, {"splits": {}})
        entry["source_url"] = f"{BASE_URL}/#downloads"

        readme = fetch_readme(dataset)
        if readme is not None:
            readme_path = HERE / dataset / "README.md"
            readme_path.parent.mkdir(parents=True, exist_ok=True)
            readme_path.write_text(readme)
            entry["provenance"] = parse_provenance(readme)
            entry["readme"] = str(readme_path.relative_to(HERE))

        wanted = [
            s for s in available[dataset] if args.splits is None or s in args.splits
        ]
        if not wanted:
            print(
                f"  {dataset}: none of the requested splits exist "
                f"(has: {', '.join(sorted(available[dataset]))})",
                file=sys.stderr,
            )
            failures.append(dataset)
            continue

        for split in sorted(wanted):
            url = f"{BASE_URL}/{SPLITS_PATH}/{dataset}/{split}.csv.gz"
            target_path = HERE / dataset / f"{split}.csv.gz"

            if target_path.exists() and not args.force:
                print(f"  {target_path.relative_to(HERE)} exists, skipping")
                continue

            try:
                content = download(url, target_path, magic=b"\x1f\x8b")  # gzip
            except (urllib.error.URLError, RuntimeError) as error:
                print(f"  {dataset}/{split}: FAILED ({error})", file=sys.stderr)
                failures.append(f"{dataset}/{split}")
                continue

            entry["splits"][split] = _record(content, target_path, url)
            print(f"  {target_path.relative_to(HERE)} ({len(content):,} bytes)")

        # Drop records of files that are no longer on disk, so metadata.json never overclaims.
        entry["splits"] = {
            split: record
            for split, record in entry["splits"].items()
            if (HERE / record["path"]).exists()
        }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {METADATA_PATH.relative_to(HERE)}")

    if failures:
        print(
            f"{len(failures)} download(s) failed: {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
