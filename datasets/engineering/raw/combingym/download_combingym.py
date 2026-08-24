"""Download combinatorial mutagenesis datasets from CombinGym (https://combingym.org).

CombinGym is not on HuggingFace; its Vue frontend talks to a small JSON API that also serves the
per-dataset files. Two endpoints are used here:

* ``GET /protein/browse/getBrowseList``            -> the dataset table behind https://combingym.org/#/browse
* ``GET /protein/common/download?fileName=<name>`` -> one file, by its (timestamped) server-side name

The server-side file names carry an upload timestamp (``GB1_20251021100854A049.fasta``) and are
therefore not stable, so this script always resolves them from the browse listing instead of
hardcoding them. Files land under ``<protein>/<protein>_<kind>.<ext>`` with the timestamp dropped,
and every download is recorded in ``metadata.json`` (browse row + server file name + sha256) so the
raw data stays traceable to its source.

Usage (from this directory):

    uv run python download_combingym.py                        # WT FASTA + DMS xlsx, default proteins
    uv run python download_combingym.py --include msa structure
    uv run python download_combingym.py --proteins GB1 CR9114 --force
    uv run python download_combingym.py --list                  # show every dataset CombinGym offers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://combingym.org/protein"
BROWSE_ENDPOINT = f"{BASE_URL}/browse/getBrowseList"
DOWNLOAD_ENDPOINT = f"{BASE_URL}/common/download"
TIMEOUT_SECONDS = 120

DEFAULT_PROTEINS = ["GB1", "CreiLOV", "CR9114", "mTagBFP2", "SaCas9"]

HERE = Path(__file__).parent
METADATA_PATH = HERE / "metadata.json"


@dataclass(frozen=True)
class FileKind:
    """One downloadable artifact per dataset, as named in the browse listing."""

    kind: str  # local name suffix, e.g. "wt" -> GB1_wt.fasta
    browse_field: str  # key in the browse row holding the server-side file name
    magic: bytes | None  # expected leading bytes, for a cheap sanity check


FILE_KINDS: dict[str, FileKind] = {
    "sequence": FileKind("wt", "seqWtSeq", b">"),
    "dms": FileKind("dms", "dmsDataset", b"PK"),  # xlsx is a zip container
    "msa": FileKind("msa", "msaFile", b">"),
    "structure": FileKind("structure", "structure", None),
}
DEFAULT_INCLUDE = ["sequence", "dms"]


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_browse_rows() -> list[dict]:
    """Return the dataset table backing https://combingym.org/#/browse."""
    payload = _get_json(BROWSE_ENDPOINT)
    if payload.get("code") != 200:
        raise RuntimeError(
            f"CombinGym browse API returned code={payload.get('code')}: {payload.get('msg')}"
        )
    # The API nests the listing under "data", while an empty top-level "rows" key is also present.
    rows = (payload.get("data") or {}).get("rows") or payload.get("rows") or []
    if not rows:
        raise RuntimeError("CombinGym browse API returned no datasets.")
    return rows


def download_file(
    server_file_name: str, target_path: Path, expected_magic: bytes | None
) -> bytes:
    """Download one file into ``target_path`` and return its bytes."""
    url = f"{DOWNLOAD_ENDPOINT}?fileName={urllib.parse.quote(server_file_name)}&delete=false"
    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        content = response.read()

    if not content:
        raise RuntimeError(f"{server_file_name}: server returned an empty body.")
    if expected_magic and not content.startswith(expected_magic):
        # The endpoint answers with a JSON error body (HTTP 200) for unknown file names.
        preview = content[:200].decode("utf-8", errors="replace")
        raise RuntimeError(f"{server_file_name}: unexpected content, got {preview!r}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(content)
    return content


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--proteins",
        nargs="+",
        default=DEFAULT_PROTEINS,
        help=f"Protein names as listed by CombinGym (default: {' '.join(DEFAULT_PROTEINS)})",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=sorted(FILE_KINDS),
        default=DEFAULT_INCLUDE,
        help=f"Which artifacts to fetch per dataset (default: {' '.join(DEFAULT_INCLUDE)})",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available datasets and exit"
    )
    args = parser.parse_args()

    try:
        rows = fetch_browse_rows()
    except (urllib.error.URLError, RuntimeError) as error:
        print(f"Could not read the CombinGym dataset listing: {error}", file=sys.stderr)
        return 1

    rows_by_name = {row["protName"]: row for row in rows}

    if args.list:
        print(f"{len(rows)} datasets on CombinGym:")
        for name, row in rows_by_name.items():
            print(
                f"  {name:<10} {row['property']:<20} len={row['seqLen']:<5} "
                f"sites={row['dmsSiteNum']:<3} measured={row['dmsMeasured']}"
            )
        return 0

    unknown = [name for name in args.proteins if name not in rows_by_name]
    if unknown:
        print(
            f"Unknown protein name(s): {', '.join(unknown)}. "
            f"Available: {', '.join(rows_by_name)}",
            file=sys.stderr,
        )
        return 1

    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    failures: list[str] = []

    for protein in args.proteins:
        row = rows_by_name[protein]
        entry = metadata.setdefault(protein, {"browse_row": {}, "files": {}})
        # Keep the descriptive columns only; the filter/pagination fields are all null noise.
        entry["browse_row"] = {
            key: value
            for key, value in row.items()
            if not key.endswith(("Min", "Max")) and key not in {"userId", "userName"}
        }
        entry["source_url"] = "https://combingym.org/#/browse"

        for include in args.include:
            file_kind = FILE_KINDS[include]
            server_file_name = row.get(file_kind.browse_field)
            if not server_file_name:
                print(f"  {protein}: no {include} file listed, skipping")
                continue

            suffix = Path(server_file_name).suffix
            target_path = HERE / protein / f"{protein}_{file_kind.kind}{suffix}"

            if target_path.exists() and not args.force:
                print(f"  {target_path.relative_to(HERE)} exists, skipping")
                continue

            try:
                content = download_file(server_file_name, target_path, file_kind.magic)
            except (urllib.error.URLError, RuntimeError) as error:
                print(f"  {protein} {include}: FAILED ({error})", file=sys.stderr)
                failures.append(f"{protein}/{include}")
                continue

            entry["files"][include] = {
                "path": str(target_path.relative_to(HERE)),
                "server_file_name": server_file_name,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            print(
                f"  {target_path.relative_to(HERE)} <- {server_file_name} ({len(content):,} bytes)"
            )

        # Drop records of files that are no longer on disk, so metadata.json never claims more than it can back.
        entry["files"] = {
            kind: record
            for kind, record in entry["files"].items()
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
