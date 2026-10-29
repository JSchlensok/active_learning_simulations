"""Fetches ENA metadata for the SRA runs behind the Logan50 hits.

Logan50 headers name the source run (`<run>_<contig>_<gene>`); ENA's portal API resolves runs to organism or
metagenome type and library source. Writes outputs/mmseqs_logan50/run_metadata.tsv.
Run from the experiment root with any Python that has polars: python scripts/fetch_run_metadata.py
"""

import io
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

ENA_SEARCH = "https://www.ebi.ac.uk/ena/portal/api/search"
FIELDS = [
    "run_accession", "scientific_name", "tax_id", "library_source", "library_strategy",
    "study_accession", "environment_biome", "isolation_source", "host",
]  # fmt: skip
CHUNK = 500


def fetch(runs: list[str]) -> pl.DataFrame:
    body = urllib.parse.urlencode(
        {
            "result": "read_run",
            "includeAccessionType": "run",
            "includeAccessions": ",".join(runs),
            "fields": ",".join(FIELDS),
            "format": "tsv",
        }
    ).encode()
    with urllib.request.urlopen(
        urllib.request.Request(ENA_SEARCH, data=body), timeout=120
    ) as response:
        text = response.read().decode()
    return pl.read_csv(
        io.StringIO(text), separator="\t", quote_char=None, infer_schema=False
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    hits = pl.read_csv(
        "outputs/mmseqs_logan50/Q46085_vs_logan50.tsv.gz",
        separator="\t",
        quote_char=None,
        columns=["target"],
    )
    runs = hits["target"].str.extract(r"^([A-Z]+\d+)_").unique().sort().to_list()
    frames = []
    for i in range(0, len(runs), CHUNK):
        frames.append(fetch(runs[i : i + CHUNK]))
        logger.info("fetched %d / %d runs", min(i + CHUNK, len(runs)), len(runs))
    meta = pl.concat(frames).unique("run_accession").sort("run_accession")
    out = Path("outputs/mmseqs_logan50/run_metadata.tsv")
    meta.write_csv(out, separator="\t")
    logger.info("%d of %d runs resolved; wrote %s", meta.height, len(runs), out)


if __name__ == "__main__":
    main()
