"""Summarises the ColH hits from Foldseek (PDB) and MMseqs2 (Logan50) and renders the report.

Reads outputs/ (including run_metadata.tsv from fetch_run_metadata.py and the UniRef50 taxonomy of the Logan50
hits), fills the data block of report/template.html and writes report/colh_homolog_report.html.
Run from the experiment root with any Python that has polars: python scripts/build_report.py
"""

import json
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

QUERY_LENGTH = 1021
# 9LRK chain A models UniProt residues 45-808 without gaps, so structure residue i is UniProt i + 44
STRUCTURE_OFFSET = 44
# UniProt Q46085 regions; the helper subdomain (609-721) is cut at 717 where PKD1 (S2a, 718-810) begins
DOMAINS = [
    {"key": "signal", "name": "Signal peptide", "start": 1, "end": 30},
    {"key": "propeptide", "name": "Propeptide", "start": 31, "end": 40},
    {"key": "activator", "name": "Activator domain", "start": 41, "end": 320},
    {"key": "catalytic", "name": "Catalytic subdomain", "start": 330, "end": 601},
    {"key": "helper", "name": "Helper subdomain", "start": 609, "end": 717},
    {"key": "pkd1", "name": "PKD1 (S2a)", "start": 718, "end": 810},
    {"key": "pkd2", "name": "PKD2 (S2b)", "start": 811, "end": 904},
    {"key": "cbd", "name": "Collagen-binding domain (S3)", "start": 905, "end": 1021},
]
MATURE_DOMAINS = [d for d in DOMAINS if d["key"] not in ("signal", "propeptide")]
DOMAIN_SHORT = {
    "activator": "Activator",
    "catalytic": "Catalytic",
    "helper": "Helper",
    "pkd1": "PKD1",
    "pkd2": "PKD2",
    "cbd": "CBD",
}


def covered_domains(qs: int, qe: int) -> list[str]:
    """Mature ColH domains an alignment covers by at least half of their length."""
    keys = []
    for d in MATURE_DOMAINS:
        overlap = min(qe, d["end"]) - max(qs, d["start"]) + 1
        if overlap >= 0.5 * (d["end"] - d["start"] + 1):
            keys.append(d["key"])
    return keys


def architecture(qs: int, qe: int) -> str:
    keys = covered_domains(qs, qe)
    return (
        " + ".join(DOMAIN_SHORT[k] for k in keys)
        if keys
        else "Under half of any domain"
    )


def hit_class(qs: int, qe: int) -> str:
    """ColH unit a hit stands for: the peptidase unit wins over the collagen-binding domain over PKD."""
    keys = set(covered_domains(qs, qe))
    if keys & {"activator", "catalytic", "helper"}:
        return "peptidase"
    if "cbd" in keys:
        return "cbd"
    return "pkd" if keys else "other"


def coverage_depth(starts: list[int], ends: list[int]) -> list[int]:
    """Number of alignments covering each query residue 1..QUERY_LENGTH."""
    diff = [0] * (QUERY_LENGTH + 2)
    for s, e in zip(starts, ends, strict=True):
        diff[s] += 1
        diff[e + 1] -= 1
    depth, running = [], 0
    for pos in range(1, QUERY_LENGTH + 1):
        running += diff[pos]
        depth.append(running)
    return depth


def foldseek_group(header: str) -> str:
    text = header.lower()
    if "collagenase" not in text:
        return "other"
    return "m9a" if "vibrio" in text or "grimontia" in text else "m9b"


def summarise_foldseek(path: Path) -> dict:
    df = pl.read_csv(path, separator="\t", quote_char=None).with_columns(
        qs=pl.col("qstart") + STRUCTURE_OFFSET,
        qe=pl.col("qend") + STRUCTURE_OFFSET,
        pdb=pl.col("target").str.slice(0, 4).str.to_uppercase(),
        chain=pl.col("target").str.extract(r"_([^_]+)$"),
        title=pl.col("theader").str.replace(r"^\S+\s+", ""),
    )
    df = df.with_columns(
        group=pl.col("theader").map_elements(foldseek_group, return_dtype=pl.String),
        arch=pl.struct("qs", "qe").map_elements(
            lambda r: architecture(r["qs"], r["qe"]), return_dtype=pl.String
        ),
        cls=pl.struct("qs", "qe").map_elements(
            lambda r: hit_class(r["qs"], r["qe"]), return_dtype=pl.String
        ),
    )
    confident = df.filter(pl.col("prob") >= 0.9).sort("bits", descending=True)
    columns = [
        "pdb", "chain", "title", "taxname", "group", "cls", "arch", "qs", "qe", "tlen",
        "fident", "alntmscore", "lddt", "prob", "evalue", "bits",
    ]  # fmt: skip
    return {
        "counts": {
            "hits": df.height,
            "e_1e3": df.filter(pl.col("evalue") <= 1e-3).height,
            "prob_09": confident.height,
            "tm_05": df.filter(pl.col("alntmscore") >= 0.5).height,
            "groups": dict(confident.group_by("group").len().iter_rows()),
        },
        "depth": coverage_depth(confident["qs"].to_list(), confident["qe"].to_list()),
        "hits": confident.select(columns).to_dicts(),
    }


def top_counts(
    series: pl.Series, n: int, missing: str = "No organism in ENA"
) -> list[dict]:
    counts = series.fill_null(missing).value_counts(sort=True)
    head = counts.head(n).rows()
    rest = int(counts["count"][n:].sum())
    other = {"name": "Other", "n": rest, "k": counts.height - n}
    return [{"name": k, "n": v} for k, v in head] + ([other] if rest else [])


def read_taxonomy(path: Path) -> pl.DataFrame:
    """Per-hit LCA from easy-taxonomy --tax-lineage 1, with phylum and genus taken from the lineage."""
    lca = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        quote_char=None,
        infer_schema=False,
        new_columns=["target", "taxid", "tax_rank", "taxon", "lineage"],
    )
    unclassified = pl.col("taxid") == "0"
    return lca.with_columns(
        taxon=pl.when(unclassified).then(pl.lit("Unclassified")).otherwise("taxon"),
        tax_rank=pl.when(unclassified).then(None).otherwise("tax_rank"),
        superkingdom=pl.col("lineage").str.extract(
            r"-_(Bacteria|Archaea|Eukaryota|Viruses)(?:;|$)"
        ),
        phylum=pl.when(unclassified)
        .then(pl.lit("Unclassified"))
        .otherwise(
            pl.col("lineage")
            .str.extract(r"(?:^|;)p_([^;]+)")
            .fill_null("Above phylum level")
        ),
        genus=pl.when(unclassified)
        .then(pl.lit("Unclassified"))
        .otherwise(
            pl.col("lineage")
            .str.extract(r"(?:^|;)g_([^;]+)")
            .fill_null("Above genus level")
        ),
    ).drop("lineage")


def read_closest(path: Path) -> pl.DataFrame:
    """Closest UniRef50 entry of each hit, from the plain search in closest_uniref50.sbatch."""
    return pl.read_csv(path, separator="\t", quote_char=None).select(
        target="query",
        top_uniref="target",
        top_fident="fident",
        top_qcov="qcov",
        top_taxon="taxname",
    )


def summarise_logan(path: Path, metadata: Path, taxonomy: Path) -> dict:
    meta = pl.read_csv(metadata, separator="\t", infer_schema=False)
    df = (
        pl.read_csv(path, separator="\t", quote_char=None)
        .with_columns(run=pl.col("target").str.extract(r"^([A-Z]+\d+)_"))
        .join(meta, left_on="run", right_on="run_accession", how="left")
        .join(read_taxonomy(taxonomy), on="target", how="left")
        .join(
            read_closest(taxonomy.with_name("logan50_hits_closest_uniref50.tsv")),
            on="target",
            how="left",
        )
        .with_columns(
            cls=pl.struct("qstart", "qend").map_elements(
                lambda r: hit_class(r["qstart"], r["qend"]), return_dtype=pl.String
            ),
            arch=pl.struct("qstart", "qend").map_elements(
                lambda r: architecture(r["qstart"], r["qend"]), return_dtype=pl.String
            ),
        )
        .sort("bits", descending=True)
    )
    archs = (
        df.group_by("arch", "cls")
        .len()
        .sort("len", descending=True)
        .rename({"len": "n"})
        .to_dicts()
    )
    full = df.filter(pl.col("arch").str.count_matches(r"\+") == 5)
    peptidase = df.filter(pl.col("cls") == "peptidase")
    sources = (
        df["scientific_name"].fill_null("No organism in ENA").unique().sort().to_list()
    )
    source_index = {name: i for i, name in enumerate(sources)}
    taxa = df["taxon"].unique().sort().to_list()
    taxon_index = {name: i for i, name in enumerate(taxa)}
    points = [
        [r["fident"], r["qcov"], r["cls"], r["target"], r["tlen"], r["evalue"],
         source_index[r["scientific_name"] or "No organism in ENA"], taxon_index[r["taxon"]]]
        for r in df.select(
            "fident", "qcov", "cls", "target", "tlen", "evalue", "scientific_name", "taxon"
        ).iter_rows(named=True)
    ]  # fmt: skip
    columns = [
        "target", "run", "scientific_name", "library_source", "taxon", "tax_rank", "top_uniref",
        "top_taxon", "top_fident", "top_qcov", "tlen", "fident", "qcov", "qstart", "qend", "arch", "evalue", "bits",
    ]  # fmt: skip
    classified = df.filter(pl.col("taxon") != "Unclassified")
    return {
        "counts": {
            "hits": df.height,
            "runs": df["run"].n_unique(),
            "runs_resolved": df.filter(pl.col("scientific_name").is_not_null())[
                "run"
            ].n_unique(),
            "classes": dict(df.group_by("cls").len().iter_rows()),
            "full_length": full.height,
            "best_full": full.select(
                "target", "fident", "qcov", "tlen", "scientific_name"
            ).row(0, named=True),
            "e_1e10": df.filter(pl.col("evalue") <= 1e-10).height,
            "library": dict(
                df.group_by(pl.col("library_source").fill_null("Run not in ENA"))
                .len()
                .iter_rows()
            ),
        },
        "depth": {
            cls: coverage_depth(part["qstart"].to_list(), part["qend"].to_list())
            for (cls,), part in df.group_by("cls")
        },
        "archs": archs,
        "sources_all": top_counts(df["scientific_name"], 12),
        "sources_peptidase": top_counts(peptidase["scientific_name"], 12),
        "sources": sources,
        "taxonomy": {
            "classified": classified.height,
            "superkingdoms": dict(
                classified.group_by(
                    pl.col("superkingdom").fill_null("Above superkingdom level")
                )
                .len()
                .iter_rows()
            ),
            "phylum_all": top_counts(df["phylum"], 12),
            "genus_peptidase": top_counts(peptidase["genus"], 12),
        },
        "taxa": taxa,
        "points": points,
        "peptidase_hits": peptidase.select(columns).to_dicts(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = {
        "query": {"length": QUERY_LENGTH, "domains": DOMAINS, "resolved": [45, 808]},
        "foldseek": summarise_foldseek(Path("outputs/foldseek_pdb/9LRK_vs_pdb.tsv")),
        "logan": summarise_logan(
            Path("outputs/mmseqs_logan50/Q46085_vs_logan50.tsv.gz"),
            Path("outputs/mmseqs_logan50/run_metadata.tsv"),
            Path("outputs/taxonomy_uniref50/logan50_hits_lca.tsv"),
        ),
    }
    template = Path("report/template.html").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = Path("report/colh_homolog_report.html")
    out.write_text(template.replace("__REPORT_DATA__", payload))
    logger.info("wrote %s (%d kB)", out, out.stat().st_size // 1024)


if __name__ == "__main__":
    main()
