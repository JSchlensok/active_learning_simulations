# active_learning_simulations

Benchmarks **active learning screening campaigns** on protein data *in silico*, via the
[Biocentral API](https://github.com/biocentral/biocentral_api). Labels of a fully labelled dataset are
masked, a campaign starts from a seed set, and a surrogate iteratively picks what to "measure" next.

## Setup

Python 3.12.11 and [uv](https://docs.astral.sh/uv/). `biocentral-api` resolves to an **editable sibling
checkout** as `2.0` is required compared to `1.2.1` on PyPI.

```bash
git clone https://github.com/biocentral/biocentral_api.git ../biocentral_api && uv sync
uv run python run_simulations_v1.py       # run the grid, compress reports, project
uv run streamlit run al_sim_dashboard.py  # explore a compressed run
```

Embedding, training and simulation run on a **local**
[biocentral_server](https://github.com/biocentral/biocentral_server) 2.x (dockerized, run separately);
the hosted server reports 1.2.1 and is rejected by `wait_until_healthy()`. Run from the repo root, except
`dataset_preprocessor.py` and the compile scripts, which run from their own directory. Results are reused,
not recomputed, so runs are resumable — delete a file to force a rerun.

## Datasets
**Screening** — pools of unrelated sequences, i.e. the grid: `MELTOME_MAXIMIZE`/`MELTOME_MINIMIZE`
(20 360, melting temperature), `SCL` (11 517, `DISCRETE` `Peroxisome`), `AMYLASE` (3 924), `PHOT` (2 122,
ProteinGym DMS), `EXOTOX` (2 333, `DISCRETE` `EXOTOXIN`).

**Engineering** — single-parent combinatorial landscapes, registered but **not** in `all()`, so the grid is
unchanged; opt in via `engineering()`. They declare a wild type, so the mutation-aware splits apply:
`COMBINGYM_{GB1, CREILOV, CR9114, MTAGBFP2, SACAS9}` (1 296–165 428 variants) and `FLIP2_{TRPB, NUCB}`
(228 298 / 55 759; NucB is `DISCRETE` on `{2,3}` = "beats wild type"). FLIP2 ships no parent sequence, so
both are reconstructed — TrpB's is **Tm9D8\***, an engineered TmTrpB variant rather than wild-type TmTrpB.

```bash
cd datasets/screening/raw/exotox      && uv run python compile_exotox.py
cd datasets/engineering/raw/combingym && uv run python download_combingym.py && uv run python compile_combingym.py
cd datasets/engineering/raw/flip2     && uv run python download_flip2.py && uv run python compile_flip2.py
cd datasets && uv run python dataset_preprocessor.py   # drops sequences ≥ 2000 residues
```

Dataset-specific information is recorded in specific README files for [CombinGym](datasets/engineering/raw/combingym/README.md)
and [FLIP2](datasets/engineering/raw/flip2/README.md) READMEs. Adding a dataset = an `ALSimulatorDataset`
member plus a `_DATASET_DEFINITIONS` entry.