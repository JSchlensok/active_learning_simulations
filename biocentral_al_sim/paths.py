"""Shared locations for simulation results and projections.

Bump RUN_VERSION to the start date of a new run generation to give it fresh directories.
"""
from pathlib import Path

RUN_VERSION = "2026-11-05"

RESULTS_DIR = Path(f"simulation_{RUN_VERSION}_results")
PROJECTIONS_DIR = Path(f"simulation_{RUN_VERSION}_projections")
