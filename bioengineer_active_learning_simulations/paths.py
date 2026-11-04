"""Shared locations for simulation results and projections.

Bump RUN_VERSION to start a fresh set of directories for a new run generation.
"""
from pathlib import Path

RUN_VERSION = "v1"

RESULTS_DIR = Path(f"simulation_{RUN_VERSION}_results")
PROJECTIONS_DIR = Path(f"simulation_{RUN_VERSION}_projections")
