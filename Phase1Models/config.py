"""
config.py — HAB_Phase1 project-wide path configuration
=======================================================
Place this file at the project root:
    HAB_Phase1/config.py

All scripts and notebooks import from here so paths are defined
in exactly one place.  To move the project, only this file changes.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — the folder that contains THIS file
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "UCDavis_Data_for_Analysis2_220days_ToxinProdMicrocystis.csv"

# 10-row illustrative sample used by the test suite
EXAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "example_data.csv"

# ---------------------------------------------------------------------------
# Output directories — one subfolder per pipeline stage
# ---------------------------------------------------------------------------
OUTPUT_DIRS = {
    "01": PROJECT_ROOT / "outputs" / "01_FC_4Models_Results",
    "02": PROJECT_ROOT / "outputs" / "02_FC_ANN_Results",
    "03": PROJECT_ROOT / "outputs" / "03_HS_CV_5ML",
    "04": PROJECT_ROOT / "outputs" / "04_ML5Models_Analysis_Results",
    "05": PROJECT_ROOT / "outputs" / "05_PDP_Results",
    "06": PROJECT_ROOT / "outputs" / "06_LOSO_Results",
}

# ---------------------------------------------------------------------------
# Test output directories — written to outputs/test_runs/ (never committed)
# ---------------------------------------------------------------------------
TEST_OUTPUT_DIRS = {
    "01": PROJECT_ROOT / "outputs" / "test_runs" / "01_FC_4Models_Results",
    "02": PROJECT_ROOT / "outputs" / "test_runs" / "02_FC_ANN_Results",
    "03": PROJECT_ROOT / "outputs" / "test_runs" / "03_HS_CV_5ML",
    "04": PROJECT_ROOT / "outputs" / "test_runs" / "04_ML5Models_Analysis_Results",
    "05": PROJECT_ROOT / "outputs" / "test_runs" / "05_PDP_Results",
    "06": PROJECT_ROOT / "outputs" / "test_runs" / "06_LOSO_Results",
}
