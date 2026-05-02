from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEGRADED_DATA_DIR = DATA_DIR / "degraded"

RESULTS_DIR = PROJECT_ROOT / "results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
DEGRADATION_RESULTS_DIR = RESULTS_DIR / "degradation"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
