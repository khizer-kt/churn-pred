"""Central configuration: paths, column groups, and constants.

Everything that another module might otherwise hardcode lives here. Import from
this module rather than repeating literals -- the agent, the model and the app
must agree on column names and category values exactly.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "Customer-Churn.csv"
PROCESSED_PARQUET = DATA_DIR / "processed" / "churn_clean.parquet"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURE_META_PATH = ARTIFACTS_DIR / "feature_meta.json"

# --------------------------------------------------------------------------
# Dataset shape -- asserted on load, see docs/01-DATA-FINDINGS.md sec.4
# --------------------------------------------------------------------------
EXPECTED_ROWS = 7043

ID_COL = "customerID"
TARGET_COL = "Churn"

# Columns carrying the "No internet service" sentinel (finding I4).
INTERNET_ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
# Column carrying the "No phone service" sentinel (finding I4).
PHONE_ADDON_COLS = ["MultipleLines"]

SENTINEL_NO_INTERNET = "No internet service"
SENTINEL_NO_PHONE = "No phone service"

# --------------------------------------------------------------------------
# Feature groups -- consumed by src/model/features.py
# --------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
]

BINARY_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

MULTICLASS_FEATURES = [
    "InternetService",
    "Contract",
    "PaymentMethod",
]

MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + MULTICLASS_FEATURES

# Derived, for EDA and agent segment queries only -- never fed to the model (C9).
EDA_ONLY_COLS = ["tenure_bucket"]

TENURE_BUCKETS = [
    (0, 6, "0-6 months"),
    (6, 12, "6-12 months"),
    (12, 24, "1-2 years"),
    (24, 48, "2-4 years"),
    (48, 72, "4-6 years"),
    (72, 10_000, "6+ years"),
]

# --------------------------------------------------------------------------
# Modelling constants
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --- Cost model driving threshold selection (docs/02-MODEL-SPEC.md sec.4.4) ---
# Stated as its components rather than one opaque ratio, because the ratio is an
# assumption and a reader should be able to see and challenge each part.
CLV_MARGIN = 500.0        # margin lost when a customer leaves, USD
OFFER_COST = 50.0         # cost of one retention offer + outreach, USD
OFFER_SUCCESS_RATE = 0.30 # fraction of correctly-flagged churners actually saved

# Catching a churner is only worth what the intervention recovers -- an offer
# that works 30% of the time recovers 30% of the margin, not all of it. Omitting
# this term is the usual reason churn models are tuned to flag implausibly large
# fractions of the customer base.
COST_FN_OVER_FP = (CLV_MARGIN * OFFER_SUCCESS_RATE) / OFFER_COST   # = 3.0
COST_RATIO_SENSITIVITY = [1.0, 3.0, 5.0, 10.0, 20.0]

# A candidate within this much PR-AUC of the best is treated as tied on ranking,
# and the tie is broken on calibration. See train.py.
PR_AUC_TIE_TOLERANCE = 0.01

# Risk bands applied to predicted probabilities for human-readable output.
RISK_BANDS = [(0.0, 0.33, "Low"), (0.33, 0.66, "Medium"), (0.66, 1.01, "High")]

# --------------------------------------------------------------------------
# Agent limits -- see docs/03-AGENT-SPEC.md sec.5 and sec.7
# --------------------------------------------------------------------------
MAX_TOOL_STEPS = 6
MAX_REPLANS = 1
MAX_ANSWER_RETRIES = 1
EXEC_TIMEOUT_SECONDS = 5
EXEC_MAX_ROWS = 50
EXEC_MAX_CHARS = 4000


def risk_band(score: float) -> str:
    """Map a probability to a Low/Medium/High label."""
    for lo, hi, name in RISK_BANDS:
        if lo <= score < hi:
            return name
    return "Unknown"
