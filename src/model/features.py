"""Feature pipeline definition (docs/02-MODEL-SPEC.md section 1).

Every transformation lives inside the sklearn Pipeline. Nothing is transformed
outside it. That is what makes the what-if path safe: a hypothetical customer
goes through byte-identical preprocessing to a training row, with no chance of
the app and the model disagreeing about encoding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor() -> ColumnTransformer:
    """Numeric scaling + one-hot encoding, keyed by the groups in config."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "bin",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                config.BINARY_FEATURES,
            ),
            (
                "cat",
                # drop="first" avoids the dummy trap for the multi-level columns.
                # handle_unknown must be "infrequent_if_exist" rather than "ignore"
                # because sklearn forbids combining drop="first" with "ignore".
                OneHotEncoder(
                    drop="first",
                    handle_unknown="infrequent_if_exist",
                    sparse_output=False,
                    min_frequency=1,
                ),
                config.MULTICLASS_FEATURES,
            ),
        ],
        remainder="drop",  # customerID, Churn, tenure_bucket never reach the model
        verbose_feature_names_out=False,
    )


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a frame to the model's feature columns, in a stable order.

    Enforces C4: customerID is a lookup key, never a feature. Selecting by an
    explicit list (rather than dropping known-bad columns) means a newly added
    column cannot leak into the model by accident.
    """
    missing = [c for c in config.MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing required feature columns: {missing}")
    return df[config.MODEL_FEATURES].copy()


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Output column names of a fitted preprocessor."""
    return [str(n) for n in preprocessor.get_feature_names_out()]


def map_output_to_source(output_name: str) -> tuple[str, str | None]:
    """Map an encoded feature name back to (source column, level).

    One-hot columns come out as 'Contract_Two year'; numeric ones keep their
    name. top_factors needs the source column and the level to say
    "Contract = Two year" rather than quoting an encoder artefact.

    Matching is longest-source-name-first so 'TotalCharges' is not shadowed by
    a prefix match on another column.
    """
    if output_name in config.NUMERIC_FEATURES:
        return output_name, None

    candidates = sorted(config.BINARY_FEATURES + config.MULTICLASS_FEATURES, key=len, reverse=True)
    for source in candidates:
        if output_name == source:
            # Binary column collapsed to a single indicator by drop="if_binary".
            return source, None
        if output_name.startswith(source + "_"):
            return source, output_name[len(source) + 1 :]
    return output_name, None


def training_means(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Mean of every encoded feature across the training set.

    Needed for the per-customer attribution in service.py: a contribution is
    coef_j * (x_ij - mean_j), so the means define the reference point that
    "average customer" means.
    """
    return np.asarray(preprocessor.transform(X)).mean(axis=0)
