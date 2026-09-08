from __future__ import annotations

from typing import Any

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# FEATURE GROUPS
# ============================================================

MARKET_FEATURES = [
    "return_5d",
    "return_20d",
    "return_60d",
    "price_vs_20d_ma",
    "price_vs_50d_ma",
    "momentum_20d",
    "volatility_20d",
]


SEC_FEATURES = [
    "sec_filings_5d",
    "sec_filings_20d",
    "sec_filings_60d",

    "sec_10k_recent",
    "sec_10q_recent",
    "sec_8k_recent",

    "sec_filing_velocity",
    "sec_filing_acceleration",

    "sec_8k_ratio",
    "sec_10q_ratio",

    "days_since_last_10k",
    "days_since_last_10q",
]

NEWS_FEATURES = [
    "news_event_count",
    "average_sentiment",
    "positive_news_count",
    "negative_news_count",
    "neutral_news_count",
    "unique_news_sources",
    "sentiment_std",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "sentiment_strength",
    "source_diversity",
]


MACRO_FEATURES = [
    "dgs10",
    "dgs3mo",
]


# ============================================================
# ABLATION FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "MARKET ONLY": MARKET_FEATURES,

    "SEC V3 ONLY": SEC_FEATURES,

    "MARKET + SEC V3": (
        MARKET_FEATURES
        + SEC_FEATURES
    ),

    "MARKET + MACRO": (
        MARKET_FEATURES
        + MACRO_FEATURES
    ),

    "MARKET + SEC V3 + MACRO": (
        MARKET_FEATURES
        + SEC_FEATURES
        + MACRO_FEATURES
    ),

    "NEWS V2 ONLY": NEWS_FEATURES,

    "MARKET + NEWS V2": (
        MARKET_FEATURES
        + NEWS_FEATURES
    ),

    "ALL V3": (
        MARKET_FEATURES
        + SEC_FEATURES
        + NEWS_FEATURES
        + MACRO_FEATURES
    ),
}
# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataset(
    dataset: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert the generated dataset into a clean ML dataframe.

    All feature columns are converted to numeric values.
    Constant columns are removed globally.
    """

    df = pd.DataFrame(dataset)

    required_columns = [
        "target_direction",
        "as_of",
    ]

    # All features potentially used by any experiment.
    all_features = sorted(
        set(
            MARKET_FEATURES
            + SEC_FEATURES
            + NEWS_FEATURES
            + MACRO_FEATURES
        )
    )

    required_columns.extend(
        all_features
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    df["as_of"] = pd.to_datetime(
        df["as_of"],
        errors="coerce",
    )

    if df["as_of"].isna().any():
        raise ValueError(
            "Invalid values found in 'as_of'."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "as_of"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Convert features to numeric
    # --------------------------------------------------------

    for column in all_features:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove globally constant features
    # --------------------------------------------------------

    constant_columns = [
        column
        for column in all_features
        if df[column].nunique(
            dropna=True
        ) <= 1
    ]

    if constant_columns:
        print(
            "Removing constant features:",
            constant_columns,
        )

        df = df.drop(
            columns=constant_columns
        )

    return df


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
    requested_features: list[str],
) -> list[str]:
    """
    Return requested features that are actually available
    in the cleaned dataframe.
    """

    feature_columns = [
        column
        for column in requested_features
        if column in df.columns
    ]

    if not feature_columns:
        raise ValueError(
            "No usable feature columns found."
        )

    return feature_columns


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically to prevent temporal leakage.
    """

    split_index = int(
        len(df) * train_ratio
    )

    if (
        split_index <= 0
        or split_index >= len(df)
    ):
        raise ValueError(
            "Invalid train ratio."
        )

    train_df = df.iloc[
        :split_index
    ].copy()

    test_df = df.iloc[
        split_index:
    ].copy()

    return train_df, test_df


# ============================================================
# TRAIN BASELINE
# ============================================================

def train_baseline(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[
    StandardScaler,
    LogisticRegression,
]:
    """
    Train standardized logistic regression.

    The scaler is fitted ONLY on the training data.
    Class balancing prevents collapse toward the majority class.
    """

    feature_columns = get_feature_columns(
        train_df,
        feature_columns,
    )

    X_train = (
        train_df[
            feature_columns
        ]
        .fillna(0)
    )

    y_train = train_df[
        "target_direction"
    ]

    # --------------------------------------------------------
    # Scale using training data only
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    # --------------------------------------------------------
    # Logistic regression
    # --------------------------------------------------------

    model = LogisticRegression(
        C=0.01,
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    return scaler, model
# ============================================================
# EVALUATE BASELINE
# ============================================================

def evaluate_baseline(
    scaler: StandardScaler,
    model: LogisticRegression,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, Any]:
    """
    Evaluate the model on future unseen observations.
    """

    feature_columns = get_feature_columns(
        test_df,
        feature_columns,
    )

    X_test = (
        test_df[
            feature_columns
        ]
        .fillna(0)
    )

    y_test = test_df[
        "target_direction"
    ]

    # --------------------------------------------------------
    # Apply training scaler
    # --------------------------------------------------------

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_scaled
    )

    probabilities = model.predict_proba(
        X_test_scaled
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),

        "confusion_matrix": (
            confusion_matrix(
                y_test,
                predictions,
            ).tolist()
        ),
    }

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    if y_test.nunique() > 1:
        metrics["roc_auc"] = (
            roc_auc_score(
                y_test,
                probabilities,
            )
        )
    else:
        metrics["roc_auc"] = None

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    metrics[
        "classification_report"
    ] = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Probability diagnostics
    # --------------------------------------------------------

    metrics[
        "probability_min"
    ] = float(
        probabilities.min()
    )

    metrics[
        "probability_max"
    ] = float(
        probabilities.max()
    )

    metrics[
        "probability_mean"
    ] = float(
        probabilities.mean()
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    prediction_counts = (
        pd.Series(
            predictions
        )
        .value_counts()
        .sort_index()
        .to_dict()
    )

    metrics[
        "prediction_distribution"
    ] = prediction_counts

    # --------------------------------------------------------
    # Return predictions for diagnostics
    # --------------------------------------------------------

    metrics[
        "predictions"
    ] = predictions.tolist()

    metrics[
        "probabilities"
    ] = probabilities.tolist()

    return metrics