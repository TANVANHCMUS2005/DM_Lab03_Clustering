"""Evaluation utilities for the clustering reproduction experiments."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


STRONGER_THRESHOLD = 0.501
WEAKER_THRESHOLD = 0.499


def classify_strength_relation(win_values: np.ndarray) -> np.ndarray:
    """Convert win values to {weaker, same, stronger} labels."""

    win_values = np.asarray(win_values, dtype=float)
    return np.where(
        win_values > STRONGER_THRESHOLD,
        1,
        np.where(win_values < WEAKER_THRESHOLD, -1, 0),
    )


def strength_relation_accuracy(
    predicted_win_values: np.ndarray,
    reference_win_values: np.ndarray,
) -> float:
    """Compute the strength relation accuracy used in the paper."""

    predicted_labels = classify_strength_relation(predicted_win_values)
    reference_labels = classify_strength_relation(reference_win_values)
    return float(np.mean(predicted_labels == reference_labels))


def compute_pairwise_ground_truth(
    dataframe: pd.DataFrame,
    *,
    comp_a_col: str = "comp_a",
    comp_b_col: str = "comp_b",
    label_col: str = "label",
) -> pd.DataFrame:
    """Aggregate match records into tabular PairWin ground truth.

    The paper defines the ground-truth strength relation from the average win
    value of each matchup. This function computes that average and joins it back
    to every record.
    """

    grouped = (
        dataframe.groupby([comp_a_col, comp_b_col], sort=False)[label_col]
        .mean()
        .rename("pairwin_value")
        .reset_index()
    )
    return dataframe.merge(grouped, on=[comp_a_col, comp_b_col], how="left")


def pairwin_table(
    dataframe: pd.DataFrame,
    *,
    comp_a_col: str = "comp_a",
    comp_b_col: str = "comp_b",
    label_col: str = "label",
) -> pd.Series:
    """Build a PairWin lookup table for one split only.

    The returned Series is indexed by matchup tuples `(comp_a, comp_b)` and can
    be used to derive clean train/test ground-truth labels inside each fold.
    """

    return dataframe.groupby([comp_a_col, comp_b_col], sort=False)[label_col].mean()


def attach_pairwin_from_table(
    dataframe: pd.DataFrame,
    lookup: pd.Series,
    *,
    comp_a_col: str = "comp_a",
    comp_b_col: str = "comp_b",
    output_col: str = "pairwin_value",
) -> pd.DataFrame:
    """Attach PairWin values from a precomputed lookup table to one dataframe."""

    result = dataframe.copy()
    keys = list(zip(result[comp_a_col], result[comp_b_col]))
    result[output_col] = [lookup.loc[key] for key in keys]
    return result


def unsupervised_clustering_scores(
    latent_features: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict[str, float]:
    """Compute the three clustering metrics required by the lab."""

    latent_features = np.asarray(latent_features)
    cluster_labels = np.asarray(cluster_labels)

    unique_labels = np.unique(cluster_labels)
    if latent_features.ndim != 2:
        raise ValueError("latent_features must be a 2D array.")
    if len(latent_features) != len(cluster_labels):
        raise ValueError("latent_features and cluster_labels must have the same length.")
    if len(unique_labels) < 2 or len(unique_labels) >= len(cluster_labels):
        return {
            "silhouette_score": np.nan,
            "davies_bouldin_index": np.nan,
            "calinski_harabasz_index": np.nan,
        }

    return {
        "silhouette_score": float(silhouette_score(latent_features, cluster_labels)),
        "davies_bouldin_index": float(davies_bouldin_score(latent_features, cluster_labels)),
        "calinski_harabasz_index": float(calinski_harabasz_score(latent_features, cluster_labels)),
    }


def codebook_utilization(cluster_labels: Iterable[int], num_embeddings: int) -> Dict[str, float]:
    """Summarize how much of the VQ codebook is actually used."""

    labels = np.asarray(list(cluster_labels), dtype=int)
    used = int(np.unique(labels).size)
    return {
        "used_codes": used,
        "total_codes": int(num_embeddings),
        "utilization_ratio": float(used / max(num_embeddings, 1)),
    }


def relative_deviation(reproduced: float, reported: float) -> float:
    """Compute the relative deviation required for the comparison table."""

    if np.isclose(reported, 0.0):
        return np.nan
    return float((reproduced - reported) / reported)
