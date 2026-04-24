"""Synthetic datasets used to reproduce the paper's controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


Composition = Tuple[int, ...]


@dataclass(frozen=True)
class SimpleCombinationConfig:
    """Configuration for the paper's Simple Combination Game."""

    num_elements: int = 20
    team_size: int = 3
    num_matches: int = 100_000
    random_state: int = 42


def enumerate_compositions(
    num_elements: int = 20,
    team_size: int = 3,
) -> List[Composition]:
    """Enumerate all valid team compositions."""

    return [tuple(comp) for comp in combinations(range(1, num_elements + 1), team_size)]


def composition_score(composition: Sequence[int]) -> int:
    """Compute the sum-based score defined in the paper."""

    return int(sum(composition))


def win_probability(comp_a: Sequence[int], comp_b: Sequence[int]) -> float:
    """Compute P(comp_a > comp_b) for the Simple Combination Game."""

    score_a = composition_score(comp_a)
    score_b = composition_score(comp_b)
    return float((score_a ** 2) / ((score_a ** 2) + (score_b ** 2)))


def composition_to_multihot(
    composition: Sequence[int],
    num_elements: int = 20,
) -> np.ndarray:
    """Convert a composition to the paper's binary feature representation."""

    feature = np.zeros(num_elements, dtype=np.float32)
    for element in composition:
        feature[int(element) - 1] = 1.0
    return feature


def build_feature_matrix(
    compositions: Iterable[Sequence[int]],
    num_elements: int = 20,
) -> np.ndarray:
    """Encode multiple compositions as a 2D multihot matrix."""

    return np.stack(
        [composition_to_multihot(comp, num_elements=num_elements) for comp in compositions],
        axis=0,
    )


def sample_simple_combination_matches(
    config: SimpleCombinationConfig = SimpleCombinationConfig(),
) -> pd.DataFrame:
    """Generate the Simple Combination Game dataset described in Section 4.1.1."""

    rng = np.random.default_rng(config.random_state)
    all_compositions = enumerate_compositions(
        num_elements=config.num_elements,
        team_size=config.team_size,
    )
    composition_array = np.asarray(all_compositions, dtype=np.int16)

    idx_a = rng.integers(0, len(composition_array), size=config.num_matches)
    idx_b = rng.integers(0, len(composition_array), size=config.num_matches)
    comp_a = composition_array[idx_a]
    comp_b = composition_array[idx_b]

    score_a = comp_a.sum(axis=1)
    score_b = comp_b.sum(axis=1)
    prob_a = (score_a.astype(np.float64) ** 2) / (
        (score_a.astype(np.float64) ** 2) + (score_b.astype(np.float64) ** 2)
    )
    outcome = rng.binomial(1, prob_a).astype(np.float32)

    dataframe = pd.DataFrame(
        {
            "match_id": np.arange(config.num_matches, dtype=np.int64),
            "comp_a": [tuple(row.tolist()) for row in comp_a],
            "comp_b": [tuple(row.tolist()) for row in comp_b],
            "a_1": comp_a[:, 0],
            "a_2": comp_a[:, 1],
            "a_3": comp_a[:, 2],
            "b_1": comp_b[:, 0],
            "b_2": comp_b[:, 1],
            "b_3": comp_b[:, 2],
            "score_a": score_a,
            "score_b": score_b,
            "win_prob_a": prob_a.astype(np.float32),
            "label": outcome,
        }
    )
    return dataframe


def save_simple_combination_dataset(
    output_path: str | Path,
    config: SimpleCombinationConfig = SimpleCombinationConfig(),
) -> pd.DataFrame:
    """Generate and save the synthetic dataset as CSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = sample_simple_combination_matches(config=config)
    dataframe.to_csv(output_path, index=False)
    return dataframe
