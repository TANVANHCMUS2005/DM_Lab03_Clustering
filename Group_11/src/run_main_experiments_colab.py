"""Run the main Member 2 experiments in Google Colab or local Python.

Usage on Colab:
    !python /content/DM_Lab03_Clustering/Group_11/run_main_experiments_colab.py \
        --project-root /content/DM_Lab03_Clustering

If you clone the repo into Google Drive instead:
    !python /content/drive/MyDrive/DM_Lab03_Clustering/Group_11/run_main_experiments_colab.py \
        --project-root /content/drive/MyDrive/DM_Lab03_Clustering
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import KFold


def detect_project_root(override: str | None = None) -> Path:
    if override is not None:
        root = Path(override).expanduser().resolve()
        if not (root / "Group_11" / "src").exists():
            raise FileNotFoundError(f"Invalid --project-root: {root}")
        return root

    candidates = []
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    candidates.extend(
        [
            Path("/content"),
            Path("/content/DM_Lab03_Clustering"),
            Path("/content/drive/MyDrive"),
            Path("/content/drive/MyDrive/DM_Lab03_Clustering"),
        ]
    )

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "Group_11" / "src").exists():
            return candidate

    raise FileNotFoundError(
        "Cannot detect project root automatically. Pass --project-root explicitly."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the main NRT/NCT experiments from 01_main_experiments.ipynb."
    )
    parser.add_argument("--project-root", default=None, help="Path containing Group_11/")
    parser.add_argument("--output-prefix", default="member2_cv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-matches", type=int, default=100_000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--nrt-epochs", type=int, default=100)
    parser.add_argument("--nct-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--beta-n", type=float, default=0.01)
    parser.add_argument("--beta-m", type=float, default=0.25)
    parser.add_argument("--m-values", type=int, nargs="+", default=[3, 9, 27, 81])
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable CUDA even if Colab GPU is available.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = detect_project_root(args.project_root)
    group_dir = project_root / "Group_11"
    docs_dir = group_dir / "docs"
    data_dir = group_dir / "data"
    docs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if str(group_dir) not in sys.path:
        sys.path.append(str(group_dir))

    from src.metrics import (
        attach_pairwin_from_table,
        codebook_utilization,
        pairwin_table,
        relative_deviation,
        strength_relation_accuracy,
        unsupervised_clustering_scores,
    )
    from src.model import (
        NeuralCounterTable,
        NeuralRatingTable,
        predict_nct,
        predict_nrt,
        set_random_seed,
        train_nct_fullbatch,
        train_nrt_fullbatch,
    )
    from src.synthetic_data import (
        SimpleCombinationConfig,
        build_feature_matrix,
        save_simple_combination_dataset,
    )

    sns.set_theme(style="whitegrid")
    use_cuda = torch.cuda.is_available() and not args.force_cpu
    device = torch.device("cuda" if use_cuda else "cpu")

    data_path = data_dir / "simple_combination_game.csv"
    figure_path = docs_dir / f"{args.output_prefix}_training_and_accuracy.png"
    fold_summary_path = docs_dir / f"{args.output_prefix}_fold_summary.csv"
    aggregate_path = docs_dir / f"{args.output_prefix}_aggregate.csv"
    comparison_path = docs_dir / f"{args.output_prefix}_comparison.csv"
    results_json_path = docs_dir / f"{args.output_prefix}_results.json"

    print("PROJECT_ROOT =", project_root)
    print("GROUP_DIR =", group_dir)
    print("DEVICE =", device)
    print("NUM_MATCHES =", args.num_matches)
    print("NRT_EPOCHS =", args.nrt_epochs)
    print("NCT_EPOCHS =", args.nct_epochs)
    print("M_VALUES =", args.m_values)

    config = SimpleCombinationConfig(
        num_elements=20,
        team_size=3,
        num_matches=args.num_matches,
        random_state=args.seed,
    )
    df = save_simple_combination_dataset(data_path, config=config)
    print(f"Dataset saved to: {data_path}")
    print("Dataset shape:", df.shape)

    comp_a_features = build_feature_matrix(df["comp_a"], num_elements=config.num_elements)
    comp_b_features = build_feature_matrix(df["comp_b"], num_elements=config.num_elements)
    indices = np.arange(len(df))

    set_random_seed(args.seed)
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_runs = []
    run_start = time.time()

    for fold_id, (train_idx, test_idx) in enumerate(kf.split(indices), start=1):
        fold_start = time.time()
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        train_pairwin_lookup = pairwin_table(train_df)
        test_pairwin_lookup = pairwin_table(test_df)
        train_df = attach_pairwin_from_table(train_df, train_pairwin_lookup)
        test_df = attach_pairwin_from_table(test_df, test_pairwin_lookup)

        train_a = comp_a_features[train_idx]
        train_b = comp_b_features[train_idx]
        train_y = train_df["label"].to_numpy(dtype=np.float32)

        aug_a = np.concatenate([train_a, train_b], axis=0)
        aug_b = np.concatenate([train_b, train_a], axis=0)
        aug_y = np.concatenate([train_y, 1.0 - train_y], axis=0)

        test_a = comp_a_features[test_idx]
        test_b = comp_b_features[test_idx]
        test_pairwin = test_df["pairwin_value"].to_numpy(dtype=np.float32)

        set_random_seed(args.seed + fold_id)
        nrt_model = NeuralRatingTable(input_dim=config.num_elements)
        nrt_history = train_nrt_fullbatch(
            nrt_model,
            aug_a,
            aug_b,
            aug_y,
            epochs=args.nrt_epochs,
            learning_rate=args.learning_rate,
            device=device,
        )
        nrt_pred = predict_nrt(nrt_model, test_a, test_b, device=device)
        nrt_acc = strength_relation_accuracy(nrt_pred, test_pairwin)

        fold_record = {
            "fold": fold_id,
            "nrt_accuracy": nrt_acc,
            "nrt_history": nrt_history,
            "nct_results": {},
        }

        for m_value in args.m_values:
            set_random_seed(args.seed + fold_id * 100 + m_value)
            nct_model = NeuralCounterTable(
                input_dim=config.num_elements,
                num_embeddings=m_value,
                embedding_dim=128,
                hidden_dim=128,
            )
            nct_history = train_nct_fullbatch(
                nct_model,
                nrt_model,
                aug_a,
                aug_b,
                aug_y,
                epochs=args.nct_epochs,
                learning_rate=args.learning_rate,
                beta_n=args.beta_n,
                beta_m=args.beta_m,
                device=device,
            )
            nct_outputs = predict_nct(nrt_model, nct_model, test_a, test_b, device=device)
            nct_acc = strength_relation_accuracy(nct_outputs["win_value"], test_pairwin)
            clustering = unsupervised_clustering_scores(
                nct_outputs["latent_a"], nct_outputs["category_a"]
            )
            utilization = codebook_utilization(
                nct_outputs["category_a"], num_embeddings=m_value
            )

            fold_record["nct_results"][m_value] = {
                "accuracy": nct_acc,
                "clustering": clustering,
                "utilization": utilization,
                "history": nct_history,
            }

        fold_runs.append(fold_record)
        print(f"Finished fold {fold_id}/{args.n_splits} in {time.time() - fold_start:.1f}s")

    print(f"Total runtime: {time.time() - run_start:.1f}s")

    records = []
    for fold_run in fold_runs:
        records.append(
            {
                "fold": fold_run["fold"],
                "method": "NRT",
                "accuracy": fold_run["nrt_accuracy"],
                "silhouette_score": np.nan,
                "davies_bouldin_index": np.nan,
                "calinski_harabasz_index": np.nan,
                "used_codes": np.nan,
                "utilization_ratio": np.nan,
            }
        )
        for m_value, result in fold_run["nct_results"].items():
            records.append(
                {
                    "fold": fold_run["fold"],
                    "method": f"NCT (M={m_value})",
                    "accuracy": result["accuracy"],
                    "silhouette_score": result["clustering"]["silhouette_score"],
                    "davies_bouldin_index": result["clustering"]["davies_bouldin_index"],
                    "calinski_harabasz_index": result["clustering"]["calinski_harabasz_index"],
                    "used_codes": result["utilization"]["used_codes"],
                    "utilization_ratio": result["utilization"]["utilization_ratio"],
                }
            )
    fold_summary = pd.DataFrame(records)

    aggregate = (
        fold_summary.groupby("method", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            silhouette_mean=("silhouette_score", "mean"),
            dbi_mean=("davies_bouldin_index", "mean"),
            chi_mean=("calinski_harabasz_index", "mean"),
            used_codes_mean=("used_codes", "mean"),
            utilization_mean=("utilization_ratio", "mean"),
        )
    )

    paper_reference = pd.DataFrame(
        [
            {"method": "NRT", "paper_accuracy": 0.649},
            {"method": "NCT (M=3)", "paper_accuracy": 0.644},
            {"method": "NCT (M=9)", "paper_accuracy": 0.642},
            {"method": "NCT (M=27)", "paper_accuracy": 0.642},
            {"method": "NCT (M=81)", "paper_accuracy": 0.639},
        ]
    )

    comparison = paper_reference.merge(
        aggregate[["method", "accuracy_mean"]],
        on="method",
        how="left",
    )
    comparison["relative_deviation"] = comparison.apply(
        lambda row: relative_deviation(row["accuracy_mean"], row["paper_accuracy"]),
        axis=1,
    )

    fold_summary.to_csv(fold_summary_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    results_json_path.write_text(
        json.dumps(
            {
                "config": {
                    "seed": args.seed,
                    "num_matches": args.num_matches,
                    "n_splits": args.n_splits,
                    "nrt_epochs": args.nrt_epochs,
                    "nct_epochs": args.nct_epochs,
                    "learning_rate": args.learning_rate,
                    "beta_n": args.beta_n,
                    "beta_m": args.beta_m,
                    "m_values": args.m_values,
                },
                "fold_summary": fold_summary.to_dict(orient="records"),
                "aggregate": aggregate.to_dict(orient="records"),
                "comparison": comparison.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_m = max(args.m_values)
    best_fold = max(fold_runs, key=lambda item: item["nct_results"][best_m]["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(best_fold["nrt_history"].total_loss, label="NRT")
    for m_value in args.m_values:
        axes[0].plot(
            best_fold["nct_results"][m_value]["history"].total_loss,
            label=f"NCT M={m_value}",
        )
    axes[0].set_title("Training Loss On One Representative Fold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    sns.barplot(data=fold_summary, x="method", y="accuracy", ax=axes[1], errorbar="sd")
    axes[1].set_title("5-Fold Accuracy Comparison")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:")
    print("-", fold_summary_path)
    print("-", aggregate_path)
    print("-", comparison_path)
    print("-", results_json_path)
    print("-", figure_path)
    print("\nAggregate results:")
    print(aggregate.to_string(index=False))
    print("\nComparison against paper:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
