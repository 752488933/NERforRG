"""Recalculate TVI/PVI after excluding the three dominant regions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from common import (
    add_indices,
    annual_vote_counts,
    centered_smooth,
    classify_regions,
    load_entities,
)


ROOT = Path(__file__).resolve().parents[1]


def finite_correlation(left: pd.Series, right: pd.Series) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    return float(pearsonr(left[valid], right[valid]).statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entities", type=Path, default=ROOT / "data/processed/climate_entities.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--smooth-window", type=int, default=30)
    args = parser.parse_args()

    entities = load_entities(args.entities)
    entities["region"] = classify_regions(entities)
    outside = entities.loc[entities["region"] == "Other"].copy()

    all_indices = add_indices(annual_vote_counts(entities))
    outside_indices = add_indices(annual_vote_counts(outside))

    comparison = all_indices[["year", "tvi", "pvi"]].rename(
        columns={"tvi": "tvi_all", "pvi": "pvi_all"}
    )
    comparison["tvi_outside"] = outside_indices["tvi"]
    comparison["pvi_outside"] = outside_indices["pvi"]
    for name in ("tvi_all", "pvi_all", "tvi_outside", "pvi_outside"):
        comparison[f"{name}_{args.smooth_window}yr"] = centered_smooth(
            comparison[name], args.smooth_window
        )

    category_summary = pd.concat(
        [
            entities["flag"].value_counts().rename("all"),
            outside["flag"].value_counts().rename("outside_core_regions"),
        ],
        axis=1,
    ).fillna(0).astype(int).reindex(["warm", "cold", "wet", "dry"])
    category_summary["remaining_percent"] = (
        100 * category_summary["outside_core_regions"] / category_summary["all"]
    )

    stats = pd.DataFrame(
        [
            {
                "index": "TVI",
                "raw_r": finite_correlation(comparison["tvi_all"], comparison["tvi_outside"]),
                "smoothed_r": finite_correlation(
                    comparison[f"tvi_all_{args.smooth_window}yr"],
                    comparison[f"tvi_outside_{args.smooth_window}yr"],
                ),
            },
            {
                "index": "PVI",
                "raw_r": finite_correlation(comparison["pvi_all"], comparison["pvi_outside"]),
                "smoothed_r": finite_correlation(
                    comparison[f"pvi_all_{args.smooth_window}yr"],
                    comparison[f"pvi_outside_{args.smooth_window}yr"],
                ),
            },
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "sensitivity_indices.csv", index=False)
    category_summary.to_csv(args.output_dir / "sensitivity_entity_summary.csv")
    stats.to_csv(args.output_dir / "sensitivity_correlations.csv", index=False)
    print(category_summary.round(2).to_string())
    print(stats.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

