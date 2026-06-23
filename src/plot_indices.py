"""Plot the annual TVI and PVI with Wilson score intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results/annual_indices.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures/voting_indices.png")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, index_name, lower, upper, label in (
        (axes[0], "tvi", "tvi_lower", "tvi_upper", "Temperature Voting Index"),
        (axes[1], "pvi", "pvi_lower", "pvi_upper", "Precipitation Voting Index"),
    ):
        axis.fill_between(frame["year"], frame[lower], frame[upper], color="0.8", label="95% Wilson CI")
        axis.plot(frame["year"], frame[index_name], color="black", linewidth=0.8)
        axis.set_ylim(0, 1)
        axis.set_ylabel(label)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        axis.legend(frameon=False)
    axes[1].set_xlabel("Year CE")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

