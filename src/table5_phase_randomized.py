"""Random-phase Monte Carlo robustness test for the Table 5 correlations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

from table5_significance import ROOT, THRESHOLDS, build_aligned_data


def phase_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    transformed = np.fft.rfft(values)
    amplitudes = np.abs(transformed)
    phases = np.angle(transformed)
    stop = len(transformed) - 1 if len(values) % 2 == 0 else len(transformed)
    phases[1:stop] = rng.uniform(0, 2 * np.pi, stop - 1)
    return np.fft.irfft(amplitudes * np.exp(1j * phases), n=len(values))


def test(x: np.ndarray, y: np.ndarray, simulations: int, seed: int) -> dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    observed = float(pearsonr(x, y).statistic)
    rng = np.random.default_rng(seed)
    null = np.empty(simulations)
    for index in range(simulations):
        x_null = phase_surrogate(x, rng)
        y_null = phase_surrogate(y, rng)
        null[index] = np.corrcoef(x_null, y_null)[0, 1]
    p_value = float((np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (simulations + 1))
    return {
        "N": len(x),
        "r": observed,
        "p_MC": p_value,
        "null_abs_r_95": float(np.quantile(np.abs(null), 0.95)),
        "null_abs_r_99": float(np.quantile(np.abs(null), 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, default=ROOT / "data/processed/climate_entities.csv")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/reference/nh_hydroclimate.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "results/table5_phase_randomized.csv")
    parser.add_argument("--simulations", type=int, default=19999)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()

    aligned, reference = build_aligned_data(args.entities, args.reference)
    rows = []
    for index, (threshold, column) in enumerate(THRESHOLDS.items()):
        result = test(
            aligned["PVI_11yr"].to_numpy(),
            reference[column].to_numpy(),
            args.simulations,
            args.seed + index * 1000,
        )
        rows.append({"anomaly_threshold": threshold, **result})
    output = pd.DataFrame(rows)
    rejected, adjusted, _, _ = multipletests(output["p_MC"], alpha=0.05, method="fdr_bh")
    output["p_MC_FDR"] = adjusted
    output["significant_after_FDR"] = rejected
    output["simulations"] = args.simulations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

