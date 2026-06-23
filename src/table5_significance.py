"""Reproduce Table 5 correlations with autocorrelation and FDR correction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t
from statsmodels.stats.multitest import multipletests

from common import add_indices, annual_vote_counts, load_entities


ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = {
    "-2.0": "drought_-2.0",
    "-1.5": "drought_-1.5",
    "-1.0": "drought_-1.0",
    "-0.5": "drought_-0.5",
    "0.5": "wet_0.5",
    "1.0": "wet_1.0",
    "1.5": "wet_1.5",
    "2.0": "wet_2.0",
}


def prepare_hydroclimate(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="PanelB")
    frame["drought_-2.0"] = -frame["负2"]
    frame["drought_-1.5"] = -(frame["负2"] + frame["负1.5"])
    frame["drought_-1.0"] = -(frame["负2"] + frame["负1.5"] + frame["负1"])
    frame["drought_-0.5"] = -(
        frame["负2"] + frame["负1.5"] + frame["负1"] + frame["负0.5"]
    )
    frame["wet_2.0"] = frame["2"]
    frame["wet_1.5"] = frame["2"] + frame["1.5"]
    frame["wet_1.0"] = frame["2"] + frame["1.5"] + frame["1"]
    frame["wet_0.5"] = frame["2"] + frame["1.5"] + frame["1"] + frame["0.5"]
    year_column = next((c for c in ("Year", "year", "年份", "年代") if c in frame), None)
    frame["Year"] = (
        pd.to_numeric(frame[year_column], errors="raise")
        if year_column
        else np.arange(850, 850 + 25 * len(frame), 25)
    )
    return frame


def lag1(values: np.ndarray) -> float:
    if len(values) < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return 0.0
    return float(pearsonr(values[:-1], values[1:]).statistic)


def corrected_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    n = len(x)
    r = float(pearsonr(x, y).statistic)
    r1_x, r1_y = lag1(x), lag1(y)
    product = r1_x * r1_y
    neff = float(np.clip(n * (1 - product) / (1 + product), 3, n))
    degrees = neff - 2
    test_r = np.clip(r, -0.999999999, 0.999999999)
    statistic = test_r * np.sqrt(degrees / (1 - test_r**2))
    p_value = float(2 * t.sf(abs(statistic), df=degrees))
    return {"N": n, "r": r, "r1_pvi": r1_x, "r1_reference": r1_y, "N_eff": neff, "p": p_value}


def build_aligned_data(entities_path: Path, reference_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    entities = load_entities(entities_path)
    entities = entities.loc[entities["lat"] >= 0]
    annual = add_indices(annual_vote_counts(entities)).set_index("year")
    pvi = annual["pvi"].rolling(11, center=True, min_periods=11).mean()
    reference = prepare_hydroclimate(reference_path)
    years = reference["Year"].astype(int).to_numpy()
    aligned = pd.DataFrame({"Year": years, "PVI_11yr": pvi.reindex(years).to_numpy()})
    for threshold, column in THRESHOLDS.items():
        aligned[f"Anomaly_{threshold}"] = reference[column].to_numpy()
    return aligned, reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, default=ROOT / "data/processed/climate_entities.csv")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/reference/nh_hydroclimate.xlsx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    aligned, reference = build_aligned_data(args.entities, args.reference)
    rows = []
    for threshold, column in THRESHOLDS.items():
        result = corrected_correlation(
            aligned["PVI_11yr"].to_numpy(), reference[column].to_numpy()
        )
        rows.append({"anomaly_threshold": threshold, **result})
    output = pd.DataFrame(rows)
    rejected, adjusted, _, _ = multipletests(output["p"], alpha=0.05, method="fdr_bh")
    output["p_FDR"] = adjusted
    output["significant_after_FDR"] = rejected

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "table5_significance.csv", index=False)
    aligned.to_csv(args.output_dir / "table5_aligned_series.csv", index=False)
    print(output.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

