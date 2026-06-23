"""Shared data loading and voting-index calculations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


CATEGORIES = ("cold", "warm", "dry", "wet")
CORE_REGIONS = ("Europe", "North America", "Asia")


def load_entities(path: str | Path) -> pd.DataFrame:
    """Load and validate the flat climate-entity table."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported entity file: {path}")

    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")]
    required = {"id", "flag", "begin", "end", "lon", "lat"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing entity columns: {sorted(missing)}")

    frame = frame[list(required)].copy()
    frame["flag"] = frame["flag"].astype(str).str.strip().str.lower()
    for column in ("id", "begin", "end", "lon", "lat"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid = (
        frame["flag"].isin(CATEGORIES)
        & frame[["id", "begin", "end", "lon", "lat"]].notna().all(axis=1)
    )
    if not valid.all():
        raise ValueError(f"Invalid entity rows: {int((~valid).sum())}")
    if (frame["begin"] > frame["end"]).any():
        raise ValueError("At least one entity begins after it ends")

    frame[["id", "begin", "end"]] = frame[["id", "begin", "end"]].astype(int)
    return frame[["id", "flag", "begin", "end", "lon", "lat"]]


def classify_regions(frame: pd.DataFrame) -> pd.Series:
    """Apply the rectangular sensitivity masks from the revision notebook."""
    lon = frame["lon"]
    lat = frame["lat"]
    region = np.full(len(frame), "Other", dtype=object)
    region[(lon.between(-25, 45)) & (lat.between(35, 72))] = "Europe"
    region[(lon.between(-170, -50)) & (lat.between(10, 85))] = "North America"
    region[(lon.between(60, 180)) & (lat.between(10, 80))] = "Asia"
    return pd.Series(region, index=frame.index, name="region")


def annual_vote_counts(
    frame: pd.DataFrame, start_year: int = 1, end_year: int = 2000
) -> pd.DataFrame:
    """Count inclusive entity intervals efficiently with difference arrays."""
    years = np.arange(start_year, end_year + 1)
    result = pd.DataFrame({"year": years})
    width = len(years)

    for category in CATEGORIES:
        difference = np.zeros(width + 1, dtype=int)
        subset = frame.loc[frame["flag"] == category, ["begin", "end"]]
        for begin, end in subset.itertuples(index=False, name=None):
            begin = max(start_year, int(begin))
            end = min(end_year, int(end))
            if begin > end:
                continue
            difference[begin - start_year] += 1
            difference[end - start_year + 1] -= 1
        result[category] = np.cumsum(difference[:-1])

    return result


def wilson_interval(successes: np.ndarray, totals: np.ndarray, confidence: float = 0.95):
    """Return lower and upper Wilson score bounds."""
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    z = norm.ppf(0.5 + confidence / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = successes / totals
        denominator = 1 + z**2 / totals
        center = (p + z**2 / (2 * totals)) / denominator
        margin = z * np.sqrt(p * (1 - p) / totals + z**2 / (4 * totals**2)) / denominator
    lower = np.where(totals > 0, center - margin, np.nan)
    upper = np.where(totals > 0, center + margin, np.nan)
    return lower, upper


def add_indices(counts: pd.DataFrame) -> pd.DataFrame:
    """Add TVI, PVI, and their 95% Wilson intervals."""
    output = counts.copy()
    temperature_total = output["warm"] + output["cold"]
    precipitation_total = output["wet"] + output["dry"]
    output["tvi"] = output["warm"].div(temperature_total.where(temperature_total > 0))
    output["pvi"] = output["wet"].div(precipitation_total.where(precipitation_total > 0))
    output["tvi_lower"], output["tvi_upper"] = wilson_interval(output["warm"], temperature_total)
    output["pvi_lower"], output["pvi_upper"] = wilson_interval(output["wet"], precipitation_total)
    return output


def centered_smooth(values: pd.Series, window: int) -> pd.Series:
    """Centered rolling mean with edge values retained for plotting."""
    return values.interpolate(limit_direction="both").rolling(
        window=window, center=True, min_periods=1
    ).mean()

