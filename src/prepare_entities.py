"""Create the portable entity CSV from the retained final workbook."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import load_entities


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/source/climate_entities.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/climate_entities.csv",
    )
    args = parser.parse_args()

    entities = load_entities(args.input)
    if entities.duplicated().any():
        raise ValueError(f"Exact duplicate rows: {int(entities.duplicated().sum())}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    entities.to_csv(args.output, index=False, encoding="utf-8")
    counts = entities["flag"].value_counts().reindex(["cold", "warm", "dry", "wet"])
    print(f"Wrote {len(entities):,} entities to {args.output}")
    print(counts.to_string())
    print(f"Unique source IDs: {entities['id'].nunique():,}")


if __name__ == "__main__":
    main()

