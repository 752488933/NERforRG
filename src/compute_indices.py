"""Generate annual vote counts, TVI, PVI, and Wilson intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_indices, annual_vote_counts, load_entities


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entities",
        type=Path,
        default=ROOT / "data/processed/climate_entities.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/annual_indices.csv",
    )
    parser.add_argument("--northern-hemisphere", action="store_true")
    args = parser.parse_args()

    entities = load_entities(args.entities)
    if args.northern_hemisphere:
        entities = entities.loc[entities["lat"] >= 0].copy()

    output = add_indices(annual_vote_counts(entities))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(output):,} annual rows to {args.output}")


if __name__ == "__main__":
    main()

