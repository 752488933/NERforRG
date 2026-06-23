"""Exact-span entity metrics for BIO predictions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence


Entity = tuple[str, int, int]


def bio_entities(tags: Sequence[str]) -> set[Entity]:
    """Convert BIO tags into ``(type, start, exclusive_end)`` spans.

    An invalid I-tag is treated as a new B-tag. This makes evaluation robust to
    unconstrained BERT-base predictions without silently discarding entities.
    """

    entities: set[Entity] = set()
    current_type: str | None = None
    current_start = 0
    for index, tag in enumerate(tuple(tags) + ("O",)):
        prefix, _, tag_type = tag.partition("-")
        continues = prefix == "I" and current_type == tag_type
        if current_type is not None and not continues:
            entities.add((current_type, current_start, index))
            current_type = None
        if prefix == "B" or (prefix == "I" and not continues):
            current_type = tag_type
            current_start = index
    return entities


def _scores(true_positive: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": gold,
        "predicted": predicted,
        "true_positive": true_positive,
    }


@dataclass
class EntityMetricAccumulator:
    """Accumulate exact entity counts over independent token chunks."""

    true_positive: Counter = field(default_factory=Counter)
    predicted: Counter = field(default_factory=Counter)
    gold: Counter = field(default_factory=Counter)

    def update(self, predicted_tags: Sequence[str], gold_tags: Sequence[str]) -> None:
        predicted = bio_entities(predicted_tags)
        gold = bio_entities(gold_tags)
        for entity_type, _, _ in predicted:
            self.predicted[entity_type] += 1
        for entity_type, _, _ in gold:
            self.gold[entity_type] += 1
        for entity_type, _, _ in predicted & gold:
            self.true_positive[entity_type] += 1

    def update_many(
        self,
        predictions: Iterable[Sequence[str]],
        references: Iterable[Sequence[str]],
    ) -> None:
        for predicted, gold in zip(predictions, references):
            self.update(predicted, gold)

    def compute(self, entity_types: Sequence[str]) -> dict[str, dict[str, float | int]]:
        per_type = {
            name: _scores(self.true_positive[name], self.predicted[name], self.gold[name])
            for name in entity_types
        }
        micro = _scores(
            sum(self.true_positive.values()),
            sum(self.predicted.values()),
            sum(self.gold.values()),
        )
        macro = {
            metric: sum(float(per_type[name][metric]) for name in entity_types) / len(entity_types)
            for metric in ("precision", "recall", "f1")
        }
        return {"micro": micro, "macro": macro, "per_type": per_type}
