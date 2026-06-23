"""Canonical paper labels and mappings from the legacy annotation names."""

from __future__ import annotations


PAPER_ENTITY_TYPES = ("LOC", "COLD", "WARM", "DRY", "WET")
ID_TO_LABEL = ("O",) + tuple(
    tag
    for entity_type in PAPER_ENTITY_TYPES
    for tag in (f"B-{entity_type}", f"I-{entity_type}")
)
LABEL_TO_ID = {label: index for index, label in enumerate(ID_TO_LABEL)}


SOURCE_TO_PAPER = {
    "研究区域": "LOC",
    "寒冷时期": "COLD",
    "温暖时期": "WARM",
    "干旱期": "DRY",
    "湿润期": "WET",
}
IGNORED_SOURCE_LABELS = {"研究内容"}


def entity_type(tag: str) -> str | None:
    """Return the entity type encoded in a BIO tag."""

    if tag == "O" or "-" not in tag:
        return None
    return tag.split("-", 1)[1]


def valid_bio_transition(previous: str | None, current: str) -> bool:
    """Return whether ``previous -> current`` is legal under BIO tagging."""

    if current == "O" or current.startswith("B-"):
        return True
    if not current.startswith("I-"):
        return False
    if previous is None:
        return False
    expected = current[2:]
    return previous in {f"B-{expected}", f"I-{expected}"}
