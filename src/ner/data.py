"""Read character-span JSONL and build BERT token-classification chunks."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .labels import (
    ID_TO_LABEL,
    IGNORED_SOURCE_LABELS,
    LABEL_TO_ID,
    SOURCE_TO_PAPER,
)


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    source_label: str
    entity_type: str
    source_order: int


@dataclass(frozen=True)
class AnnotationRecord:
    record_id: str
    text: str
    labels: tuple[tuple[int, int, str], ...]


@dataclass
class AlignmentReport:
    source_records: int = 0
    output_chunks: int = 0
    accepted_entities: Counter = field(default_factory=Counter)
    ignored_entities: Counter = field(default_factory=Counter)
    overlap_conflicts: Counter = field(default_factory=Counter)
    empty_token_spans: Counter = field(default_factory=Counter)
    forced_entity_splits: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_records": self.source_records,
            "output_chunks": self.output_chunks,
            "accepted_entities": dict(self.accepted_entities),
            "ignored_entities": dict(self.ignored_entities),
            "overlap_conflicts": dict(self.overlap_conflicts),
            "empty_token_spans": dict(self.empty_token_spans),
            "forced_entity_splits": self.forced_entity_splits,
        }


@dataclass
class EncodedChunk:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    labels: list[int]
    label_mask: list[int]
    offsets: list[tuple[int, int]]
    record_id: str
    chunk_index: int


def read_jsonl(path: Path) -> list[AnnotationRecord]:
    """Read and strictly validate the legacy ``id/text/label`` format."""

    records: list[AnnotationRecord] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            missing = {"id", "text", "label"} - raw.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing keys: {sorted(missing)}")
            record_id = str(raw["id"])
            if record_id in seen_ids:
                raise ValueError(f"Duplicate record id {record_id!r} at line {line_number}")
            seen_ids.add(record_id)
            text = raw["text"]
            if not isinstance(text, str) or not text:
                raise ValueError(f"{path}:{line_number} has invalid text")
            labels: list[tuple[int, int, str]] = []
            for item in raw["label"]:
                if not isinstance(item, list) or len(item) != 3:
                    raise ValueError(f"{path}:{line_number} has malformed span {item!r}")
                start, end, source_label = item
                if not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError(f"{path}:{line_number} has non-integer offsets")
                if not 0 <= start < end <= len(text):
                    raise ValueError(f"{path}:{line_number} has out-of-range span {item!r}")
                if source_label not in set(SOURCE_TO_PAPER) | IGNORED_SOURCE_LABELS:
                    raise ValueError(f"Unknown source label {source_label!r} at line {line_number}")
                labels.append((start, end, source_label))
            records.append(AnnotationRecord(record_id, text, tuple(labels)))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def paper_spans(
    record: AnnotationRecord,
    report: AlignmentReport | None = None,
) -> list[SourceSpan]:
    """Map source labels and resolve overlaps unsupported by a BIO sequence.

    Longest spans take precedence; source order breaks ties. Every discarded
    overlap is reported so it can be reviewed instead of being silently lost.
    """

    candidates: list[SourceSpan] = []
    for source_order, (start, end, source_label) in enumerate(record.labels):
        if source_label in IGNORED_SOURCE_LABELS:
            if report is not None:
                report.ignored_entities[source_label] += 1
            continue
        candidates.append(
            SourceSpan(start, end, source_label, SOURCE_TO_PAPER[source_label], source_order)
        )

    selected: list[SourceSpan] = []
    for candidate in sorted(candidates, key=lambda span: (-(span.end - span.start), span.source_order)):
        conflict = next(
            (
                accepted
                for accepted in selected
                if candidate.start < accepted.end and accepted.start < candidate.end
            ),
            None,
        )
        if conflict is not None:
            if report is not None:
                report.overlap_conflicts[f"{candidate.entity_type}->{conflict.entity_type}"] += 1
            continue
        selected.append(candidate)
        if report is not None:
            report.accepted_entities[candidate.entity_type] += 1
    return sorted(selected, key=lambda span: (span.start, span.end))


def _split_sizes(total: int, ratios: Sequence[float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    sizes = [int(value) for value in raw]
    remainder = total - sum(sizes)
    order = sorted(range(len(ratios)), key=lambda index: raw[index] - sizes[index], reverse=True)
    for index in order[:remainder]:
        sizes[index] += 1
    return sizes


def stratified_record_split(
    records: Sequence[AnnotationRecord],
    seed: int = 42,
    ratios: Sequence[float] = (0.7, 0.2, 0.1),
) -> dict[str, list[AnnotationRecord]]:
    """Create a deterministic 7:2:1 split by declared empty/non-empty labels.

    Empty source label lists identify deliberately sampled negatives. An
    out-of-scope-only positive can become all-O after mapping; the trainer
    reports that separate effective count.
    """

    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Three split ratios summing to one are required")
    buckets: dict[str, list[AnnotationRecord]] = {"positive": [], "negative": []}
    for record in records:
        bucket = "positive" if record.labels else "negative"
        buckets[bucket].append(record)

    result = {"train": [], "validation": [], "test": []}
    names = tuple(result)
    for bucket_index, bucket in enumerate(("positive", "negative")):
        values = list(buckets[bucket])
        random.Random(seed + bucket_index).shuffle(values)
        sizes = _split_sizes(len(values), ratios)
        cursor = 0
        for name, size in zip(names, sizes):
            result[name].extend(values[cursor : cursor + size])
            cursor += size
    for split_index, name in enumerate(names):
        random.Random(seed + 100 + split_index).shuffle(result[name])
    return result


def write_split_manifest(splits: dict[str, Sequence[AnnotationRecord]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: [record.record_id for record in records] for name, records in splits.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _token_labels(
    record: AnnotationRecord,
    offsets: Sequence[tuple[int, int]],
    report: AlignmentReport,
) -> list[int]:
    labels = [LABEL_TO_ID["O"]] * len(offsets)
    for span in paper_spans(record, report):
        token_indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < span.end and span.start < end
        ]
        if not token_indices:
            report.empty_token_spans[span.entity_type] += 1
            continue
        labels[token_indices[0]] = LABEL_TO_ID[f"B-{span.entity_type}"]
        for index in token_indices[1:]:
            labels[index] = LABEL_TO_ID[f"I-{span.entity_type}"]
    return labels


def _chunk_boundaries(labels: Sequence[int], capacity: int, report: AlignmentReport) -> list[tuple[int, int]]:
    boundaries: list[tuple[int, int]] = []
    start = 0
    while start < len(labels):
        tentative_end = min(start + capacity, len(labels))
        end = tentative_end
        while end > start and end < len(labels) and ID_TO_LABEL[labels[end]].startswith("I-"):
            end -= 1
        if end == start:
            end = tentative_end
            report.forced_entity_splits += 1
        boundaries.append((start, end))
        start = end
    return boundaries


def encode_record(
    record: AnnotationRecord,
    tokenizer: Any,
    max_length: int,
    report: AlignmentReport,
) -> list[EncodedChunk]:
    """Tokenize one record and split it without crossing normal entity spans."""

    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast Hugging Face tokenizer is required for offset mapping")
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    capacity = max_length - special_count
    if capacity < 8:
        raise ValueError("max_length leaves fewer than eight content tokens")

    tokenized = tokenizer(
        record.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    input_ids = list(tokenized["input_ids"])
    offsets = [tuple(pair) for pair in tokenized["offset_mapping"]]
    token_labels = _token_labels(record, offsets, report)
    chunks: list[EncodedChunk] = []

    for chunk_index, (start, end) in enumerate(_chunk_boundaries(token_labels, capacity, report)):
        chunk_ids = input_ids[start:end]
        prepared_ids = tokenizer.build_inputs_with_special_tokens(chunk_ids)
        # A literal "[CLS]" or "[SEP]" in an abstract can produce the same ID
        # as a wrapper token. Build a positional template instead of comparing
        # token IDs, so such content is not accidentally removed from metrics.
        marker = -1
        template = tokenizer.build_inputs_with_special_tokens([marker] * len(chunk_ids))
        special_mask = [int(token_id != marker) for token_id in template]
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(chunk_ids)
        if len(prepared_ids) != len(special_mask):
            raise AssertionError("Tokenizer special-token template is inconsistent")
        if len(prepared_ids) > max_length:
            raise AssertionError("Constructed chunk exceeds max_length")
        content_labels = token_labels[start:end]
        content_offsets = offsets[start:end]
        labels: list[int] = []
        label_mask: list[int] = []
        output_offsets: list[tuple[int, int]] = []
        cursor = 0
        for is_special in special_mask:
            if is_special:
                labels.append(LABEL_TO_ID["O"])
                label_mask.append(0)
                output_offsets.append((0, 0))
            else:
                labels.append(content_labels[cursor])
                label_mask.append(1)
                output_offsets.append(content_offsets[cursor])
                cursor += 1
        if cursor != len(content_labels):
            raise AssertionError("Tokenizer special-token mapping is inconsistent")
        chunks.append(
            EncodedChunk(
                input_ids=list(prepared_ids),
                attention_mask=[1] * len(prepared_ids),
                token_type_ids=list(token_type_ids),
                labels=labels,
                label_mask=label_mask,
                offsets=output_offsets,
                record_id=record.record_id,
                chunk_index=chunk_index,
            )
        )
    report.source_records += 1
    report.output_chunks += len(chunks)
    return chunks


def encode_records(
    records: Iterable[AnnotationRecord],
    tokenizer: Any,
    max_length: int,
) -> tuple[list[EncodedChunk], AlignmentReport]:
    report = AlignmentReport()
    chunks: list[EncodedChunk] = []
    for record in records:
        chunks.extend(encode_record(record, tokenizer, max_length, report))
    return chunks, report


class NerDataset(Dataset):
    def __init__(self, chunks: Sequence[EncodedChunk]) -> None:
        self.chunks = list(chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> EncodedChunk:
        return self.chunks[index]


class NerCollator:
    """Pad chunks and retain metadata separately from model tensors."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, chunks: Sequence[EncodedChunk]) -> dict[str, Any]:
        max_length = max(len(chunk.input_ids) for chunk in chunks)

        def pad(values: Sequence[int], fill: int) -> list[int]:
            return list(values) + [fill] * (max_length - len(values))

        return {
            "input_ids": torch.tensor(
                [pad(chunk.input_ids, self.pad_token_id) for chunk in chunks], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [pad(chunk.attention_mask, 0) for chunk in chunks], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                [pad(chunk.token_type_ids, 0) for chunk in chunks], dtype=torch.long
            ),
            "labels": torch.tensor(
                [pad(chunk.labels, LABEL_TO_ID["O"]) for chunk in chunks], dtype=torch.long
            ),
            "label_mask": torch.tensor(
                [pad(chunk.label_mask, 0) for chunk in chunks], dtype=torch.bool
            ),
            "metadata": [
                {
                    "record_id": chunk.record_id,
                    "chunk_index": chunk.chunk_index,
                    "offsets": chunk.offsets,
                }
                for chunk in chunks
            ],
        }
