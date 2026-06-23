"""Train the three manuscript NER variants under one controlled pipeline."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    NerCollator,
    NerDataset,
    encode_records,
    paper_spans,
    read_jsonl,
    stratified_record_split,
    write_split_manifest,
)
from .labels import ID_TO_LABEL, PAPER_ENTITY_TYPES
from .metrics import EntityMetricAccumulator
from .models import MODEL_NAMES, BertTaggerBase, build_model, checkpoint_model_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "ner" / "paper_experiment.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BERT-base, BERT-CRF, and BERT-BiLSTM-CRF on character-span JSONL."
    )
    parser.add_argument("--data", type=Path, required=True, help="Annotated id/text/label JSONL")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "ner_training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count")
    parser.add_argument("--batch-size", type=int, help="Override the configured batch size")
    parser.add_argument("--learning-rate", type=float, help="Override the configured learning rate")
    parser.add_argument("--max-length", type=int, help="Override the configured token length")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader worker count")
    return parser.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    overrides = {
        ("training", "epochs"): args.epochs,
        ("training", "batch_size"): args.batch_size,
        ("training", "learning_rate"): args.learning_rate,
        ("data", "max_length"): args.max_length,
        ("training", "num_workers"): args.num_workers,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
    return config


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def model_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device)
        for key in ("input_ids", "attention_mask", "token_type_ids", "labels", "label_mask")
    }


def evaluate(
    model: BertTaggerBase,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    accumulator = EntityMetricAccumulator()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            tensors = model_batch(batch, device)
            output = model(**tensors)
            total_loss += float(output.loss.item()) if output.loss is not None else 0.0
            batches += 1
            predictions = output.predictions.detach().cpu()
            references = batch["labels"]
            masks = batch["label_mask"]
            for predicted, reference, mask in zip(predictions, references, masks):
                valid = mask.bool()
                predicted_tags = [ID_TO_LABEL[index] for index in predicted[valid].tolist()]
                reference_tags = [ID_TO_LABEL[index] for index in reference[valid].tolist()]
                accumulator.update(predicted_tags, reference_tags)
    result = accumulator.compute(PAPER_ENTITY_TYPES)
    result["loss"] = total_loss / max(batches, 1)
    return result


def optimizer_for(model: nn.Module, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm")
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    groups = [
        {
            "params": [parameter for name, parameter in named_parameters if not any(x in name for x in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [parameter for name, parameter in named_parameters if any(x in name for x in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(groups, lr=learning_rate)


def write_history(path: Path, history: Iterable[dict[str, Any]]) -> None:
    rows = list(history)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: BertTaggerBase,
    model_name: str,
    pretrained_model_name: str,
    epoch: int,
    validation_metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": checkpoint_model_config(model, model_name, pretrained_model_name),
            "labels": list(ID_TO_LABEL),
            "epoch": epoch,
            "validation_metrics": validation_metrics,
            "experiment_config": config,
        },
        path,
    )


def train_one_model(
    model_name: str,
    config: dict[str, Any],
    loaders: dict[str, DataLoader],
    tokenizer: Any,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    from transformers import get_linear_schedule_with_warmup

    training = config["training"]
    architecture = config["model"]
    seed = int(config["experiment"]["seed"])
    set_seed(seed, bool(config["experiment"].get("deterministic", True)))
    if loaders["train"].generator is not None:
        # Give every architecture the same epoch-wise shuffle sequence.
        loaders["train"].generator.manual_seed(seed)
    model = build_model(
        model_name=model_name,
        pretrained_model_name=architecture["pretrained_model_name"],
        labels=ID_TO_LABEL,
        dropout=float(architecture["dropout"]),
        lstm_hidden_size=int(architecture["lstm_hidden_size"]),
        lstm_layers=int(architecture["lstm_layers"]),
    ).to(device)
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }
    optimizer = optimizer_for(
        model,
        float(training["learning_rate"]),
        float(training["weight_decay"]),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(loaders["train"]) / accumulation)
    total_updates = updates_per_epoch * int(training["epochs"])
    warmup_steps = int(total_updates * float(training["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    model_dir = output_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir / "tokenizer")
    (model_dir / "parameters.json").write_text(
        json.dumps(parameter_counts, indent=2), encoding="utf-8"
    )
    checkpoint_path = model_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    patience = int(training.get("early_stopping_patience", 0))

    for epoch in range(1, int(training["epochs"]) + 1):
        started = time.time()
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        optimizer_updates = 0
        for step, batch in enumerate(loaders["train"], start=1):
            output = model(**model_batch(batch, device))
            if output.loss is None:
                raise RuntimeError("Training model did not return a loss")
            (output.loss / accumulation).backward()
            train_loss += float(output.loss.item())
            should_update = step % accumulation == 0 or step == len(loaders["train"])
            if should_update:
                nn.utils.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_updates += 1

        validation = evaluate(model, loaders["validation"], device)
        current_f1 = float(validation["micro"]["f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(len(loaders["train"]), 1),
            "validation_loss": validation["loss"],
            "validation_precision": validation["micro"]["precision"],
            "validation_recall": validation["micro"]["recall"],
            "validation_f1": current_f1,
            "optimizer_updates": optimizer_updates,
            "seconds": round(time.time() - started, 2),
        }
        history.append(row)
        write_history(model_dir / "history.csv", history)
        print(
            f"[{model_name}] epoch {epoch:02d}/{training['epochs']} "
            f"loss={row['train_loss']:.4f} dev_f1={current_f1:.4f} "
            f"P={row['validation_precision']:.4f} R={row['validation_recall']:.4f}"
        )
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                model_name,
                architecture["pretrained_model_name"],
                epoch,
                validation,
                config,
            )
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"[{model_name}] early stopping after {epoch} epochs")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate(model, loaders["test"], device)
    result = {
        "model": model_name,
        "best_epoch": best_epoch,
        "parameters": parameter_counts,
        "validation": checkpoint["validation_metrics"],
        "test": test_metrics,
    }
    (model_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def split_summary(splits: dict[str, list[Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, records in splits.items():
        entity_counts = Counter(
            span.entity_type for record in records for span in paper_spans(record)
        )
        declared_negative = sum(not record.labels for record in records)
        effective_all_o = sum(not paper_spans(record) for record in records)
        result[name] = {
            "records": len(records),
            "declared_positive_records": len(records) - declared_negative,
            "declared_negative_records": declared_negative,
            "effective_all_o_records_after_label_mapping": effective_all_o,
            "entities": dict(entity_counts),
        }
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)
    seed = int(config["experiment"]["seed"])
    set_seed(seed, bool(config["experiment"].get("deterministic", True)))
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["pretrained_model_name"],
        use_fast=True,
    )
    records = read_jsonl(args.data)
    ratios = tuple(float(value) for value in config["data"]["split_ratios"])
    splits = stratified_record_split(records, seed=seed, ratios=ratios)
    write_split_manifest(splits, args.output_dir / "split_manifest.json")

    datasets: dict[str, NerDataset] = {}
    alignment: dict[str, Any] = {}
    max_length = int(config["data"]["max_length"])
    for name, split_records in splits.items():
        chunks, report = encode_records(split_records, tokenizer, max_length)
        datasets[name] = NerDataset(chunks)
        alignment[name] = report.as_dict()
    data_report = {
        "source": str(args.data.resolve()),
        "split": split_summary(splits),
        "alignment": alignment,
        "paper_labels": list(PAPER_ENTITY_TYPES),
        "bio_labels": list(ID_TO_LABEL),
    }
    (args.output_dir / "data_report.json").write_text(
        json.dumps(data_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    collator = NerCollator(tokenizer.pad_token_id)
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(seed),
        )
        for name, dataset in datasets.items()
    }
    resolved_config = {
        **config,
        "run": {
            "data": str(args.data.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "models": args.models,
            "device": str(device),
        },
    }
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2), encoding="utf-8"
    )

    results = [
        train_one_model(name, config, loaders, tokenizer, args.output_dir, device)
        for name in args.models
    ]
    comparison = [
        {
            "model": result["model"],
            "best_epoch": result["best_epoch"],
            "precision": result["test"]["micro"]["precision"],
            "recall": result["test"]["micro"]["recall"],
            "f1": result["test"]["micro"]["f1"],
        }
        for result in results
    ]
    with (args.output_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
