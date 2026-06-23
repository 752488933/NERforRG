"""The three PyTorch model variants compared in manuscript Table 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .crf import LinearChainCRF


@dataclass
class NerOutput:
    loss: Tensor | None
    emissions: Tensor
    predictions: Tensor


class BertTaggerBase(nn.Module):
    """Shared encoder and output contract for all model variants."""

    def __init__(self, encoder: nn.Module, hidden_size: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None,
    ) -> Tensor:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        return self.encoder(**kwargs).last_hidden_state


class BertForPaperNer(BertTaggerBase):
    """BERT-base with an independent token-level softmax classifier."""

    def __init__(self, encoder: nn.Module, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__(encoder, hidden_size, num_labels, dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
        label_mask: Tensor | None = None,
    ) -> NerOutput:
        hidden = self.dropout(self.encode(input_ids, attention_mask, token_type_ids))
        emissions = self.classifier(hidden)
        predictions = emissions.argmax(dim=-1)
        loss = None
        if labels is not None:
            valid = label_mask.bool() if label_mask is not None else attention_mask.bool()
            loss = nn.functional.cross_entropy(emissions[valid], labels[valid])
        return NerOutput(loss, emissions, predictions)


class BertCrfForPaperNer(BertTaggerBase):
    """BERT token emissions decoded by a constrained linear-chain CRF."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        labels: Sequence[str],
        dropout: float = 0.1,
    ) -> None:
        super().__init__(encoder, hidden_size, len(labels), dropout)
        self.classifier = nn.Linear(hidden_size, len(labels))
        self.crf = LinearChainCRF(labels, constrain_bio=True)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
        label_mask: Tensor | None = None,
    ) -> NerOutput:
        del label_mask  # CRF trains over all non-padding positions, including CLS/SEP as O.
        hidden = self.dropout(self.encode(input_ids, attention_mask, token_type_ids))
        emissions = self.classifier(hidden)
        mask = attention_mask.bool()
        predictions = self.crf.decode(emissions, mask)
        loss = None if labels is None else -self.crf(emissions, labels, mask, reduction="mean")
        return NerOutput(loss, emissions, predictions)


class BertBiLstmCrfForPaperNer(BertTaggerBase):
    """BERT -> BiLSTM -> linear emissions -> constrained CRF."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        labels: Sequence[str],
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(encoder, hidden_size, len(labels), dropout)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden_size * 2, len(labels))
        self.crf = LinearChainCRF(labels, constrain_bio=True)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
        label_mask: Tensor | None = None,
    ) -> NerOutput:
        del label_mask
        encoded = self.dropout(self.encode(input_ids, attention_mask, token_type_ids))
        lengths = attention_mask.long().sum(dim=1).cpu()
        packed = pack_padded_sequence(encoded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bilstm(packed)
        sequence_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.size(1),
        )
        emissions = self.classifier(self.dropout(sequence_output))
        mask = attention_mask.bool()
        predictions = self.crf.decode(emissions, mask)
        loss = None if labels is None else -self.crf(emissions, labels, mask, reduction="mean")
        return NerOutput(loss, emissions, predictions)


MODEL_NAMES = ("bert_base", "bert_crf", "bert_bilstm_crf")


def build_model(
    model_name: str,
    pretrained_model_name: str,
    labels: Sequence[str],
    dropout: float = 0.1,
    lstm_hidden_size: int = 256,
    lstm_layers: int = 1,
    encoder: nn.Module | None = None,
    encoder_hidden_size: int | None = None,
) -> BertTaggerBase:
    """Construct one model, optionally with an injected encoder for testing."""

    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model_name!r}; choose from {MODEL_NAMES}")
    if encoder is None:
        # Imported lazily so CRF/data tests do not require Transformers.
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(pretrained_model_name)
    hidden_size = encoder_hidden_size or int(getattr(encoder.config, "hidden_size"))
    if model_name == "bert_base":
        return BertForPaperNer(encoder, hidden_size, len(labels), dropout)
    if model_name == "bert_crf":
        return BertCrfForPaperNer(encoder, hidden_size, labels, dropout)
    return BertBiLstmCrfForPaperNer(
        encoder,
        hidden_size,
        labels,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
    )


def checkpoint_model_config(model: BertTaggerBase, model_name: str, pretrained: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_name": model_name,
        "pretrained_model_name": pretrained,
        "dropout": model.dropout.p,
        "num_labels": model.num_labels,
    }
    if isinstance(model, BertBiLstmCrfForPaperNer):
        config.update(
            {
                "lstm_hidden_size": model.lstm_hidden_size,
                "lstm_layers": model.lstm_layers,
            }
        )
    return config
