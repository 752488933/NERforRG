"""A compact first-order linear-chain CRF implemented with PyTorch.

The implementation follows the standard forward algorithm and Viterbi
decoding. Optional BIO constraints are applied both while normalizing the
training likelihood and while decoding, so illegal I-tags cannot be selected.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .labels import valid_bio_transition


class LinearChainCRF(nn.Module):
    """First-order CRF for batch-first emission tensors."""

    def __init__(self, labels: Sequence[str], constrain_bio: bool = True) -> None:
        super().__init__()
        if not labels:
            raise ValueError("At least one label is required")
        self.labels = tuple(labels)
        self.num_tags = len(self.labels)
        self.start_transitions = nn.Parameter(torch.empty(self.num_tags))
        self.end_transitions = nn.Parameter(torch.empty(self.num_tags))
        self.transitions = nn.Parameter(torch.empty(self.num_tags, self.num_tags))

        start_constraint = torch.zeros(self.num_tags)
        transition_constraint = torch.zeros(self.num_tags, self.num_tags)
        if constrain_bio:
            for current_id, current in enumerate(self.labels):
                if not valid_bio_transition(None, current):
                    start_constraint[current_id] = -10000.0
            for previous_id, previous in enumerate(self.labels):
                for current_id, current in enumerate(self.labels):
                    if not valid_bio_transition(previous, current):
                        transition_constraint[previous_id, current_id] = -10000.0
        self.register_buffer("start_constraint", start_constraint)
        self.register_buffer("transition_constraint", transition_constraint)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: Tensor,
        tags: Tensor,
        mask: Tensor,
        reduction: str = "mean",
    ) -> Tensor:
        """Return the log likelihood of the supplied tag sequences."""

        self._validate(emissions, tags, mask)
        numerator = self._score_sentence(emissions, tags, mask)
        denominator = self._log_partition(emissions, mask)
        likelihood = numerator - denominator
        if reduction == "none":
            return likelihood
        if reduction == "sum":
            return likelihood.sum()
        if reduction == "mean":
            return likelihood.mean()
        raise ValueError(f"Unknown reduction: {reduction}")

    def decode(self, emissions: Tensor, mask: Tensor) -> Tensor:
        """Viterbi decode and return a padded ``[batch, time]`` tag tensor."""

        self._validate(emissions, None, mask)
        mask = mask.bool()
        batch_size, sequence_length, _ = emissions.shape
        score = self.start_transitions + self.start_constraint + emissions[:, 0]
        history: list[Tensor] = []
        constrained = self.transitions + self.transition_constraint

        for step in range(1, sequence_length):
            candidates = score.unsqueeze(2) + constrained.unsqueeze(0)
            best_score, best_path = candidates.max(dim=1)
            best_score = best_score + emissions[:, step]
            score = torch.where(mask[:, step].unsqueeze(1), best_score, score)
            history.append(best_path)

        score = score + self.end_transitions
        best_last_tag = score.argmax(dim=1)
        lengths = mask.long().sum(dim=1)
        paths = emissions.new_zeros((batch_size, sequence_length), dtype=torch.long)

        for batch_index in range(batch_size):
            length = int(lengths[batch_index].item())
            tag = best_last_tag[batch_index]
            paths[batch_index, length - 1] = tag
            for step in range(length - 2, -1, -1):
                tag = history[step][batch_index, tag]
                paths[batch_index, step] = tag
        return paths

    def _score_sentence(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        mask = mask.bool()
        batch_indices = torch.arange(emissions.size(0), device=emissions.device)
        first_tags = tags[:, 0]
        score = (
            self.start_transitions[first_tags]
            + self.start_constraint[first_tags]
            + emissions[batch_indices, 0, first_tags]
        )
        constrained = self.transitions + self.transition_constraint
        for step in range(1, emissions.size(1)):
            previous_tags = tags[:, step - 1]
            current_tags = tags[:, step]
            step_score = (
                constrained[previous_tags, current_tags]
                + emissions[batch_indices, step, current_tags]
            )
            score = score + step_score * mask[:, step]
        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def _log_partition(self, emissions: Tensor, mask: Tensor) -> Tensor:
        mask = mask.bool()
        score = self.start_transitions + self.start_constraint + emissions[:, 0]
        constrained = self.transitions + self.transition_constraint
        for step in range(1, emissions.size(1)):
            candidates = (
                score.unsqueeze(2)
                + constrained.unsqueeze(0)
                + emissions[:, step].unsqueeze(1)
            )
            next_score = torch.logsumexp(candidates, dim=1)
            score = torch.where(mask[:, step].unsqueeze(1), next_score, score)
        return torch.logsumexp(score + self.end_transitions, dim=1)

    def _validate(self, emissions: Tensor, tags: Tensor | None, mask: Tensor) -> None:
        if emissions.dim() != 3 or emissions.size(2) != self.num_tags:
            raise ValueError("emissions must have shape [batch, time, num_tags]")
        if mask.shape != emissions.shape[:2]:
            raise ValueError("mask must match the first two emission dimensions")
        if tags is not None and tags.shape != emissions.shape[:2]:
            raise ValueError("tags must match the first two emission dimensions")
        if not mask[:, 0].bool().all():
            raise ValueError("The first timestep must be valid for every sequence")
