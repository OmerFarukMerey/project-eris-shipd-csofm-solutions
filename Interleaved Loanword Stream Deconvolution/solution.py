#!/usr/bin/env python3
"""Train a neural stream model and perform constrained loanword deinterleaving."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


START_TIME = time.time()
MASTER_SEED = 1729
TRAINING_DEADLINE_SECONDS = 3000.0
FULL_TRAINING_CUTOFF_SECONDS = 2850.0
BOUNDARY = "<STREAM_BOUNDARY>"


def warn(message: str) -> None:
    print(f"[warning] {message}", file=sys.stderr, flush=True)
def popcount(value: int) -> int:
    count = 0
    while value:
        value &= value - 1
        count += 1
    return count




def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_packet(raw: dict[str, str], labeled: bool) -> dict:
    packet = json.loads(raw["deinterleaving_packet_json"])
    row = {
        "id": str(raw["id"]),
        "recipient": str(packet["recipient_code"]),
        "mixed": [str(token) for token in packet["mixed_glyph_stream"]],
        "slots": list(packet["lexeme_slots"]),
    }
    if labeled:
        row["answers"] = json.loads(raw["answer_json"])["lexeme_streams"]
    return row


def is_interleaving(mixed: list[str], streams: list[list[str]]) -> bool:
    states = {(0, 0, 0)}
    for token in mixed:
        following: set[tuple[int, int, int]] = set()
        for state in states:
            for slot in range(3):
                if state[slot] < len(streams[slot]) and streams[slot][state[slot]] == token:
                    nxt = list(state)
                    nxt[slot] += 1
                    following.add(tuple(nxt))
        states = following
        if not states:
            return False
    return tuple(len(stream) for stream in streams) in states


def valid_streams(row: dict, streams: list[list[str]]) -> bool:
    if not isinstance(streams, list) or len(streams) != 3 or len(row["slots"]) != 3:
        return False
    for slot, stream in zip(row["slots"], streams):
        if not isinstance(stream, list):
            return False
        if len(stream) != int(slot["target_length"]):
            return False
        if len(set(stream)) != int(slot["unique_glyph_count"]):
            return False
    flattened = [token for stream in streams for token in stream]
    if Counter(flattened) != Counter(row["mixed"]):
        return False
    return is_interleaving(row["mixed"], streams)


def structural_partition(row: dict) -> list[list[str]]:
    """Find a constraint-valid partition for the early placeholder or emergency fallback."""
    mixed = row["mixed"]
    slots = row["slots"]
    if len(slots) != 3:
        raise ValueError("the challenge requires exactly three slots")

    token_to_bit = {token: index for index, token in enumerate(sorted(set(mixed)))}
    bits = [1 << token_to_bit[token] for token in mixed]
    lengths = tuple(int(slot["target_length"]) for slot in slots)
    uniques = tuple(int(slot["unique_glyph_count"]) for slot in slots)
    if sum(lengths) != len(mixed):
        raise ValueError("target lengths do not sum to mixed length")

    suffix_counts: list[dict[int, int]] = [{} for _ in range(len(bits) + 1)]
    running: Counter[int] = Counter()
    for index in range(len(bits) - 1, -1, -1):
        running = running.copy()
        running[bits[index]] += 1
        suffix_counts[index] = dict(running)

    def feasible(counts: tuple[int, int, int], masks: tuple[int, int, int], pos: int) -> bool:
        remaining_counts = suffix_counts[pos]
        for slot in range(3):
            remaining_length = lengths[slot] - counts[slot]
            needed_types = uniques[slot] - popcount(masks[slot])
            if needed_types < 0 or remaining_length < needed_types:
                return False
            existing_occurrences = sum(
                count for bit, count in remaining_counts.items() if masks[slot] & bit
            )
            new_type_occurrences = sorted(
                (count for bit, count in remaining_counts.items() if not (masks[slot] & bit)),
                reverse=True,
            )
            if len(new_type_occurrences) < needed_types:
                return False
            if existing_occurrences + sum(new_type_occurrences[:needed_types]) < remaining_length:
                return False
        return True

    rejected: set[tuple] = set()

    def search(
        pos: int,
        counts: tuple[int, int, int],
        masks: tuple[int, int, int],
    ) -> tuple[int, ...] | None:
        state = (pos, counts, masks)
        if state in rejected:
            return None
        rejected.add(state)
        if pos == len(bits):
            if counts == lengths and tuple(popcount(mask) for mask in masks) == uniques:
                return ()
            return None

        bit = bits[pos]
        choices = []
        for slot in range(3):
            if counts[slot] >= lengths[slot]:
                continue
            next_mask = masks[slot] | bit
            if popcount(next_mask) > uniques[slot]:
                continue
            next_counts_list = list(counts)
            next_counts_list[slot] += 1
            next_masks_list = list(masks)
            next_masks_list[slot] = next_mask
            next_counts = tuple(next_counts_list)
            next_masks = tuple(next_masks_list)
            if not feasible(next_counts, next_masks, pos + 1):
                continue
            remaining_capacity = lengths[slot] - counts[slot]
            needed_types = uniques[slot] - popcount(masks[slot])
            choices.append(
                (
                    0 if masks[slot] & bit else 1,
                    -needed_types / max(remaining_capacity, 1),
                    remaining_capacity,
                    slot,
                    next_counts,
                    next_masks,
                )
            )

        for _, _, _, slot, next_counts, next_masks in sorted(choices):
            tail = search(pos + 1, next_counts, next_masks)
            if tail is not None:
                return (slot,) + tail
        return None

    assignment = search(0, (0, 0, 0), (0, 0, 0))
    if assignment is None:
        raise ValueError("no structurally valid deinterleaving exists")
    result = [[], [], []]
    for token, slot in zip(mixed, assignment):
        result[slot].append(token)
    if not valid_streams(row, result):
        raise ValueError("internal structural partition verification failed")
    return result


def answer_text(streams: list[list[str]]) -> str:
    return json.dumps({"lexeme_streams": streams}, separators=(",", ":"), ensure_ascii=False)


def write_submission(path: Path, rows: list[dict], predictions: list[list[list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer_json"], lineterminator="\n")
        writer.writeheader()
        for row, streams in zip(rows, predictions):
            writer.writerow({"id": row["id"], "answer_json": answer_text(streams)})


def submission_is_complete(rows: list[dict], predictions: list[list[list[str]]]) -> bool:
    if len(predictions) != len(rows):
        return False
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not identifier for identifier in ids):
        return False
    return all(valid_streams(row, streams) for row, streams in zip(rows, predictions))


def levenshtein(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for row_index, left_token in enumerate(left, 1):
        current = [row_index]
        for column_index, right_token in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def flattened_streams(streams: list[list[str]]) -> list[str]:
    flattened: list[str] = []
    for index, stream in enumerate(streams):
        if index:
            flattened.append(BOUNDARY)
        flattened.extend(stream)
    return flattened


def row_similarity(prediction: list[list[str]], gold: list[list[str]]) -> float:
    predicted_tokens = flattened_streams(prediction)
    gold_tokens = flattened_streams(gold)
    denominator = max(len(predicted_tokens), len(gold_tokens), 1)
    return 1.0 - levenshtein(predicted_tokens, gold_tokens) / denominator


def grouped_holdout(rows: list[dict]) -> tuple[list[int], list[int]]:
    """Recipient-stratified holdout with exact recipient/concept pairs kept together."""
    parent = list(range(len(rows)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parent[second_root] = first_root

    owner: dict[tuple[str, str], int] = {}
    for row_index, row in enumerate(rows):
        for slot in row["slots"]:
            key = (row["recipient"], str(slot["concept_code"]))
            if key in owner:
                union(row_index, owner[key])
            else:
                owner[key] = row_index

    components: dict[int, list[int]] = defaultdict(list)
    for row_index in range(len(rows)):
        components[root(row_index)].append(row_index)
    by_recipient: dict[str, list[list[int]]] = defaultdict(list)
    for component in components.values():
        by_recipient[rows[component[0]]["recipient"]].append(component)

    rng = random.Random(MASTER_SEED)
    validation: set[int] = set()
    for recipient in sorted(by_recipient):
        groups = by_recipient[recipient]
        rng.shuffle(groups)
        total = sum(len(group) for group in groups)
        target = max(1, round(total * 0.20))
        chosen: list[int] = []
        for group in groups:
            if len(chosen) >= target:
                break
            if len(group) < total:
                chosen.extend(group)
        validation.update(chosen)

    training = [index for index in range(len(rows)) if index not in validation]
    return training, sorted(validation)


def run_learning(train_rows: list[dict], test_rows: list[dict]) -> tuple[list[list[list[str]]], dict]:
    import gc

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cpu")

    def fit_vocabs(rows: list[dict], indices: list[int]) -> dict:
        glyphs = sorted({token for index in indices for token in rows[index]["mixed"]})
        recipients = sorted({rows[index]["recipient"] for index in indices})
        fields = sorted(
            {str(slot["semantic_field"]) for index in indices for slot in rows[index]["slots"]}
        )
        concepts = sorted(
            {str(slot["concept_code"]) for index in indices for slot in rows[index]["slots"]}
        )
        maximum_size = max(
            [
                max(int(slot["target_length"]), int(slot["unique_glyph_count"]))
                for index in indices
                for slot in rows[index]["slots"]
            ]
            or [1]
        )
        return {
            "glyph": {value: idx for idx, value in enumerate(glyphs)},
            "recipient": {value: idx + 1 for idx, value in enumerate(recipients)},
            "field": {value: idx + 1 for idx, value in enumerate(fields)},
            "concept": {value: idx + 1 for idx, value in enumerate(concepts)},
            "size_count": maximum_size + 2,
        }

    def make_records(rows: list[dict], indices: list[int], vocab: dict) -> list[tuple]:
        records = []
        unknown_glyph = len(vocab["glyph"])
        for index in indices:
            row = rows[index]
            recipient = vocab["recipient"].get(row["recipient"], 0)
            for slot_index, slot in enumerate(row["slots"]):
                sequence = [vocab["glyph"].get(token, unknown_glyph) for token in row["answers"][slot_index]]
                records.append(
                    (
                        sequence,
                        recipient,
                        vocab["field"].get(str(slot["semantic_field"]), 0),
                        vocab["concept"].get(str(slot["concept_code"]), 0),
                        int(slot["target_length"]),
                        int(slot["unique_glyph_count"]),
                    )
                )
        return records

    class StreamLanguageModel(nn.Module):
        def __init__(self, vocab: dict, config: dict):
            super().__init__()
            self.glyph_count = len(vocab["glyph"])
            self.output_count = self.glyph_count + 2
            self.recipient_count = len(vocab["recipient"])
            self.size_count = int(vocab["size_count"])
            hidden = int(config["hidden"])
            condition_width = int(config["condition_width"])

            self.glyph_embedding = nn.Embedding(self.glyph_count + 2, 32)
            self.local_glyph_embedding = nn.Embedding(
                (self.recipient_count + 1) * (self.glyph_count + 2), 24
            )
            self.recipient_embedding = nn.Embedding(self.recipient_count + 1, 24)
            self.field_embedding = nn.Embedding(len(vocab["field"]) + 1, 16)
            self.concept_embedding = nn.Embedding(len(vocab["concept"]) + 1, 32)
            self.length_embedding = nn.Embedding(self.size_count, 10)
            self.unique_embedding = nn.Embedding(self.size_count, 10)
            self.direction_embedding = nn.Embedding(2, 6)
            self.condition_projection = nn.Sequential(
                nn.Linear(98, condition_width),
                nn.GELU(),
                nn.LayerNorm(condition_width),
            )
            self.initial_hidden = nn.Linear(condition_width, hidden)
            self.gru = nn.GRU(32 + 24 + condition_width, hidden, batch_first=True)
            self.shared_output = nn.Linear(hidden, self.output_count)
            self.local_output_weight = nn.Parameter(
                torch.zeros(self.recipient_count + 1, self.output_count, hidden)
            )
            self.local_output_bias = nn.Parameter(
                torch.zeros(self.recipient_count + 1, self.output_count)
            )

        def condition(
            self,
            recipient,
            field,
            concept,
            length,
            unique,
            direction,
            concept_dropout: float = 0.0,
        ):
            if self.training and concept_dropout > 0.0:
                drop = torch.rand(concept.shape, device=concept.device) < concept_dropout
                concept = concept.masked_fill(drop, 0)
            length = length.clamp_max(self.size_count - 1)
            unique = unique.clamp_max(self.size_count - 1)
            concatenated = torch.cat(
                [
                    self.recipient_embedding(recipient),
                    self.field_embedding(field),
                    self.concept_embedding(concept),
                    self.length_embedding(length),
                    self.unique_embedding(unique),
                    self.direction_embedding(direction),
                ],
                dim=-1,
            )
            return self.condition_projection(concatenated)

        def next_logits(self, hidden, recipient):
            local_weight = self.local_output_weight[recipient]
            local_bias = self.local_output_bias[recipient]
            return (
                self.shared_output(hidden)
                + torch.einsum("bth,boh->bto", hidden, local_weight)
                + local_bias[:, None, :]
            )

        def beam_logits(self, hidden, recipient_index: int):
            local_weight = self.local_output_weight[recipient_index]
            local_bias = self.local_output_bias[recipient_index]
            return (
                self.shared_output(hidden)
                + torch.einsum("...h,oh->...o", hidden, local_weight)
                + local_bias
            )

        def forward(
            self,
            inputs,
            recipient,
            field,
            concept,
            length,
            unique,
            direction,
            concept_dropout: float = 0.0,
        ):
            condition = self.condition(
                recipient,
                field,
                concept,
                length,
                unique,
                direction,
                concept_dropout,
            )
            local_ids = recipient[:, None] * (self.glyph_count + 2) + inputs
            embedded = torch.cat(
                [
                    self.glyph_embedding(inputs),
                    self.local_glyph_embedding(local_ids),
                    condition[:, None, :].expand(-1, inputs.shape[1], -1),
                ],
                dim=-1,
            )
            initial = torch.tanh(self.initial_hidden(condition)).unsqueeze(0)
            hidden, _ = self.gru(embedded, initial)
            return self.next_logits(hidden, recipient)

    def make_batch(records: list[tuple], pairs: list[tuple[int, int]], vocab: dict):
        glyph_count = len(vocab["glyph"])
        sequences = []
        recipients, fields, concepts, lengths, uniques, directions = [], [], [], [], [], []
        for record_index, direction in pairs:
            sequence, recipient, field, concept, length, unique = records[record_index]
            sequences.append(sequence if direction == 0 else sequence[::-1])
            recipients.append(recipient)
            fields.append(field)
            concepts.append(concept)
            lengths.append(length)
            uniques.append(unique)
            directions.append(direction)
        maximum = max(len(sequence) for sequence in sequences) + 1
        inputs = torch.full((len(sequences), maximum), glyph_count, dtype=torch.long, device=device)
        targets = torch.full((len(sequences), maximum), -100, dtype=torch.long, device=device)
        boundary_id = glyph_count + 1
        for batch_index, sequence in enumerate(sequences):
            inputs[batch_index, 0] = boundary_id
            if sequence:
                inputs[batch_index, 1 : len(sequence) + 1] = torch.tensor(sequence, device=device)
                targets[batch_index, : len(sequence)] = torch.tensor(sequence, device=device)
            targets[batch_index, len(sequence)] = boundary_id
        tensors = [
            torch.tensor(values, dtype=torch.long, device=device)
            for values in (recipients, fields, concepts, lengths, uniques, directions)
        ]
        return inputs, targets, tensors

    def validation_nll(model, records: list[tuple], vocab: dict) -> float:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        pairs = [(record_index, direction) for record_index in range(len(records)) for direction in (0, 1)]
        with torch.no_grad():
            for offset in range(0, len(pairs), 128):
                inputs, targets, tensors = make_batch(records, pairs[offset : offset + 128], vocab)
                logits = model(inputs, *tensors)
                total_loss += F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ).item()
                total_tokens += int((targets != -100).sum().item())
        return total_loss / max(total_tokens, 1)

    def train_epoch(model, records: list[tuple], vocab: dict, config: dict, optimizer, epoch: int) -> None:
        model.train()
        pairs = [(record_index, direction) for record_index in range(len(records)) for direction in (0, 1)]
        random.Random(int(config["seed"]) + epoch * 1009).shuffle(pairs)
        for offset in range(0, len(pairs), 96):
            inputs, targets, tensors = make_batch(records, pairs[offset : offset + 96], vocab)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, *tensors, concept_dropout=float(config["concept_dropout"]))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
                label_smoothing=float(config["label_smoothing"]),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def new_model(vocab: dict, config: dict):
        torch.manual_seed(int(config["seed"]))
        model = StreamLanguageModel(vocab, config).to(device)
        return model

    def new_optimizer(model, config: dict):
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )

    def train_with_holdout(
        records: list[tuple], validation_records: list[tuple], vocab: dict, config: dict
    ):
        model = new_model(vocab, config)
        optimizer = new_optimizer(model, config)
        best_nll = float("inf")
        best_state = None
        best_epoch = 1
        stale_checks = 0
        for epoch in range(1, 81):
            if time.time() - START_TIME >= TRAINING_DEADLINE_SECONDS and epoch > 1:
                break
            train_epoch(model, records, vocab, config, optimizer, epoch)
            if epoch % 4 != 0:
                continue
            nll = validation_nll(model, validation_records, vocab)
            if nll < best_nll - 1e-4:
                best_nll = nll
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_checks = 0
            else:
                stale_checks += 1
            if stale_checks >= 8:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            best_nll = validation_nll(model, validation_records, vocab)
        return model, best_epoch, best_nll

    def train_fixed(records: list[tuple], vocab: dict, config: dict, epochs: int):
        model = new_model(vocab, config)
        optimizer = new_optimizer(model, config)
        completed = 0
        for epoch in range(1, max(1, epochs) + 1):
            if time.time() - START_TIME >= TRAINING_DEADLINE_SECONDS and completed:
                break
            train_epoch(model, records, vocab, config, optimizer, epoch)
            completed = epoch
        model.eval()
        return model, completed

    def descriptor_tensors(row: dict, vocab: dict, direction: int):
        recipient = vocab["recipient"].get(row["recipient"], 0)
        recipients = torch.tensor([recipient] * 3, dtype=torch.long, device=device)
        fields = torch.tensor(
            [vocab["field"].get(str(slot["semantic_field"]), 0) for slot in row["slots"]],
            dtype=torch.long,
            device=device,
        )
        concepts = torch.tensor(
            [vocab["concept"].get(str(slot["concept_code"]), 0) for slot in row["slots"]],
            dtype=torch.long,
            device=device,
        )
        lengths = torch.tensor(
            [int(slot["target_length"]) for slot in row["slots"]], dtype=torch.long, device=device
        )
        uniques = torch.tensor(
            [int(slot["unique_glyph_count"]) for slot in row["slots"]],
            dtype=torch.long,
            device=device,
        )
        directions = torch.full((3,), direction, dtype=torch.long, device=device)
        return recipients, fields, concepts, lengths, uniques, directions

    def neural_candidates(
        row: dict,
        model,
        vocab: dict,
        direction: int,
        beam_width: int,
        keep: int,
    ) -> list[tuple[float, tuple[tuple[str, ...], ...]]]:
        model.eval()
        glyph_count = len(vocab["glyph"])
        sequence = row["mixed"] if direction == 0 else row["mixed"][::-1]
        token_to_bit = {token: index for index, token in enumerate(sorted(set(sequence)))}
        bits = [1 << token_to_bit[token] for token in sequence]
        glyph_ids = [vocab["glyph"].get(token, glyph_count) for token in sequence]
        lengths = tuple(int(slot["target_length"]) for slot in row["slots"])
        uniques = tuple(int(slot["unique_glyph_count"]) for slot in row["slots"])

        suffix_counts: list[dict[int, int]] = [{} for _ in range(len(bits) + 1)]
        running: Counter[int] = Counter()
        for index in range(len(bits) - 1, -1, -1):
            running = running.copy()
            running[bits[index]] += 1
            suffix_counts[index] = dict(running)

        def feasible(counts: tuple[int, int, int], masks: tuple[int, int, int], pos: int) -> bool:
            available = suffix_counts[pos]
            for slot in range(3):
                remaining_length = lengths[slot] - counts[slot]
                needed_types = uniques[slot] - popcount(masks[slot])
                if needed_types < 0 or remaining_length < needed_types:
                    return False
                existing = sum(count for bit, count in available.items() if masks[slot] & bit)
                outside = sorted(
                    (count for bit, count in available.items() if not (masks[slot] & bit)),
                    reverse=True,
                )
                if len(outside) < needed_types or existing + sum(outside[:needed_types]) < remaining_length:
                    return False
            return True

        with torch.no_grad():
            descriptors = descriptor_tensors(row, vocab, direction)
            condition = model.condition(*descriptors)
            hidden = torch.tanh(model.initial_hidden(condition))
            start_ids = torch.full((3,), glyph_count + 1, dtype=torch.long, device=device)
            local_ids = descriptors[0] * (glyph_count + 2) + start_ids
            embedded = torch.cat(
                [
                    model.glyph_embedding(start_ids),
                    model.local_glyph_embedding(local_ids),
                    condition,
                ],
                dim=-1,
            )
            _, hidden_state = model.gru(embedded[:, None, :], hidden[None, :, :])
            hidden_beam = hidden_state[0][None, :, :]
            metadata = [(0.0, (0, 0, 0), (0, 0, 0), ((), (), ()))]

            for position, (token, glyph_id, bit) in enumerate(zip(sequence, glyph_ids, bits)):
                log_probabilities = F.log_softmax(
                    model.beam_logits(hidden_beam, int(descriptors[0][0].item())), dim=-1
                )[:, :, glyph_id]
                options = []
                for beam_index, (score, counts, masks, outputs) in enumerate(metadata):
                    for slot in range(3):
                        if counts[slot] >= lengths[slot]:
                            continue
                        next_mask = masks[slot] | bit
                        if popcount(next_mask) > uniques[slot]:
                            continue
                        next_counts_list = list(counts)
                        next_counts_list[slot] += 1
                        next_masks_list = list(masks)
                        next_masks_list[slot] = next_mask
                        next_counts = tuple(next_counts_list)
                        next_masks = tuple(next_masks_list)
                        if not feasible(next_counts, next_masks, position + 1):
                            continue
                        next_outputs = list(outputs)
                        next_outputs[slot] = next_outputs[slot] + (token,)
                        options.append(
                            (
                                score + float(log_probabilities[beam_index, slot].item()),
                                beam_index,
                                slot,
                                next_counts,
                                next_masks,
                                tuple(next_outputs),
                            )
                        )
                if not options:
                    return []
                options.sort(key=lambda item: item[0], reverse=True)
                options = options[:beam_width]

                parents = torch.tensor([item[1] for item in options], dtype=torch.long, device=device)
                chosen_slots = torch.tensor([item[2] for item in options], dtype=torch.long, device=device)
                hidden_beam = hidden_beam[parents].clone()
                batch_indices = torch.arange(len(options), device=device)
                previous_hidden = hidden_beam[batch_indices, chosen_slots]
                input_ids = torch.full(
                    (len(options),), glyph_id, dtype=torch.long, device=device
                )
                selected_recipients = descriptors[0][chosen_slots]
                selected_condition = condition[chosen_slots]
                selected_local_ids = selected_recipients * (glyph_count + 2) + input_ids
                selected_embedded = torch.cat(
                    [
                        model.glyph_embedding(input_ids),
                        model.local_glyph_embedding(selected_local_ids),
                        selected_condition,
                    ],
                    dim=-1,
                )
                _, updated_hidden = model.gru(
                    selected_embedded[:, None, :], previous_hidden[None, :, :]
                )
                hidden_beam[batch_indices, chosen_slots] = updated_hidden[0]
                metadata = [(item[0], item[3], item[4], item[5]) for item in options]

            eos_scores = F.log_softmax(
                model.beam_logits(hidden_beam, int(descriptors[0][0].item())), dim=-1
            )[:, :, glyph_count + 1].sum(dim=1)
            completed = []
            for beam_index, (score, counts, masks, outputs) in enumerate(metadata):
                if counts != lengths or tuple(popcount(mask) for mask in masks) != uniques:
                    continue
                normalized_outputs = (
                    outputs
                    if direction == 0
                    else tuple(tuple(reversed(stream)) for stream in outputs)
                )
                completed.append((score + float(eos_scores[beam_index].item()), normalized_outputs))
            completed.sort(key=lambda item: item[0], reverse=True)
            return completed[:keep]

    def score_candidates(row: dict, candidates: list[tuple], model, vocab: dict, direction: int) -> list[float]:
        glyph_count = len(vocab["glyph"])
        recipient = vocab["recipient"].get(row["recipient"], 0)
        sequences = []
        recipients, fields, concepts, lengths, uniques, directions = [], [], [], [], [], []
        for candidate in candidates:
            for slot_index, slot in enumerate(row["slots"]):
                encoded = [vocab["glyph"].get(token, glyph_count) for token in candidate[slot_index]]
                sequences.append(encoded if direction == 0 else encoded[::-1])
                recipients.append(recipient)
                fields.append(vocab["field"].get(str(slot["semantic_field"]), 0))
                concepts.append(vocab["concept"].get(str(slot["concept_code"]), 0))
                lengths.append(int(slot["target_length"]))
                uniques.append(int(slot["unique_glyph_count"]))
                directions.append(direction)

        form_scores: list[float] = []
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(sequences), 256):
                batch_sequences = sequences[offset : offset + 256]
                maximum = max(len(sequence) for sequence in batch_sequences) + 1
                inputs = torch.full(
                    (len(batch_sequences), maximum), glyph_count, dtype=torch.long, device=device
                )
                targets = torch.full(
                    (len(batch_sequences), maximum), -100, dtype=torch.long, device=device
                )
                for batch_index, sequence in enumerate(batch_sequences):
                    inputs[batch_index, 0] = glyph_count + 1
                    if sequence:
                        inputs[batch_index, 1 : len(sequence) + 1] = torch.tensor(
                            sequence, device=device
                        )
                        targets[batch_index, : len(sequence)] = torch.tensor(
                            sequence, device=device
                        )
                    targets[batch_index, len(sequence)] = glyph_count + 1
                count = len(batch_sequences)
                descriptor_values = [
                    recipients[offset : offset + count],
                    fields[offset : offset + count],
                    concepts[offset : offset + count],
                    lengths[offset : offset + count],
                    uniques[offset : offset + count],
                    directions[offset : offset + count],
                ]
                tensors = [
                    torch.tensor(values, dtype=torch.long, device=device)
                    for values in descriptor_values
                ]
                log_probabilities = F.log_softmax(model(inputs, *tensors), dim=-1)
                safe_targets = targets.clamp_min(0)
                token_scores = log_probabilities.gather(
                    -1, safe_targets[:, :, None]
                ).squeeze(-1)
                token_scores = token_scores.masked_fill(targets < 0, 0.0).sum(dim=1)
                form_scores.extend(float(value) for value in token_scores.tolist())
        return [sum(form_scores[index : index + 3]) for index in range(0, len(form_scores), 3)]

    def candidate_bundle(row: dict, model, vocab: dict, beam_width: int, keep: int) -> dict:
        forward = neural_candidates(row, model, vocab, 0, beam_width, keep)
        backward = neural_candidates(row, model, vocab, 1, beam_width, keep)
        if not forward and not backward and beam_width < 1024:
            forward = neural_candidates(row, model, vocab, 0, 1024, keep)
            backward = neural_candidates(row, model, vocab, 1, 1024, keep)
        pool: dict[tuple, tuple] = {}
        for _, output in forward + backward:
            pool[output] = output
        if not pool:
            fallback = tuple(tuple(stream) for stream in structural_partition(row))
            return {
                "candidates": [fallback],
                "forward_scores": [0.0],
                "backward_scores": [0.0],
                "forward_rank": {fallback: 0},
                "backward_rank": {fallback: 0},
            }
        candidates = list(pool)
        return {
            "candidates": candidates,
            "forward_scores": score_candidates(row, candidates, model, vocab, 0),
            "backward_scores": score_candidates(row, candidates, model, vocab, 1),
            "forward_rank": {output: rank for rank, (_, output) in enumerate(forward)},
            "backward_rank": {output: rank for rank, (_, output) in enumerate(backward)},
        }

    def eligible_indices(bundle: dict, keep: int) -> list[int]:
        missing_rank = 10**9
        return [
            index
            for index, candidate in enumerate(bundle["candidates"])
            if bundle["forward_rank"].get(candidate, missing_rank) < keep
            or bundle["backward_rank"].get(candidate, missing_rank) < keep
        ]

    def choose_from_bundle(bundle: dict, keep: int, alpha: float) -> tuple:
        eligible = eligible_indices(bundle, keep)
        if not eligible:
            eligible = list(range(len(bundle["candidates"])))
        best_index = max(
            eligible,
            key=lambda index: alpha * bundle["forward_scores"][index]
            + (1.0 - alpha) * bundle["backward_scores"][index],
        )
        return bundle["candidates"][best_index]

    alpha_grid = [index / 10.0 for index in range(11)]

    def best_alpha(rows: list[dict], bundles: list[dict], keep: int) -> tuple[float, float]:
        best_score = -1.0
        selected_alpha = 0.0
        for alpha in alpha_grid:
            scores = []
            for row, bundle in zip(rows, bundles):
                prediction = [list(stream) for stream in choose_from_bundle(bundle, keep, alpha)]
                scores.append(row_similarity(prediction, row["answers"]))
            score = sum(scores) / max(len(scores), 1)
            if score > best_score + 1e-12:
                best_score = score
                selected_alpha = alpha
        return selected_alpha, best_score

    training_indices, validation_indices = grouped_holdout(train_rows)
    inner_vocab = fit_vocabs(train_rows, training_indices)
    inner_records = make_records(train_rows, training_indices, inner_vocab)
    validation_records = make_records(train_rows, validation_indices, inner_vocab)
    validation_rows = [train_rows[index] for index in validation_indices]

    search_configs = [
        {
            "seed": MASTER_SEED + 101,
            "hidden": 96,
            "condition_width": 56,
            "learning_rate": 0.0018,
            "weight_decay": 0.003,
            "concept_dropout": 0.50,
            "label_smoothing": 0.020,
        },
        {
            "seed": MASTER_SEED + 211,
            "hidden": 128,
            "condition_width": 64,
            "learning_rate": 0.0015,
            "weight_decay": 0.004,
            "concept_dropout": 0.60,
            "label_smoothing": 0.025,
        },
        {
            "seed": MASTER_SEED + 307,
            "hidden": 160,
            "condition_width": 72,
            "learning_rate": 0.0012,
            "weight_decay": 0.005,
            "concept_dropout": 0.70,
            "label_smoothing": 0.030,
        },
        {
            "seed": MASTER_SEED + 401,
            "hidden": 96,
            "condition_width": 56,
            "learning_rate": 0.0018,
            "weight_decay": 0.003,
            "concept_dropout": 0.50,
            "label_smoothing": 0.020,
        },
        {
            "seed": MASTER_SEED + 503,
            "hidden": 128,
            "condition_width": 64,
            "learning_rate": 0.0015,
            "weight_decay": 0.004,
            "concept_dropout": 0.60,
            "label_smoothing": 0.025,
        },
        {
            "seed": MASTER_SEED + 607,
            "hidden": 160,
            "condition_width": 72,
            "learning_rate": 0.0012,
            "weight_decay": 0.005,
            "concept_dropout": 0.70,
            "label_smoothing": 0.030,
        },
    ]

    selected_model = None
    selected_config = None
    selected_epoch = 1
    selected_model_score = -1.0
    for config_index, config in enumerate(search_configs):
        if time.time() - START_TIME >= FULL_TRAINING_CUTOFF_SECONDS and selected_model is not None:
            break
        model, best_epoch, nll = train_with_holdout(
            inner_records, validation_records, inner_vocab, config
        )
        bundles = []
        for row in validation_rows:
            try:
                bundles.append(candidate_bundle(row, model, inner_vocab, 128, 16))
            except Exception as exc:
                warn(f"validation candidate fallback for {row['id']}: {exc}")
                fallback = tuple(tuple(stream) for stream in structural_partition(row))
                bundles.append(
                    {
                        "candidates": [fallback],
                        "forward_scores": [0.0],
                        "backward_scores": [0.0],
                        "forward_rank": {fallback: 0},
                        "backward_rank": {fallback: 0},
                    }
                )
        _, metric_score = best_alpha(validation_rows, bundles, 16)
        print(
            f"HPO config={config_index} epoch={best_epoch} val_nll={nll:.6f} "
            f"val_metric={metric_score:.6f}",
            flush=True,
        )
        if metric_score > selected_model_score + 1e-12:
            selected_model = model
            selected_config = dict(config)
            selected_epoch = best_epoch
            selected_model_score = metric_score
        else:
            del model
        del bundles
        gc.collect()

    if selected_model is None or selected_config is None:
        raise RuntimeError("model search did not complete a trainable configuration")

    beam_grid = [80, 160, 320]
    keep_grid = [8, 16, 24]
    selected_beam = 160
    selected_keep = 16
    selected_alpha = 0.5
    selected_validation_score = -1.0
    for beam_width in beam_grid:
        bundles = []
        for row in validation_rows:
            try:
                bundles.append(candidate_bundle(row, selected_model, inner_vocab, beam_width, max(keep_grid)))
            except Exception as exc:
                warn(f"decode HPO fallback for {row['id']}: {exc}")
                fallback = tuple(tuple(stream) for stream in structural_partition(row))
                bundles.append(
                    {
                        "candidates": [fallback],
                        "forward_scores": [0.0],
                        "backward_scores": [0.0],
                        "forward_rank": {fallback: 0},
                        "backward_rank": {fallback: 0},
                    }
                )
        for keep in keep_grid:
            alpha, score = best_alpha(validation_rows, bundles, keep)
            if score > selected_validation_score + 1e-12:
                selected_validation_score = score
                selected_beam = beam_width
                selected_keep = keep
                selected_alpha = alpha
        del bundles
        gc.collect()

    print(
        f"selected epoch={selected_epoch} beam={selected_beam} keep={selected_keep} "
        f"forward_weight={selected_alpha:.1f} validation_score={selected_validation_score:.6f}",
        flush=True,
    )

    full_indices = list(range(len(train_rows)))
    if time.time() - START_TIME < FULL_TRAINING_CUTOFF_SECONDS:
        final_vocab = fit_vocabs(train_rows, full_indices)
        full_records = make_records(train_rows, full_indices, final_vocab)
        final_model, completed_epochs = train_fixed(
            full_records, final_vocab, selected_config, selected_epoch
        )
        print(f"final training epochs={completed_epochs}", flush=True)
        del selected_model
    else:
        warn("wall-clock guard selected the holdout-trained model for inference")
        final_model = selected_model
        final_vocab = inner_vocab

    predictions: list[list[list[str]]] = []
    for row in test_rows:
        try:
            bundle = candidate_bundle(
                row, final_model, final_vocab, selected_beam, selected_keep
            )
            chosen = [
                list(stream)
                for stream in choose_from_bundle(bundle, selected_keep, selected_alpha)
            ]
            if not valid_streams(row, chosen):
                raise ValueError("neural decoder returned a structurally invalid result")
            predictions.append(chosen)
        except Exception as exc:
            warn(f"per-row inference fallback for {row['id']}: {exc}")
            predictions.append(structural_partition(row))

    metadata = {
        "training_rows": len(training_indices),
        "validation_rows": len(validation_indices),
        "validation_score": selected_validation_score,
        "beam_width": selected_beam,
        "candidate_keep": selected_keep,
        "forward_weight": selected_alpha,
        "epoch": selected_epoch,
    }
    return predictions, metadata


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python3 solution.py <public_dir> <submission_out>", file=sys.stderr)
        raise SystemExit(2)

    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required input: " + ", ".join(missing))

    raw_test = read_csv_rows(test_path)
    test_rows = [parse_packet(raw, labeled=False) for raw in raw_test]

    placeholder = []
    for row in test_rows:
        try:
            placeholder.append(structural_partition(row))
        except Exception as exc:
            raise RuntimeError(f"cannot create valid placeholder for {row['id']}: {exc}") from exc
    if not submission_is_complete(test_rows, placeholder):
        raise RuntimeError("placeholder submission failed strict structural validation")
    write_submission(submission_out, test_rows, placeholder)
    print(f"wrote early valid placeholder: {submission_out}", flush=True)

    raw_train = read_csv_rows(train_path)
    train_rows = []
    for raw in raw_train:
        try:
            row = parse_packet(raw, labeled=True)
            gold = [[str(token) for token in stream] for stream in row["answers"]]
            row["answers"] = gold
            if not valid_streams(row, gold):
                warn(f"skipping structurally invalid training row {row['id']}")
                continue
            train_rows.append(row)
        except Exception as exc:
            warn(f"skipping unreadable training row: {exc}")
    if not train_rows:
        warn("no usable training rows; retaining placeholder submission")
        return

    try:
        predictions, metadata = run_learning(train_rows, test_rows)
    except Exception as exc:
        warn(f"training/inference failed; retaining placeholder submission: {exc}")
        return

    if not submission_is_complete(test_rows, predictions):
        warn("final predictions failed strict validation; retaining placeholder submission")
        return
    write_submission(submission_out, test_rows, predictions)

    written = read_csv_rows(submission_out)
    if [row.get("id") for row in written] != [row["id"] for row in test_rows]:
        warn("written ID audit failed; restoring placeholder submission")
        write_submission(submission_out, test_rows, placeholder)
        return
    written_columns = list(written[0].keys()) if written else []
    if written_columns != ["id", "answer_json"]:
        warn("written column audit failed; restoring placeholder submission")
        write_submission(submission_out, test_rows, placeholder)
        return
    if any(not row.get("answer_json") for row in written):
        warn("written prediction audit failed; restoring placeholder submission")
        write_submission(submission_out, test_rows, placeholder)
        return

    print(
        f"wrote {len(predictions)} predictions to {submission_out}; "
        f"validation_score={metadata['validation_score']:.6f}; "
        f"elapsed={time.time() - START_TIME:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
