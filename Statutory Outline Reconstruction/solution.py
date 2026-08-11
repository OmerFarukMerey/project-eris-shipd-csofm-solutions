#!/usr/bin/env python3
"""Train-only depth-transition model for statutory outline reconstruction."""

import json
import math
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 73129
TRAIN_DEADLINE = 3150.0
BACKBONE = "microsoft/deberta-v3-small"
MAX_TOKENS = 160
START = time.time()
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]", re.I)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def log(*x):
    print(*x, file=sys.stderr, flush=True)


def parse_provisions(value):
    result = json.loads(value)
    if not isinstance(result, list) or not result:
        raise ValueError("provisions_json must be a nonempty JSON array")
    return [str(x) for x in result]


def valid_forest(parents, n):
    return len(parents) == n and all(
        isinstance(parent, (int, np.integer)) and (parent == -1 or 0 <= parent < i)
        for i, parent in enumerate(parents)
    )


def depths(parents):
    result = []
    for parent in parents:
        result.append(0 if parent < 0 else result[parent] + 1)
    return result


def parents_from_depths(sequence):
    result = []
    latest = {}
    for i, depth in enumerate(sequence):
        if depth == 0:
            result.append(-1)
        else:
            result.append(latest[depth - 1])
        latest[depth] = i
        for stale in [key for key in latest if key > depth]:
            del latest[stale]
    return result


def sibling_pairs(parents):
    groups = {}
    for i, parent in enumerate(parents):
        groups.setdefault(parent, []).append(i)
    return {
        (a, b)
        for group in groups.values()
        for position, a in enumerate(group)
        for b in group[position + 1:]
    }


def raw_score(truth, prediction):
    n = len(truth)
    parent_accuracy = float(np.mean(np.asarray(truth) == np.asarray(prediction)))
    true_depth = depths(truth)
    pred_depth = depths(prediction)
    span = max(max(true_depth), 1)
    depth_score = float(np.mean([
        max(0.0, 1.0 - abs(a - b) / span) for a, b in zip(true_depth, pred_depth)
    ]))
    true_pairs = sibling_pairs(truth)
    pred_pairs = sibling_pairs(prediction)
    if not true_pairs and not pred_pairs:
        pair_f1 = 1.0
    elif not true_pairs or not pred_pairs:
        pair_f1 = 0.0
    else:
        common = len(true_pairs & pred_pairs)
        precision = common / len(pred_pairs)
        recall = common / len(true_pairs)
        pair_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return 0.55 * parent_accuracy + 0.25 * depth_score + 0.20 * pair_f1


def challenge_score(truths, predictions):
    values = []
    for truth, prediction in zip(truths, predictions):
        raw = raw_score(truth, prediction)
        trivial = raw_score(truth, [-1] * len(truth))
        values.append(max(0.0, (raw - trivial) / max(1.0 - trivial, 1e-12)))
    return float(np.mean(values))


def write_submission(path, act_ids, predictions):
    frame = pd.DataFrame({
        "act_id": [str(x) for x in act_ids],
        "parents_json": [json.dumps([int(v) for v in p], separators=(",", ":")) for p in predictions],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def surface_features(text):
    length = max(len(text), 1)
    token_count = len(TOKEN_PATTERN.findall(text.lower()))
    punctuation = [",", ";", ":", ".", "(", ")", "-"]
    values = [
        math.log1p(len(text)) / math.log(1202.0),
        math.log1p(token_count) / math.log(302.0),
        sum(c.isdigit() for c in text) / length,
        sum(c.isupper() for c in text) / length,
        sum(c.isspace() for c in text) / length,
    ]
    values.extend(min(text.count(mark), 12) / 12.0 for mark in punctuation)
    return values


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    torch.set_num_threads(min(10, os.cpu_count() or 1))
    return torch.device("cpu")


def load_backbone(device):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE, local_files_only=True)
    model = AutoModel.from_pretrained(BACKBONE, local_files_only=True).to(device).eval()
    return tokenizer, model


def embed_texts(text_acts, tokenizer, backbone, device, batch_size):
    flat = [text for act in text_acts for text in act]
    pieces = []
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            batch_text = flat[start:start + batch_size]
            encoded = tokenizer(
                batch_text, padding=True, truncation=True, max_length=MAX_TOKENS,
                return_tensors="pt"
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = backbone(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pieces.append(pooled.float().cpu().numpy().astype(np.float16))
    matrix = np.concatenate(pieces).astype(np.float32)
    result = []
    offset = 0
    for act in text_acts:
        result.append(matrix[offset:offset + len(act)])
        offset += len(act)
    return result


def make_items(ids, text_acts, embeddings, parent_acts=None):
    items = []
    for row, (act_id, texts, embedding) in enumerate(zip(ids, text_acts, embeddings)):
        item = {
            "act_id": str(act_id),
            "embedding": embedding,
            "features": np.asarray([surface_features(text) for text in texts], np.float32),
        }
        if parent_acts is not None:
            parent = [int(x) for x in parent_acts[row]]
            item["parents"] = parent
            item["depths"] = depths(parent)
            item["transitions"] = [0] + [
                item["depths"][i] - item["depths"][i - 1] + 6
                for i in range(1, len(parent))
            ]
        items.append(item)
    return items


def collate(items, targets):
    batch_size = len(items)
    nodes = max(len(item["embedding"]) for item in items)
    embedding_dim = items[0]["embedding"].shape[1]
    feature_dim = items[0]["features"].shape[1]
    embedding = np.zeros((batch_size, nodes, embedding_dim), np.float32)
    features = np.zeros((batch_size, nodes, feature_dim), np.float32)
    mask = np.zeros((batch_size, nodes), bool)
    lengths = np.zeros(batch_size, np.int64)
    if targets:
        depth_target = np.zeros((batch_size, nodes), np.int64)
        transition_target = np.zeros((batch_size, nodes), np.int64)
    for b, item in enumerate(items):
        n = len(item["embedding"])
        embedding[b, :n] = item["embedding"]
        features[b, :n] = item["features"]
        mask[b, :n] = True
        lengths[b] = n
        if targets:
            depth_target[b, :n] = item["depths"]
            transition_target[b, :n] = item["transitions"]
    result = {
        "embedding": torch.from_numpy(embedding),
        "features": torch.from_numpy(features),
        "mask": torch.from_numpy(mask),
        "lengths": torch.from_numpy(lengths),
        "items": items,
    }
    if targets:
        result["depth_target"] = torch.from_numpy(depth_target)
        result["transition_target"] = torch.from_numpy(transition_target)
    return result


def loader(items, batch_size, shuffle, targets, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        items, batch_size=batch_size, shuffle=shuffle, num_workers=0,
        collate_fn=lambda rows: collate(rows, targets), generator=generator
    )


def move(batch, device):
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


class DepthTransitionModel(nn.Module):
    def __init__(self, embedding_dim, feature_dim, maximum_nodes, depth_classes,
                 hidden_dim=256, dropout=0.18):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(embedding_dim + feature_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )
        self.position = nn.Embedding(maximum_nodes, hidden_dim)
        self.sequence = nn.GRU(
            hidden_dim, hidden_dim // 2, num_layers=3, batch_first=True,
            bidirectional=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden_dim, depth_classes)
        )
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(hidden_dim, depth_classes + 1)
        )
        self.maximum_nodes = maximum_nodes

    def forward(self, batch):
        nodes = batch["embedding"].shape[1]
        x = self.input(torch.cat([batch["embedding"], batch["features"]], -1))
        positions = torch.arange(nodes, device=x.device).clamp(max=self.maximum_nodes - 1)
        x = x + self.position(positions).unsqueeze(0)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, batch["lengths"].cpu(), batch_first=True, enforce_sorted=False
        )
        packed_h, _ = self.sequence(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_h, batch_first=True, total_length=nodes)
        h = self.norm(h + x)
        before, after = h[:, :-1], h[:, 1:]
        pair = torch.cat([before, after, before * after, (before - after).abs()], -1)
        return self.depth_head(h), self.transition_head(pair)


def viterbi_decode(depth_logits, transition_logits, transition_weight):
    n, classes = depth_logits.shape
    dynamic = np.full((n, classes), -1e30, np.float64)
    back = np.zeros((n, classes), np.int16)
    dynamic[0, 0] = float(depth_logits[0, 0])
    for i in range(1, n):
        for current in range(classes):
            values = np.full(classes, -1e30, np.float64)
            for previous in range(classes):
                delta = current - previous
                if -6 <= delta <= 1:
                    values[previous] = (
                        dynamic[i - 1, previous]
                        + transition_weight * float(transition_logits[i - 1, delta + 6])
                    )
            chosen = int(values.argmax())
            dynamic[i, current] = values[chosen] + float(depth_logits[i, current])
            back[i, current] = chosen
    current = int(dynamic[-1].argmax())
    sequence = [current]
    for i in range(n - 1, 0, -1):
        current = int(back[i, current])
        sequence.append(current)
    sequence.reverse()
    return sequence


def predict_weights(model, items, device, batch_size, weights):
    result = {float(weight): [] for weight in weights}
    model.eval()
    with torch.no_grad():
        for raw in loader(items, batch_size, False, False, SEED):
            try:
                batch = move(raw, device)
                depth_logits, transition_logits = model(batch)
                depth_logits = depth_logits.float().cpu().numpy()
                transition_logits = transition_logits.float().cpu().numpy()
                for row, item in enumerate(raw["items"]):
                    n = len(item["embedding"])
                    for weight in weights:
                        sequence = viterbi_decode(
                            depth_logits[row, :n], transition_logits[row, :max(n - 1, 0)], float(weight)
                        )
                        result[float(weight)].append(parents_from_depths(sequence))
            except Exception:
                if len(raw["items"]) == 1:
                    item = raw["items"][0]
                    for weight in weights:
                        result[float(weight)].append([-1] * len(item["embedding"]))
                    continue
                for item in raw["items"]:
                    single = predict_weights(model, [item], device, 1, weights)
                    for weight in weights:
                        result[float(weight)].append(single[float(weight)][0])
    return result

def collect_logits(model, items, device, batch_size):
    result = []
    model.eval()
    with torch.no_grad():
        for raw in loader(items, batch_size, False, False, SEED):
            try:
                batch = move(raw, device)
                depth_logits, transition_logits = model(batch)
                depth_logits = depth_logits.float().cpu().numpy()
                transition_logits = transition_logits.float().cpu().numpy()
                for row, item in enumerate(raw["items"]):
                    n = len(item["embedding"])
                    result.append((depth_logits[row, :n], transition_logits[row, :max(n - 1, 0)]))
            except Exception:
                if len(raw["items"]) == 1:
                    result.append(None)
                else:
                    for item in raw["items"]:
                        result.extend(collect_logits(model, [item], device, 1))
    return result


def decode_ensemble(logit_sets, items, weights):
    result = {float(weight): [] for weight in weights}
    for index, item in enumerate(items):
        available = [values[index] for values in logit_sets if values[index] is not None]
        if not available:
            for weight in weights:
                result[float(weight)].append([-1] * len(item["embedding"]))
            continue
        depth_logits = np.mean([value[0] for value in available], axis=0)
        transition_logits = np.mean([value[1] for value in available], axis=0)
        for weight in weights:
            sequence = viterbi_decode(depth_logits, transition_logits, float(weight))
            result[float(weight)].append(parents_from_depths(sequence))
    return result


def train_epoch(model, items, optimizer, device, batch_size, epoch):
    model.train()
    losses = []
    for raw in loader(items, batch_size, True, True, SEED + epoch):
        batch = move(raw, device)
        optimizer.zero_grad(set_to_none=True)
        depth_logits, transition_logits = model(batch)
        mask = batch["mask"]
        transition_mask = mask[:, 1:]
        depth_loss = F.cross_entropy(
            depth_logits.transpose(1, 2), batch["depth_target"], reduction="none"
        )
        transition_loss = F.cross_entropy(
            transition_logits.transpose(1, 2), batch["transition_target"][:, 1:], reduction="none"
        )
        per_act = (
            (depth_loss * mask).sum(1) / mask.sum(1)
            + (transition_loss * transition_mask).sum(1) / transition_mask.sum(1).clamp(min=1)
        )
        loss = per_act.mean()
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    public_dir = Path(sys.argv[1])
    submission_path = Path(sys.argv[2])
    train_path, test_path = public_dir / "train.csv", public_dir / "test.csv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("train.csv and test.csv are required")
    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    if not {"act_id", "provisions_json", "parents_json"}.issubset(train_frame.columns):
        raise ValueError("train.csv columns are invalid")
    if not {"act_id", "provisions_json"}.issubset(test_frame.columns):
        raise ValueError("test.csv columns are invalid")

    test_texts, fallback = [], []
    for value in test_frame["provisions_json"]:
        try:
            texts = parse_provisions(value)
        except Exception as exc:
            log("Malformed test row:", repr(exc))
            texts = [""]
        test_texts.append(texts)
        fallback.append([-1] * len(texts))
    write_submission(submission_path, test_frame["act_id"], fallback)
    log("placeholder written", submission_path)

    train_ids, train_texts, train_parents = [], [], []
    for row in train_frame.itertuples(index=False):
        texts = parse_provisions(row.provisions_json)
        parents = [int(x) for x in json.loads(row.parents_json)]
        if valid_forest(parents, len(texts)):
            train_ids.append(row.act_id)
            train_texts.append(texts)
            train_parents.append(parents)
    if len(train_texts) < 20:
        log("too few valid training acts; placeholder retained")
        return

    device = select_device()
    batch_size = 32 if device.type == "cuda" else (24 if device.type == "mps" else 8)
    embedding_batch = 96 if device.type == "cuda" else (64 if device.type == "mps" else 16)
    try:
        tokenizer, backbone = load_backbone(device)
        train_embeddings = embed_texts(train_texts, tokenizer, backbone, device, embedding_batch)
        # Frozen transform/predict only; no test-derived fit or statistic.
        test_embeddings = embed_texts(test_texts, tokenizer, backbone, device, embedding_batch)
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        log("bundled backbone unavailable; placeholder retained:", repr(exc))
        return

    train_items = make_items(train_ids, train_texts, train_embeddings, train_parents)
    test_items = make_items(test_frame["act_id"], test_texts, test_embeddings)
    order = np.arange(len(train_items))
    split_rng = np.random.default_rng(SEED)
    split_rng.shuffle(order)
    holdout_count = max(1, int(round(0.15 * len(order))))
    held = set(order[:holdout_count].tolist())
    fit_items = [item for i, item in enumerate(train_items) if i not in held]
    validation_items = [item for i, item in enumerate(train_items) if i in held]
    maximum_nodes = max(len(item["embedding"]) for item in train_items)
    depth_classes = max(max(item["depths"]) for item in train_items) + 1
    model_args = (
        train_items[0]["embedding"].shape[1], train_items[0]["features"].shape[1],
        maximum_nodes, depth_classes
    )

    model = DepthTransitionModel(*model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
    search_weights = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    best_score, best_epoch, best_weight = -1.0, 1, 0.0
    snapshots = []
    truths = [item["parents"] for item in validation_items]
    for epoch in range(1, 27):
        if time.time() - START >= TRAIN_DEADLINE - 600:
            break
        loss = train_epoch(model, fit_items, optimizer, device, batch_size, epoch)
        sets = predict_weights(model, validation_items, device, batch_size, search_weights)
        scores = {weight: challenge_score(truths, predictions) for weight, predictions in sets.items()}
        weight = max(search_weights, key=lambda value: (scores[value], -value))
        score = scores[weight]
        log(f"epoch={epoch} loss={loss:.6f} validation_score={score:.6f} transition_weight={weight}")
        if score > best_score:
            best_score, best_epoch, best_weight = score, epoch, float(weight)
        if len(snapshots) < 3 or score > snapshots[-1][0]:
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            snapshots.append((score, epoch, state))
            snapshots.sort(key=lambda value: value[0], reverse=True)
            snapshots = snapshots[:3]
    if not snapshots:
        log("training did not finish an epoch; placeholder retained")
        return

    validation_logits = []
    ensemble_score, ensemble_count, ensemble_weight = -1.0, 1, best_weight
    for count, (_, _, state) in enumerate(snapshots, start=1):
        model.load_state_dict(state)
        validation_logits.append(collect_logits(model, validation_items, device, batch_size))
        sets = decode_ensemble(validation_logits, validation_items, search_weights)
        scores = {weight: challenge_score(truths, values) for weight, values in sets.items()}
        weight = max(search_weights, key=lambda value: (scores[value], -value))
        if scores[weight] > ensemble_score:
            ensemble_score, ensemble_count, ensemble_weight = scores[weight], count, float(weight)
    selected = snapshots[:ensemble_count]
    log(f"selected_snapshots={[(value[1], round(value[0], 6)) for value in selected]} "
        f"validation_score={ensemble_score:.6f} transition_weight={ensemble_weight}")

    test_logits = []
    for _, _, state in selected:
        model.load_state_dict(state)
        test_logits.append(collect_logits(model, test_items, device, batch_size))
    predictions = fallback.copy()
    predicted = decode_ensemble(test_logits, test_items, (ensemble_weight,))[ensemble_weight]
    for i, value in enumerate(predicted):
        if valid_forest(value, len(test_texts[i])):
            predictions[i] = value
    write_submission(submission_path, test_frame["act_id"], predictions)

    check = pd.read_csv(submission_path, keep_default_na=False)
    if list(check.columns) != ["act_id", "parents_json"] or len(check) != len(test_frame):
        raise RuntimeError("submission completeness failure")
    if check["act_id"].duplicated().any() or set(check["act_id"]) != set(test_frame["act_id"]):
        raise RuntimeError("submission id failure")
    for i, value in enumerate(check["parents_json"]):
        if not valid_forest(json.loads(value), len(test_texts[i])):
            raise RuntimeError("submission forest failure")
    log("submission complete", submission_path, "elapsed", round(time.time() - START, 1))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*nested tensors.*")
    main()
