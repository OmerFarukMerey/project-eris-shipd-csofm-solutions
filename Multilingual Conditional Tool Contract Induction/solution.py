#!/usr/bin/env python3
"""CPU-only multilingual conditional tool-contract induction."""

import csv
import gc
import json
import math
import os
import random
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "10")

SEED = 314159
ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GENERATOR_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TRAINING_CUTOFF_SECONDS = 2100.0
INFERENCE_GUARD_SECONDS = 5000.0
TOOL_PATTERN = re.compile(r"[a-z0-9_]{1,64}\Z")
CONTRACT_KEYS = [
    "target_tool",
    "routing_rule",
    "required_arguments",
    "optional_arguments",
    "peer_tool",
]


def log(message):
    print(f"[eris] {message}", flush=True)


def safe_json(value, fallback):
    try:
        parsed = json.loads(value)
        return parsed
    except Exception:
        return fallback


def safe_examples(value, expected):
    parsed = safe_json(value, [])
    result = []
    if isinstance(parsed, list):
        for item in parsed[:expected]:
            if isinstance(item, dict):
                result.append(
                    {
                        "locale": str(item.get("locale", "und")),
                        "request": str(item.get("request", "")),
                    }
                )
    while len(result) < expected:
        result.append({"locale": "und", "request": ""})
    return result


def safe_registry(value):
    parsed = safe_json(value, [])
    result = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            description = str(item.get("description", name.replace("_", " ")))
            if name and name not in {entry[0] for entry in result}:
                result.append((name, description))
    return result


def parse_target(value, registry_names):
    parsed = safe_json(value, {})
    if not isinstance(parsed, dict) or set(parsed) != set(CONTRACT_KEYS):
        raise ValueError("invalid induced_contract keys")
    route = parsed.get("routing_rule")
    if not isinstance(route, dict) or set(route) != {"argument", "operator"}:
        raise ValueError("invalid routing_rule")
    target = str(parsed["target_tool"])
    peer = str(parsed["peer_tool"])
    argument = str(route["argument"])
    operator = str(route["operator"])
    required = parsed["required_arguments"]
    optional = parsed["optional_arguments"]
    if not TOOL_PATTERN.fullmatch(target) or not TOOL_PATTERN.fullmatch(peer):
        raise ValueError("invalid tool token")
    if argument not in registry_names or operator not in {"required", "forbidden"}:
        raise ValueError("invalid route")
    if not isinstance(required, list) or not isinstance(optional, list):
        raise ValueError("invalid interface arrays")
    required = sorted(set(str(item) for item in required))
    optional = sorted(set(str(item) for item in optional))
    if any(item not in registry_names for item in required + optional):
        raise ValueError("unregistered interface argument")
    if set(required) & set(optional):
        raise ValueError("overlapping interface arrays")
    return {
        "target_tool": target,
        "routing_rule": {"argument": argument, "operator": operator},
        "required_arguments": required,
        "optional_arguments": optional,
        "peer_tool": peer,
    }


def prediction_json(contract):
    ordered = {
        "target_tool": str(contract["target_tool"]),
        "routing_rule": {
            "argument": str(contract["routing_rule"]["argument"]),
            "operator": str(contract["routing_rule"]["operator"]),
        },
        "required_arguments": sorted(set(contract["required_arguments"])),
        "optional_arguments": sorted(set(contract["optional_arguments"])),
        "peer_tool": str(contract["peer_tool"]),
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def fallback_contract(registry_names):
    argument = registry_names[0] if registry_names else "unknown"
    return {
        "target_tool": "unknown",
        "routing_rule": {"argument": argument, "operator": "required"},
        "required_arguments": [],
        "optional_arguments": [],
        "peer_tool": "unknown_peer",
    }


def write_submission(path, ids, contracts):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["contract_id", "induced_contract"]
        )
        writer.writeheader()
        for row_id, contract in zip(ids, contracts):
            writer.writerow(
                {
                    "contract_id": row_id,
                    "induced_contract": prediction_json(contract),
                }
            )


def parse_records(frame, labelled):
    records = []
    skipped = 0
    for raw in frame.to_dict(orient="records"):
        registry = safe_registry(raw.get("argument_registry_json", "[]"))
        registry_names = [entry[0] for entry in registry]
        try:
            if not registry_names:
                raise ValueError("empty registry")
            target = (
                parse_target(raw.get("induced_contract", "{}"), set(registry_names))
                if labelled
                else None
            )
            records.append(
                {
                    "id": str(raw.get("contract_id", "")),
                    "positive": safe_examples(raw.get("positive_examples_json", "[]"), 6),
                    "contrast": safe_examples(raw.get("contrast_examples_json", "[]"), 4),
                    "registry": registry,
                    "target": target,
                }
            )
        except Exception as exc:
            skipped += 1
            if not labelled:
                records.append(
                    {
                        "id": str(raw.get("contract_id", "")),
                        "positive": safe_examples(raw.get("positive_examples_json", "[]"), 6),
                        "contrast": safe_examples(raw.get("contrast_examples_json", "[]"), 4),
                        "registry": registry,
                        "target": None,
                        "parse_error": str(exc),
                    }
                )
    if skipped:
        log(f"warning: {skipped} rows required parser fallbacks")
    return records


def load_encoder(torch):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = AutoModel.from_pretrained(ENCODER_NAME)
    model.to("cpu")
    model.eval()
    return tokenizer, model


def encode_texts(texts, tokenizer, model, torch, batch_size=96, max_length=96):
    if not texts:
        width = int(getattr(model.config, "hidden_size", 384))
        return torch.empty((0, width), dtype=torch.float32).numpy()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            outputs.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
    return torch.cat(outputs, dim=0).numpy()


def episode_embeddings(records, tokenizer, model, torch):
    texts = []
    positive_indices = []
    contrast_indices = []
    for record in records:
        current = []
        for example in record["positive"]:
            current.append(len(texts))
            texts.append(example["request"])
        positive_indices.append(current)
        current = []
        for example in record["contrast"]:
            current.append(len(texts))
            texts.append(example["request"])
        contrast_indices.append(current)
    matrix = encode_texts(texts, tokenizer, model, torch)
    positive = matrix[positive_indices]
    contrast = matrix[contrast_indices]
    return positive, contrast


def normalize_rows(matrix, np):
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def set_f1(truth, prediction):
    truth = set(truth)
    prediction = set(prediction)
    if not truth and not prediction:
        return 1.0
    if not truth or not prediction:
        return 0.0
    return 2.0 * len(truth & prediction) / (len(truth) + len(prediction))


def sibling_validation_tools(records, np):
    by_family = defaultdict(list)
    for record in records:
        tool = record["target"]["target_tool"]
        by_family[tool.split("_", 1)[0]].append(tool)
    rng = np.random.default_rng(SEED)
    selected = []
    families = [family for family, tools in sorted(by_family.items()) if len(set(tools)) >= 2]
    rng.shuffle(families)
    for family in families:
        tools = sorted(set(by_family[family]))
        selected.append(tools[int(rng.integers(0, len(tools)))])
        if len(selected) >= 8:
            break
    if not selected:
        tools = sorted({record["target"]["target_tool"] for record in records})
        selected = tools[:: max(1, len(tools) // 5)][: max(1, len(tools) // 5)]
    return set(selected)


def poisson_categories(instance_probabilities, torch):
    batch, _, arguments = instance_probabilities.shape
    d0 = torch.ones(
        (batch, arguments), dtype=instance_probabilities.dtype, device=instance_probabilities.device
    )
    d1 = torch.zeros_like(d0)
    d2 = torch.zeros_like(d0)
    for position in range(instance_probabilities.shape[1]):
        probability = instance_probabilities[:, position]
        inverse = 1.0 - probability
        d2, d1, d0 = (
            d2 * inverse + d1 * probability,
            d1 * inverse + d0 * probability,
            d0 * inverse,
        )
    tail = (1.0 - d0 - d1 - d2).clamp_min(1e-8)
    return torch.stack([d0, d1 + d2, tail], dim=-1).clamp_min(1e-8)


def make_contract_model(torch, argument_embeddings):
    class ContractModel(torch.nn.Module):
        def __init__(self, argument_matrix):
            super().__init__()
            arguments, width = argument_matrix.shape
            self.arguments = arguments
            self.register_buffer("argument_base", argument_matrix.clone())
            self.argument_delta = torch.nn.Parameter(torch.zeros_like(argument_matrix))
            self.argument_bias = torch.nn.Parameter(torch.full((arguments,), -2.0))
            self.log_scale = torch.nn.Parameter(torch.tensor(2.0))
            self.instance_residual = torch.nn.Sequential(
                torch.nn.Linear(width, 128),
                torch.nn.GELU(),
                torch.nn.Dropout(0.10),
                torch.nn.Linear(128, arguments),
            )
            torch.nn.init.zeros_(self.instance_residual[-1].weight)
            torch.nn.init.zeros_(self.instance_residual[-1].bias)
            self.argument_identity = torch.nn.Embedding(arguments, 16)
            self.route_head = torch.nn.Sequential(
                torch.nn.Linear(42, 64),
                torch.nn.GELU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(64, 32),
                torch.nn.GELU(),
                torch.nn.Linear(32, 2),
            )

        def prototypes(self):
            return torch.nn.functional.normalize(
                self.argument_base + 0.15 * self.argument_delta, p=2, dim=-1
            )

        def instance_logits(self, examples):
            prototypes = self.prototypes()
            return (
                examples @ prototypes.T * self.log_scale.exp()
                + self.argument_bias
                + self.instance_residual(examples)
            )

        def forward(self, positive, contrast):
            positive_logits = self.instance_logits(positive)
            contrast_logits = self.instance_logits(contrast)
            positive_probabilities = torch.sigmoid(positive_logits)
            contrast_probabilities = torch.sigmoid(contrast_logits)
            categories = poisson_categories(positive_probabilities, torch)
            prototypes = self.prototypes()
            positive_cosine = positive @ prototypes.T
            contrast_cosine = contrast @ prototypes.T
            positive_probability_sorted = positive_probabilities.sort(dim=1, descending=True).values
            contrast_probability_sorted = contrast_probabilities.sort(dim=1, descending=True).values
            positive_cosine_sorted = positive_cosine.sort(dim=1, descending=True).values
            contrast_cosine_sorted = contrast_cosine.sort(dim=1, descending=True).values
            probability_summary = torch.stack(
                [
                    positive_probabilities.mean(dim=1),
                    contrast_probabilities.mean(dim=1),
                    positive_probabilities.mean(dim=1) - contrast_probabilities.mean(dim=1),
                ],
                dim=-1,
            )
            cosine_summary = torch.stack(
                [
                    positive_cosine.mean(dim=1),
                    contrast_cosine.mean(dim=1),
                    positive_cosine.mean(dim=1) - contrast_cosine.mean(dim=1),
                ],
                dim=-1,
            )
            identity = self.argument_identity.weight.unsqueeze(0).expand(positive.shape[0], -1, -1)
            route_features = torch.cat(
                [
                    positive_probability_sorted.transpose(1, 2),
                    contrast_probability_sorted.transpose(1, 2),
                    positive_cosine_sorted.transpose(1, 2),
                    contrast_cosine_sorted.transpose(1, 2),
                    probability_summary,
                    cosine_summary,
                    identity,
                ],
                dim=-1,
            )
            route_output = self.route_head(route_features)
            return categories, route_output[..., 0], route_output[..., 1]

    return ContractModel(argument_embeddings)


def prepare_contract_targets(records, argument_names, np):
    index = {name: position for position, name in enumerate(argument_names)}
    category = np.zeros((len(records), len(argument_names)), dtype=np.int64)
    route = np.zeros(len(records), dtype=np.int64)
    operator = np.zeros(len(records), dtype=np.float32)
    for row, record in enumerate(records):
        target = record["target"]
        for name in target["optional_arguments"]:
            category[row, index[name]] = 1
        for name in target["required_arguments"]:
            category[row, index[name]] = 2
        route[row] = index[target["routing_rule"]["argument"]]
        operator[row] = float(target["routing_rule"]["operator"] == "required")
    return category, route, operator


def decode_contract_outputs(
    category_probabilities,
    route_scores,
    operator_logits,
    argument_names,
    optional_bias,
    required_bias,
    operator_threshold,
    np,
):
    adjusted = np.log(np.maximum(category_probabilities, 1e-9))
    adjusted[:, :, 1] += optional_bias
    adjusted[:, :, 2] += required_bias
    categories = adjusted.argmax(axis=-1)
    routes = route_scores.argmax(axis=-1)
    chosen_operator_logits = operator_logits[np.arange(len(routes)), routes]
    operators = chosen_operator_logits >= operator_threshold
    for row, route in enumerate(routes):
        categories[row, route] = 2 if operators[row] else 0
    decoded = []
    for row, route in enumerate(routes):
        decoded.append(
            {
                "argument": argument_names[int(route)],
                "operator": "required" if operators[row] else "forbidden",
                "required": [
                    argument_names[column]
                    for column in range(len(argument_names))
                    if categories[row, column] == 2
                ],
                "optional": [
                    argument_names[column]
                    for column in range(len(argument_names))
                    if categories[row, column] == 1
                ],
            }
        )
    return decoded


def structured_validation(records, indices, decoded):
    argument_correct = []
    operator_correct = []
    required_scores = []
    optional_scores = []
    row_scores = []
    for local_row, source_row in enumerate(indices):
        truth = records[int(source_row)]["target"]
        prediction = decoded[local_row]
        arg_ok = prediction["argument"] == truth["routing_rule"]["argument"]
        op_ok = prediction["operator"] == truth["routing_rule"]["operator"]
        required_f1 = set_f1(truth["required_arguments"], prediction["required"])
        optional_f1 = set_f1(truth["optional_arguments"], prediction["optional"])
        routing = 0.5 * (float(arg_ok) + float(op_ok))
        interface = 0.5 * (required_f1 + optional_f1)
        component_base = 0.30 * routing + 0.20 * required_f1 + 0.20 * optional_f1
        row_scores.append(component_base * 0.5)
        argument_correct.append(arg_ok)
        operator_correct.append(op_ok)
        required_scores.append(required_f1)
        optional_scores.append(optional_f1)
    return {
        "route_argument": float(sum(argument_correct) / max(1, len(argument_correct))),
        "operator": float(sum(operator_correct) / max(1, len(operator_correct))),
        "required_f1": float(sum(required_scores) / max(1, len(required_scores))),
        "optional_f1": float(sum(optional_scores) / max(1, len(optional_scores))),
        "metric_lower_bound": float(sum(row_scores) / max(1, len(row_scores))),
    }


def train_contract_model(
    records,
    positive_embeddings,
    contrast_embeddings,
    argument_embeddings,
    argument_names,
    torch,
    np,
):
    category, route, operator = prepare_contract_targets(records, argument_names, np)
    validation_tools = sibling_validation_tools(records, np)
    validation_indices = np.array(
        [
            row
            for row, record in enumerate(records)
            if record["target"]["target_tool"] in validation_tools
        ],
        dtype=np.int64,
    )
    training_indices = np.array(
        [row for row in range(len(records)) if row not in set(validation_indices.tolist())],
        dtype=np.int64,
    )
    positive_tensor = torch.tensor(positive_embeddings, dtype=torch.float32)
    contrast_tensor = torch.tensor(contrast_embeddings, dtype=torch.float32)
    category_tensor = torch.tensor(category, dtype=torch.long)
    route_tensor = torch.tensor(route, dtype=torch.long)
    operator_tensor = torch.tensor(operator, dtype=torch.float32)

    def fit(indices, epochs, collect_best):
        torch.manual_seed(SEED)
        model = make_contract_model(torch, argument_embeddings)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
        counts = np.bincount(category[indices].ravel(), minlength=3)
        weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
        weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
        generator = np.random.default_rng(SEED)
        best = None
        for epoch in range(1, epochs + 1):
            model.train()
            order = generator.permutation(indices)
            for start in range(0, len(order), 40):
                rows = order[start : start + 40]
                probabilities, route_scores, operator_logits = model(
                    positive_tensor[rows], contrast_tensor[rows]
                )
                selected = probabilities.gather(
                    -1, category_tensor[rows].unsqueeze(-1)
                ).squeeze(-1)
                interface_loss = (
                    -selected.log() * weights[category_tensor[rows]]
                ).mean()
                route_loss = torch.nn.functional.cross_entropy(
                    route_scores, route_tensor[rows]
                )
                chosen_operator = operator_logits[
                    torch.arange(len(rows)), route_tensor[rows]
                ]
                operator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    chosen_operator, operator_tensor[rows]
                )
                loss = interface_loss + 1.5 * route_loss + 0.5 * operator_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            if collect_best and epoch % 5 == 0:
                model.eval()
                with torch.inference_mode():
                    values = model(
                        positive_tensor[validation_indices],
                        contrast_tensor[validation_indices],
                    )
                probabilities = values[0].cpu().numpy()
                route_values = values[1].cpu().numpy()
                operator_values = values[2].cpu().numpy()
                decoded = decode_contract_outputs(
                    probabilities,
                    route_values,
                    operator_values,
                    argument_names,
                    0.0,
                    0.0,
                    0.0,
                    np,
                )
                metrics = structured_validation(records, validation_indices, decoded)
                selection = (
                    0.30 * 0.5 * (metrics["route_argument"] + metrics["operator"])
                    + 0.20 * metrics["required_f1"]
                    + 0.20 * metrics["optional_f1"]
                )
                if best is None or selection > best[0]:
                    best = (
                        selection,
                        epoch,
                        {name: value.detach().clone() for name, value in model.state_dict().items()},
                        probabilities,
                        route_values,
                        operator_values,
                    )
        return model, best

    validation_model, best = fit(training_indices, 120, True)
    if best is None:
        raise RuntimeError("structured validation did not complete")
    _, best_epoch, state, probabilities, route_values, operator_values = best
    validation_model.load_state_dict(state)
    best_calibration = None
    for optional_bias in np.linspace(-2.5, 0.5, 13):
        for required_bias in np.linspace(-0.5, 2.5, 13):
            for operator_threshold in np.linspace(-1.0, 1.0, 9):
                decoded = decode_contract_outputs(
                    probabilities,
                    route_values,
                    operator_values,
                    argument_names,
                    float(optional_bias),
                    float(required_bias),
                    float(operator_threshold),
                    np,
                )
                metrics = structured_validation(records, validation_indices, decoded)
                selection = (
                    0.30 * 0.5 * (metrics["route_argument"] + metrics["operator"])
                    + 0.20 * metrics["required_f1"]
                    + 0.20 * metrics["optional_f1"]
                )
                if best_calibration is None or selection > best_calibration[0]:
                    best_calibration = (
                        selection,
                        float(optional_bias),
                        float(required_bias),
                        float(operator_threshold),
                        metrics,
                    )
    _, optional_bias, required_bias, operator_threshold, metrics = best_calibration
    log(
        "sibling-tool validation: "
        f"tools={sorted(validation_tools)} epoch={best_epoch} "
        f"route_arg={metrics['route_argument']:.4f} operator={metrics['operator']:.4f} "
        f"required_f1={metrics['required_f1']:.4f} optional_f1={metrics['optional_f1']:.4f} "
        f"exact-metric lower_bound={metrics['metric_lower_bound']:.6f}"
    )
    del validation_model
    gc.collect()
    final_model, _ = fit(np.arange(len(records), dtype=np.int64), best_epoch, False)
    final_model.eval()
    return final_model, (optional_bias, required_bias, operator_threshold)


def fit_contrast_grouper(records, positive_embeddings, contrast_embeddings, np):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression

    labels = np.array(
        [record["target"]["target_tool"] for record in records for _ in range(6)]
    )
    examples = np.asarray(
        positive_embeddings.reshape(-1, positive_embeddings.shape[-1]),
        dtype=np.float64,
    )
    contrast_examples = np.asarray(
        contrast_embeddings.reshape(-1, contrast_embeddings.shape[-1]),
        dtype=np.float64,
    )
    classifier = LogisticRegression(C=4.0, max_iter=300, random_state=SEED)
    classifier.fit(examples, labels)
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(examples, labels)
    positive_lda = lda.transform(examples).reshape(len(records), 6, -1)
    contrast_lda = lda.transform(contrast_examples).reshape(len(records), 4, -1)
    positive_lda = normalize_rows(positive_lda, np)
    contrast_lda = normalize_rows(contrast_lda, np)
    class_index = {name: position for position, name in enumerate(classifier.classes_)}
    pseudo = []
    for row, record in enumerate(records):
        probabilities = classifier.predict_proba(
            np.asarray(contrast_embeddings[row], dtype=np.float64)
        )
        target_column = class_index[record["target"]["target_tool"]]
        peer_column = class_index[record["target"]["peer_tool"]]
        odds = np.log(probabilities[:, target_column] + 1e-9) - np.log(
            probabilities[:, peer_column] + 1e-9
        )
        pseudo.append(set(np.argsort(odds)[-2:].tolist()))
    best = None
    for raw_weight in np.linspace(0.0, 1.0, 9):
        agreements = []
        for row in range(len(records)):
            raw_centroid = normalize_rows(
                positive_embeddings[row].mean(axis=0, keepdims=True), np
            )[0]
            lda_centroid = normalize_rows(
                positive_lda[row].mean(axis=0, keepdims=True), np
            )[0]
            scores = raw_weight * (contrast_embeddings[row] @ raw_centroid)
            scores += (1.0 - raw_weight) * (contrast_lda[row] @ lda_centroid)
            agreements.append(set(np.argsort(scores)[-2:].tolist()) == pseudo[row])
        score = float(np.mean(agreements))
        if best is None or score > best[0]:
            best = (score, float(raw_weight))
    log(f"train-only contrast grouping agreement={best[0]:.4f}, raw_weight={best[1]:.3f}")
    return lda, best[1]


def select_peer_examples(positive, contrast, lda, raw_weight, np):
    positive64 = np.asarray(positive, dtype=np.float64)
    contrast64 = np.asarray(contrast, dtype=np.float64)
    raw_centroid = normalize_rows(positive64.mean(axis=0, keepdims=True), np)[0]
    positive_lda = normalize_rows(lda.transform(positive64), np)
    contrast_lda = normalize_rows(lda.transform(contrast64), np)
    lda_centroid = normalize_rows(positive_lda.mean(axis=0, keepdims=True), np)[0]
    similarities = raw_weight * (contrast @ raw_centroid)
    similarities += (1.0 - raw_weight) * (contrast_lda @ lda_centroid)
    target_indices = set(np.argsort(similarities)[-2:].tolist())
    return [position for position in range(4) if position not in target_indices]


def candidate_features(
    episode_examples,
    candidate_embeddings,
    candidate_tokens,
    family_classifier,
    ridge,
    np,
):
    episode_examples = np.asarray(episode_examples, dtype=np.float64)
    candidate_embeddings = np.asarray(candidate_embeddings, dtype=np.float64)
    if len(episode_examples) < 6:
        mean = episode_examples.mean(axis=0, keepdims=True)
        episode_examples = np.concatenate(
            [episode_examples, np.repeat(mean, 6 - len(episode_examples), axis=0)], axis=0
        )
    elif len(episode_examples) > 6:
        episode_examples = episode_examples[:6]
    centroid = normalize_rows(episode_examples.mean(axis=0, keepdims=True), np)[0]
    similarities = episode_examples @ candidate_embeddings.T
    sorted_similarities = np.sort(similarities, axis=0)[::-1].T
    predicted = ridge.predict(centroid.reshape(1, -1))[0]
    predicted = predicted / max(float(np.linalg.norm(predicted)), 1e-8)
    ridge_similarity = candidate_embeddings @ predicted
    family_probabilities = family_classifier.predict_proba(centroid.reshape(1, -1))[0]
    family_index = {
        family: position for position, family in enumerate(family_classifier.classes_)
    }
    floor = max(float(family_probabilities.min()) * 0.5, 1e-8)
    family_feature = np.array(
        [
            math.log(
                max(
                    float(
                        family_probabilities[
                            family_index.get(token.split("_", 1)[0], 0)
                        ]
                    )
                    if token.split("_", 1)[0] in family_index
                    else floor,
                    1e-8,
                )
            )
            for token in candidate_tokens
        ],
        dtype=np.float64,
    )
    lengths = np.array(
        [[len(token) / 64.0, len(token.split("_")) / 6.0] for token in candidate_tokens],
        dtype=np.float64,
    )
    return np.column_stack(
        [
            sorted_similarities,
            similarities.mean(axis=0),
            similarities.std(axis=0),
            ridge_similarity,
            family_feature,
            lengths,
        ]
    ).astype(np.float64)


def fit_open_label_models(
    records,
    positive_embeddings,
    label_embeddings,
    labels,
    validation_tools,
    np,
):
    from sklearn.linear_model import LogisticRegression, Ridge

    label_index = {label: position for position, label in enumerate(labels)}
    target_labels = np.array([record["target"]["target_tool"] for record in records])
    families = np.array([label.split("_", 1)[0] for label in target_labels])
    centroids = np.asarray(
        normalize_rows(positive_embeddings.mean(axis=1), np), dtype=np.float64
    )
    validation_mask = np.isin(target_labels, list(validation_tools))
    fit_mask = ~validation_mask
    fit_labels = sorted(set(target_labels[fit_mask]))
    fit_label_positions = [label_index[label] for label in fit_labels]
    best_family = None
    for regularization in (0.5, 2.0, 8.0):
        model = LogisticRegression(
            C=regularization, max_iter=300, random_state=SEED
        ).fit(centroids[fit_mask], families[fit_mask])
        prediction = model.predict(centroids[validation_mask])
        score = float(np.mean(prediction == families[validation_mask]))
        if best_family is None or score > best_family[0]:
            best_family = (score, regularization)
    best = None
    for alpha in (0.3, 3.0, 30.0):
        ridge = Ridge(alpha=alpha).fit(
            centroids[fit_mask],
            label_embeddings[
                [label_index[label] for label in target_labels[fit_mask]]
            ],
        )
        family = LogisticRegression(
            C=best_family[1], max_iter=300, random_state=SEED
        ).fit(centroids[fit_mask], families[fit_mask])
        train_features = []
        train_targets = []
        for row in np.where(fit_mask)[0]:
            features = candidate_features(
                positive_embeddings[row],
                label_embeddings[fit_label_positions],
                fit_labels,
                family,
                ridge,
                np,
            )
            train_features.append(features)
            train_targets.extend(
                [int(candidate == target_labels[row]) for candidate in fit_labels]
            )
        train_features = np.concatenate(train_features)
        train_targets = np.asarray(train_targets)
        for regularization in (0.3, 1.0, 3.0):
            scorer = LogisticRegression(
                C=regularization,
                max_iter=300,
                class_weight="balanced",
                solver="liblinear",
                random_state=SEED,
            ).fit(train_features, train_targets)
            correct = []
            for row in np.where(validation_mask)[0]:
                candidates = [target_labels[row]] + fit_labels
                candidate_positions = [label_index[label] for label in candidates]
                features = candidate_features(
                    positive_embeddings[row],
                    label_embeddings[candidate_positions],
                    candidates,
                    family,
                    ridge,
                    np,
                )
                scores = scorer.predict_proba(features)[:, 1]
                correct.append(candidates[int(np.argmax(scores))] == target_labels[row])
            score = float(np.mean(correct)) if correct else 0.0
            if best is None or score > best[0]:
                best = (score, alpha, regularization)
    log(
        f"open-label sibling validation: family={best_family[0]:.4f}, "
        f"semantic_candidate={best[0]:.4f}"
    )
    holdout_family = LogisticRegression(
        C=best_family[1], max_iter=300, random_state=SEED
    ).fit(centroids[fit_mask], families[fit_mask])
    holdout_ridge = Ridge(alpha=best[1]).fit(
        centroids[fit_mask],
        label_embeddings[
            [label_index[label] for label in target_labels[fit_mask]]
        ],
    )
    holdout_features = []
    holdout_targets = []
    for row in np.where(fit_mask)[0]:
        holdout_features.append(
            candidate_features(
                positive_embeddings[row],
                label_embeddings[fit_label_positions],
                fit_labels,
                holdout_family,
                holdout_ridge,
                np,
            )
        )
        holdout_targets.extend(
            [int(label == target_labels[row]) for label in fit_labels]
        )
    holdout_scorer = LogisticRegression(
        C=best[2],
        max_iter=300,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED,
    ).fit(np.concatenate(holdout_features), np.asarray(holdout_targets))
    final_family = LogisticRegression(
        C=best_family[1], max_iter=300, random_state=SEED
    ).fit(centroids, families)
    final_ridge = Ridge(alpha=best[1]).fit(
        centroids,
        label_embeddings[[label_index[label] for label in target_labels]],
    )
    all_features = []
    all_targets = []
    for row in range(len(records)):
        all_features.append(
            candidate_features(
                positive_embeddings[row],
                label_embeddings,
                labels,
                final_family,
                final_ridge,
                np,
            )
        )
        all_targets.extend([int(label == target_labels[row]) for label in labels])
    final_scorer = LogisticRegression(
        C=best[2],
        max_iter=300,
        class_weight="balanced",
        solver="liblinear",
        random_state=SEED,
    ).fit(np.concatenate(all_features), np.asarray(all_targets))
    return (
        final_family,
        final_ridge,
        final_scorer,
        (holdout_family, holdout_ridge, holdout_scorer, fit_labels),
    )


def build_demonstration_bank(records, positive_embeddings, np):
    tools = sorted({record["target"]["target_tool"] for record in records})
    centroids = []
    examples = []
    for tool in tools:
        rows = [
            row
            for row, record in enumerate(records)
            if record["target"]["target_tool"] == tool
        ]
        vectors = np.asarray(positive_embeddings[rows], dtype=np.float64)
        centroid = normalize_rows(
            vectors.reshape(-1, vectors.shape[-1]).mean(axis=0, keepdims=True), np
        )[0]
        flat = vectors.reshape(-1, vectors.shape[-1])
        representative = int(np.argmax(flat @ centroid))
        source_row = rows[representative // vectors.shape[1]]
        source_position = representative % vectors.shape[1]
        centroids.append(centroid)
        examples.append(records[source_row]["positive"][source_position])
    return {
        "tools": tools,
        "centroids": np.asarray(centroids, dtype=np.float64),
        "examples": examples,
    }


def select_demonstrations(bank, episode_embeddings, excluded_tool, np, limit=4):
    centroid = normalize_rows(
        np.asarray(episode_embeddings, dtype=np.float64).mean(axis=0, keepdims=True), np
    )[0]
    similarities = bank["centroids"] @ centroid
    selected = []
    for position in np.argsort(similarities)[::-1]:
        tool = bank["tools"][int(position)]
        if tool == excluded_tool:
            continue
        example = bank["examples"][int(position)]
        selected.append(
            {
                "tool": tool,
                "locale": example["locale"],
                "request": example["request"],
            }
        )
        if len(selected) >= limit:
            break
    return selected


def naming_prompt(tokenizer, examples, observed_tools, demonstrations):
    requests = " || ".join(
        f"[{example['locale']}] {example['request']}" for example in examples
    )
    demonstration_text = " | ".join(
        f"[{item['locale']}] {item['request']} => {item['tool']}"
        for item in demonstrations
    )
    content = (
        "Requests: "
        + requests
        + ". Infer their canonical lowercase underscore intent token. The target is a "
        "new sibling, not one of the known tools. Related labeled sibling examples: "
        + demonstration_text
        + ". Other observed token examples: "
        + ", ".join(sorted(observed_tools))
        + ". Reply only with the exact intent token."
    )
    messages = [
        {
            "role": "system",
            "content": "You induce compositional virtual-assistant intent names.",
        },
        {"role": "user", "content": content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def attach_lora(model, torch, rank=4, alpha=8.0, dropout=0.05):
    class LoRALinear(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.scale = alpha / rank
            self.dropout = dropout
            self._eris_lora = True
            self.adapter_a = torch.nn.Linear(
                base.in_features, rank, bias=False, dtype=base.weight.dtype
            )
            self.adapter_b = torch.nn.Linear(
                rank, base.out_features, bias=False, dtype=base.weight.dtype
            )
            torch.nn.init.kaiming_uniform_(self.adapter_a.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.adapter_b.weight)

        def forward(self, values):
            adapted = torch.nn.functional.dropout(
                values, p=self.dropout, training=self.training
            )
            return self.base(values) + self.adapter_b(self.adapter_a(adapted)) * self.scale

    for parameter in model.parameters():
        parameter.requires_grad = False
    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".q_proj") or name.endswith(".v_proj")
    ]
    for name, module in replacements:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module))
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable == 0:
        raise RuntimeError("no attention projections found for adapter tuning")
    return trainable


def merge_lora(model, torch):
    replacements = [
        (name, module)
        for name, module in model.named_modules()
        if getattr(module, "_eris_lora", False)
    ]
    with torch.no_grad():
        for name, module in replacements:
            update = module.adapter_b.weight @ module.adapter_a.weight
            module.base.weight.add_(update, alpha=module.scale)
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, module.base)



def train_generator(
    records,
    known_tools,
    demonstration_bank,
    positive_embeddings,
    start_time,
    torch,
    np,
):
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_NAME, torch_dtype=torch.float32
    )
    model.to("cpu")
    trainable_count = attach_lora(model, torch)
    log(f"attached in-script low-rank adapters with {trainable_count} parameters")

    validation_tools = sibling_validation_tools(records, np)
    by_tool = defaultdict(list)
    for row, record in enumerate(records):
        by_tool[record["target"]["target_tool"]].append(row)
    rng = np.random.default_rng(SEED)
    selected = []
    validation_indices = []
    for tool in sorted(by_tool):
        rows = np.array(by_tool[tool], dtype=np.int64)
        rng.shuffle(rows)
        if tool in validation_tools:
            validation_indices.extend(rows[: min(5, len(rows))].tolist())
        else:
            selected.extend(rows[: min(8, len(rows))].tolist())
    rng.shuffle(selected)

    class NamingDataset(Dataset):
        def __len__(self):
            return len(selected)

        def __getitem__(self, position):
            record = records[selected[position]]
            target = record["target"]["target_tool"]
            observed = set(known_tools)
            observed.discard(target)
            demonstrations = select_demonstrations(
                demonstration_bank, positive_embeddings[selected[position]], target, np
            )
            prompt = naming_prompt(
                tokenizer, record["positive"], observed, demonstrations
            )
            prompt_ids = tokenizer(
                prompt, truncation=True, max_length=424, add_special_tokens=True
            )["input_ids"]
            target_ids = tokenizer(
                target + tokenizer.eos_token,
                truncation=True,
                max_length=24,
                add_special_tokens=False,
            )["input_ids"]
            return {
                "input_ids": prompt_ids + target_ids,
                "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
                "labels": [-100] * len(prompt_ids) + target_ids,
            }

    def collate(batch):
        length = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full(
            (len(batch), length), tokenizer.pad_token_id, dtype=torch.long
        )
        attention = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        for row, item in enumerate(batch):
            width = len(item["input_ids"])
            input_ids[row, :width] = torch.tensor(item["input_ids"])
            attention[row, :width] = 1
            labels[row, :width] = torch.tensor(item["labels"])
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}

    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        NamingDataset(),
        batch_size=4,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    adapter_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(trainable_parameters, lr=0.0004)
    validation_prompts = []
    validation_truth = []
    for row in validation_indices:
        target = records[row]["target"]["target_tool"]
        observed = set(known_tools)
        observed.discard(target)
        demonstrations = select_demonstrations(
            demonstration_bank, positive_embeddings[row], target, np
        )
        validation_prompts.append(
            naming_prompt(
                tokenizer, records[row]["positive"], observed, demonstrations
            )
        )
        validation_truth.append(target)

    best = None
    for epoch in range(1, 2):
        tokenizer.padding_side = "right"
        model.config.use_cache = False
        model.train()
        epoch_losses = []
        for batch in loader:
            if time.monotonic() - start_time >= TRAINING_CUTOFF_SECONDS:
                log("wall-clock guard: ending generator tuning before next batch")
                break
            loss = model(**batch).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        if not epoch_losses:
            break
        model.config.use_cache = True
        model.eval()
        beams = generate_beams(
            validation_prompts, tokenizer, model, start_time, torch
        )
        top_correct = []
        beam_correct = []
        for truth, row_beams in zip(validation_truth, beams):
            normalized = [normalize_tool(value) for value in row_beams]
            top_correct.append(bool(normalized) and normalized[0] == truth)
            beam_correct.append(truth in normalized)
        top_accuracy = float(np.mean(top_correct)) if top_correct else 0.0
        beam_recall = float(np.mean(beam_correct)) if beam_correct else 0.0
        mean_loss = float(np.mean(epoch_losses))
        log(
            f"generator sibling validation epoch={epoch} "
            f"top1={top_accuracy:.4f} beam_recall={beam_recall:.4f} "
            f"train_loss={mean_loss:.4f}"
        )
        criterion = (beam_recall, top_accuracy, -mean_loss)
        if best is None or criterion > best[0]:
            state = model.state_dict()
            best = (
                criterion,
                epoch,
                {
                    name: state[name].detach().clone()
                    for name in adapter_names
                },
                [list(row_beams) for row_beams in beams],
            )
    del optimizer, loader
    gc.collect()
    if best is None:
        log("warning: generator received no training batches")
    else:
        model.load_state_dict(best[2], strict=False)
        log(
            f"selected generator epoch={best[1]} on sibling-tool validation; "
            f"tools={sorted(validation_tools)}"
        )
    merge_lora(model, torch)
    model.config.use_cache = True
    model.eval()
    validation = {
        "indices": validation_indices if best is not None else [],
        "beams": best[3] if best is not None else [],
    }
    return tokenizer, model, validation


def normalize_tool(value):
    value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:64]


def generate_beams(prompts, tokenizer, model, start_time, torch):
    tokenizer.padding_side = "left"
    results = []
    for start in range(0, len(prompts), 6):
        elapsed = time.monotonic() - start_time
        beams = 3 if elapsed < 4500.0 else 2
        if elapsed >= INFERENCE_GUARD_SECONDS:
            log("warning: inference guard reached; remaining generator outputs are empty")
            results.extend([[] for _ in prompts[start:]])
            continue
        batch_prompts = prompts[start : start + 6]
        try:
            batch = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=424,
                return_tensors="pt",
            )
            with torch.inference_mode():
                generated = model.generate(
                    **batch,
                    max_new_tokens=16,
                    do_sample=False,
                    num_beams=beams,
                    num_return_sequences=beams,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            width = batch["input_ids"].shape[1]
            decoded = [
                tokenizer.decode(sequence[width:], skip_special_tokens=True).strip()
                for sequence in generated
            ]
            for row in range(len(batch_prompts)):
                results.append(decoded[row * beams : (row + 1) * beams])
        except Exception as exc:
            log(f"warning: generator batch failed: {exc}")
            results.extend([[] for _ in batch_prompts])
    return results


def make_candidates(beams, family_choices, known_tools):
    candidates = []

    def add(value):
        token = normalize_tool(value)
        if (
            token
            and TOOL_PATTERN.fullmatch(token)
            and token not in known_tools
            and token not in candidates
        ):
            candidates.append(token)

    for raw in beams:
        token = normalize_tool(raw)
        parts = [part for part in token.split("_") if part]
        add(token)
        if not parts:
            continue
        for family in family_choices:
            add(family + "_" + "_".join(parts[1:]))
            add(family + "_" + parts[0])
            add(family + "_" + parts[-1])
            for part in parts:
                add(family + "_" + part)
    if not candidates:
        for raw in beams:
            token = normalize_tool(raw)
            if token and TOOL_PATTERN.fullmatch(token) and token not in candidates:
                candidates.append(token)
    return candidates[:120]


def tune_generation_rank(
    records,
    validation,
    positive_embeddings,
    holdout_models,
    encoder_tokenizer,
    encoder_model,
    torch,
    np,
):
    family, ridge, scorer, known_labels = holdout_models
    candidate_lists = []
    for source_row, row_beams in zip(validation["indices"], validation["beams"]):
        centroid = normalize_rows(
            positive_embeddings[source_row].mean(axis=0, keepdims=True), np
        )[0]
        probabilities = family.predict_proba(centroid.reshape(1, -1))[0]
        families = family.classes_[
            np.argsort(probabilities)[-3:][::-1]
        ].tolist()
        candidate_lists.append(
            make_candidates(row_beams, families, set(known_labels))
        )
    flat = [candidate for candidates in candidate_lists for candidate in candidates]
    if not flat:
        log("warning: no generated validation candidates; semantic ranking retained")
        return 0.0
    embeddings = encode_texts(
        [candidate.replace("_", " ") for candidate in flat],
        encoder_tokenizer,
        encoder_model,
        torch,
        batch_size=192,
        max_length=24,
    )
    cursor = 0
    rows = []
    recalled = []
    for source_row, candidates in zip(validation["indices"], candidate_lists):
        width = len(candidates)
        candidate_embeddings = embeddings[cursor : cursor + width]
        cursor += width
        truth = records[source_row]["target"]["target_tool"]
        recalled.append(truth in candidates)
        if not width:
            rows.append((truth, candidates, np.empty(0), np.empty(0)))
            continue
        features = candidate_features(
            positive_embeddings[source_row],
            candidate_embeddings,
            candidates,
            family,
            ridge,
            np,
        )
        probabilities = np.clip(scorer.predict_proba(features)[:, 1], 1e-6, 1.0 - 1e-6)
        semantic = np.log(probabilities) - np.log1p(-probabilities)
        generator_order = -np.log1p(np.arange(width, dtype=np.float64))
        rows.append((truth, candidates, semantic, generator_order))
    best = None
    for weight in np.linspace(-1.0, 4.0, 21):
        correct = []
        for truth, candidates, semantic, generator_order in rows:
            if not candidates:
                correct.append(False)
                continue
            scores = semantic + float(weight) * generator_order
            correct.append(candidates[int(np.argmax(scores))] == truth)
        accuracy = float(np.mean(correct)) if correct else 0.0
        criterion = (accuracy, -abs(float(weight)))
        if best is None or criterion > best[0]:
            best = (criterion, float(weight))
    log(
        f"generation rank validation: candidate_recall={np.mean(recalled):.4f} "
        f"exact={best[0][0]:.4f} order_weight={best[1]:.2f}"
    )
    return best[1]


def audit_submission(path, test_records):
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["contract_id", "induced_contract"]:
                return False, "wrong columns or order"
            rows = list(reader)
        if len(rows) != len(test_records):
            return False, "wrong row count"
        expected = [record["id"] for record in test_records]
        actual = [row["contract_id"] for row in rows]
        if actual != expected or len(set(actual)) != len(actual):
            return False, "missing, reordered, or duplicate IDs"
        for row, record in zip(rows, test_records):
            registry = {name for name, _ in record["registry"]}
            parsed = json.loads(row["induced_contract"])
            if list(parsed) != CONTRACT_KEYS:
                return False, f"wrong contract keys for {record['id']}"
            route = parsed["routing_rule"]
            if list(route) != ["argument", "operator"]:
                return False, f"wrong route keys for {record['id']}"
            if route["argument"] not in registry or route["operator"] not in {
                "required",
                "forbidden",
            }:
                return False, f"invalid route for {record['id']}"
            required = parsed["required_arguments"]
            optional = parsed["optional_arguments"]
            if (
                required != sorted(set(required))
                or optional != sorted(set(optional))
                or set(required) & set(optional)
                or not set(required + optional).issubset(registry)
            ):
                return False, f"invalid interface for {record['id']}"
            if not TOOL_PATTERN.fullmatch(parsed["target_tool"]) or not TOOL_PATTERN.fullmatch(
                parsed["peer_tool"]
            ):
                return False, f"invalid tool token for {record['id']}"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def main():
    start_time = time.monotonic()
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required input: " + ", ".join(missing))

    import numpy as np
    import pandas as pd
    import torch

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(10, max(1, os.cpu_count() or 1)))
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    test_frame = pd.read_csv(test_path, dtype={"contract_id": str})
    test_records = parse_records(test_frame, labelled=False)
    test_ids = [record["id"] for record in test_records]
    placeholders = [
        fallback_contract([name for name, _ in record["registry"]])
        for record in test_records
    ]
    write_submission(submission_out, test_ids, placeholders)
    log(f"wrote early schema-valid placeholder with {len(test_records)} rows")

    try:
        train_frame = pd.read_csv(train_path, dtype={"contract_id": str})
        train_records = parse_records(train_frame, labelled=True)
        if len(train_records) < 20:
            log("warning: too few labelled records; placeholder retained")
            return
        argument_names = [name for name, _ in train_records[0]["registry"]]
        argument_descriptions = [description for _, description in train_records[0]["registry"]]
        if any(
            [name for name, _ in record["registry"]] != argument_names
            for record in train_records
        ):
            log("warning: training registries differ; first registry defines learned heads")

        encoder_tokenizer, encoder_model = load_encoder(torch)
        train_positive, train_contrast = episode_embeddings(
            train_records, encoder_tokenizer, encoder_model, torch
        )
        test_positive, test_contrast = episode_embeddings(
            test_records, encoder_tokenizer, encoder_model, torch
        )
        argument_matrix = encode_texts(
            argument_descriptions, encoder_tokenizer, encoder_model, torch
        )
        argument_tensor = torch.tensor(argument_matrix, dtype=torch.float32)

        contract_model, calibration = train_contract_model(
            train_records,
            train_positive,
            train_contrast,
            argument_tensor,
            argument_names,
            torch,
            np,
        )
        with torch.inference_mode():
            structured_values = contract_model(
                torch.tensor(test_positive, dtype=torch.float32),
                torch.tensor(test_contrast, dtype=torch.float32),
            )
        structured = decode_contract_outputs(
            structured_values[0].cpu().numpy(),
            structured_values[1].cpu().numpy(),
            structured_values[2].cpu().numpy(),
            argument_names,
            calibration[0],
            calibration[1],
            calibration[2],
            np,
        )

        lda, grouping_weight = fit_contrast_grouper(
            train_records, train_positive, train_contrast, np
        )
        test_peer_indices = [
            select_peer_examples(
                test_positive[row], test_contrast[row], lda, grouping_weight, np
            )
            for row in range(len(test_records))
        ]
        test_peer_embeddings = [
            test_contrast[row, test_peer_indices[row]] for row in range(len(test_records))
        ]

        labels = sorted(
            {
                value
                for record in train_records
                for value in (
                    record["target"]["target_tool"],
                    record["target"]["peer_tool"],
                )
            }
        )
        label_embeddings = encode_texts(
            [label.replace("_", " ") for label in labels],
            encoder_tokenizer,
            encoder_model,
            torch,
            batch_size=96,
            max_length=24,
        )
        validation_tools = sibling_validation_tools(train_records, np)
        (
            family_classifier,
            ridge,
            candidate_scorer,
            holdout_label_models,
        ) = fit_open_label_models(
            train_records,
            train_positive,
            label_embeddings,
            labels,
            validation_tools,
            np,
        )

        generator_tokenizer = None
        generator_model = None
        target_beams = [[] for _ in test_records]
        peer_beams = [[] for _ in test_records]
        generation_order_weight = 0.0
        demonstration_bank = build_demonstration_bank(
            train_records, train_positive, np
        )
        try:
            (
                generator_tokenizer,
                generator_model,
                generator_validation,
            ) = train_generator(
                train_records,
                set(labels),
                demonstration_bank,
                train_positive,
                start_time,
                torch,
                np,
            )
            generation_order_weight = tune_generation_rank(
                train_records,
                generator_validation,
                train_positive,
                holdout_label_models,
                encoder_tokenizer,
                encoder_model,
                torch,
                np,
            )
            target_prompts = [
                naming_prompt(
                    generator_tokenizer,
                    record["positive"],
                    set(labels),
                    select_demonstrations(
                        demonstration_bank, test_positive[row], None, np
                    ),
                )
                for row, record in enumerate(test_records)
            ]
            peer_prompts = [
                naming_prompt(
                    generator_tokenizer,
                    [record["contrast"][position] for position in test_peer_indices[row]],
                    set(labels),
                    select_demonstrations(
                        demonstration_bank, test_peer_embeddings[row], None, np
                    ),
                )
                for row, record in enumerate(test_records)
            ]
            all_beams = generate_beams(
                target_prompts + peer_prompts,
                generator_tokenizer,
                generator_model,
                start_time,
                torch,
            )
            target_beams = all_beams[: len(test_records)]
            peer_beams = all_beams[len(test_records) :]
        except Exception as exc:
            log(f"warning: open-vocabulary generator unavailable: {exc}")
        finally:
            if generator_model is not None:
                generator_model.eval()

        target_candidate_lists = []
        peer_candidate_lists = []
        for row in range(len(test_records)):
            target_centroid = normalize_rows(
                test_positive[row].mean(axis=0, keepdims=True), np
            )[0]
            peer_centroid = normalize_rows(
                test_peer_embeddings[row].mean(axis=0, keepdims=True), np
            )[0]
            target_probabilities = family_classifier.predict_proba(
                target_centroid.reshape(1, -1)
            )[0]
            peer_probabilities = family_classifier.predict_proba(
                peer_centroid.reshape(1, -1)
            )[0]
            target_families = family_classifier.classes_[
                np.argsort(target_probabilities)[-3:][::-1]
            ].tolist()
            peer_families = family_classifier.classes_[
                np.argsort(peer_probabilities)[-3:][::-1]
            ].tolist()
            target_candidate_lists.append(
                make_candidates(target_beams[row], target_families, set(labels))
            )
            peer_candidate_lists.append(
                make_candidates(peer_beams[row], peer_families, set(labels))
            )

        flat_candidates = [
            candidate
            for candidate_list in target_candidate_lists + peer_candidate_lists
            for candidate in candidate_list
        ]
        flat_embeddings = encode_texts(
            [candidate.replace("_", " ") for candidate in flat_candidates],
            encoder_tokenizer,
            encoder_model,
            torch,
            batch_size=192,
            max_length=24,
        )
        cursor = 0
        target_rankings = []
        peer_rankings = []
        for row, candidate_list in enumerate(target_candidate_lists):
            width = len(candidate_list)
            embeddings = flat_embeddings[cursor : cursor + width]
            cursor += width
            ranked = []
            if width:
                features = candidate_features(
                    test_positive[row],
                    embeddings,
                    candidate_list,
                    family_classifier,
                    ridge,
                    np,
                )
                probabilities = np.clip(
                    candidate_scorer.predict_proba(features)[:, 1],
                    1e-6,
                    1.0 - 1e-6,
                )
                scores = np.log(probabilities) - np.log1p(-probabilities)
                scores -= generation_order_weight * np.log1p(
                    np.arange(width, dtype=np.float64)
                )
                ranking = np.argsort(scores)[::-1]
                ranked.extend(candidate_list[position] for position in ranking)
            fallback_features = candidate_features(
                test_positive[row],
                label_embeddings,
                labels,
                family_classifier,
                ridge,
                np,
            )
            fallback_scores = candidate_scorer.predict_proba(fallback_features)[:, 1]
            for position in np.argsort(fallback_scores)[::-1]:
                if labels[position] not in ranked:
                    ranked.append(labels[position])
            target_rankings.append(ranked)
        for row, candidate_list in enumerate(peer_candidate_lists):
            width = len(candidate_list)
            embeddings = flat_embeddings[cursor : cursor + width]
            cursor += width
            ranked = []
            if width:
                features = candidate_features(
                    test_peer_embeddings[row],
                    embeddings,
                    candidate_list,
                    family_classifier,
                    ridge,
                    np,
                )
                probabilities = np.clip(
                    candidate_scorer.predict_proba(features)[:, 1],
                    1e-6,
                    1.0 - 1e-6,
                )
                scores = np.log(probabilities) - np.log1p(-probabilities)
                scores -= generation_order_weight * np.log1p(
                    np.arange(width, dtype=np.float64)
                )
                ranking = np.argsort(scores)[::-1]
                ranked.extend(candidate_list[position] for position in ranking)
            fallback_features = candidate_features(
                test_peer_embeddings[row],
                label_embeddings,
                labels,
                family_classifier,
                ridge,
                np,
            )
            fallback_scores = candidate_scorer.predict_proba(fallback_features)[:, 1]
            for position in np.argsort(fallback_scores)[::-1]:
                if labels[position] not in ranked:
                    ranked.append(labels[position])
            peer_rankings.append(ranked)

        predictions = []
        for row, record in enumerate(test_records):
            try:
                row_registry = {name for name, _ in record["registry"]}
                route = structured[row]
                if route["argument"] not in row_registry:
                    route["argument"] = next(iter(row_registry))
                required = sorted(set(route["required"]) & row_registry)
                optional = sorted((set(route["optional"]) & row_registry) - set(required))
                if route["operator"] == "required":
                    required = sorted(set(required) | {route["argument"]})
                    optional = sorted(set(optional) - {route["argument"]})
                else:
                    required = sorted(set(required) - {route["argument"]})
                    optional = sorted(set(optional) - {route["argument"]})
                target_tool = target_rankings[row][0] if target_rankings[row] else ""
                peer_tool = ""
                for candidate in peer_rankings[row]:
                    if candidate != target_tool:
                        peer_tool = candidate
                        break
                prediction = {
                    "target_tool": target_tool,
                    "routing_rule": {
                        "argument": route["argument"],
                        "operator": route["operator"],
                    },
                    "required_arguments": required,
                    "optional_arguments": optional,
                    "peer_tool": peer_tool,
                }
                if not TOOL_PATTERN.fullmatch(target_tool) or not TOOL_PATTERN.fullmatch(peer_tool):
                    raise ValueError("invalid generated tool token")
                predictions.append(prediction)
            except Exception as exc:
                log(f"warning: row {record['id']} used valid fallback: {exc}")
                predictions.append(
                    fallback_contract([name for name, _ in record["registry"]])
                )

        write_submission(submission_out, test_ids, predictions)
        valid, reason = audit_submission(submission_out, test_records)
        if not valid:
            log(f"warning: final audit failed ({reason}); rewriting valid placeholders")
            write_submission(submission_out, test_ids, placeholders)
            valid, reason = audit_submission(submission_out, test_records)
        log(
            f"submission audit={valid} ({reason}); rows={len(test_records)}; "
            f"elapsed={time.monotonic() - start_time:.1f}s"
        )
    except Exception as exc:
        log(f"warning: solver stopped after protected placeholder was written: {exc}")


if __name__ == "__main__":
    main()
