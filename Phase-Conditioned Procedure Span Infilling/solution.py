#!/usr/bin/env python3
"""Cross-fitted CPU span infiller for the Project Eris challenge."""

import csv
import gc
import json
import math
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

SEED = 2027
BACKBONE = "bert-base-uncased"
TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
SPAN_ALPHAS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)
N_FOLDS = 3
MAX_EPOCHS = 8
FIRST_EVAL_EPOCH = 4
TRAINING_DEADLINE_SECONDS = 3000.0


def log(message):
    print(f"[eris] {message}", flush=True)


def safe_count(value, default=1):
    try:
        return max(0, int(value))
    except Exception:
        return default


def valid_tokens(tokens, required_count):
    return (
        isinstance(tokens, list)
        and len(tokens) == required_count
        and all(isinstance(token, str) and TOKEN_PATTERN.fullmatch(token) for token in tokens)
    )


def prediction_json(tokens):
    return json.dumps({"missing_tokens": tokens}, ensure_ascii=False, separators=(",", ":"))


def write_predictions(path, ids, predictions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer_json"])
        writer.writeheader()
        for row_id, tokens in zip(ids, predictions):
            writer.writerow({"id": row_id, "answer_json": prediction_json(tokens)})


def parse_records(frame, labelled):
    records = []
    skipped = 0
    for raw in frame.to_dict(orient="records"):
        row_id = str(raw.get("id", ""))
        count = safe_count(raw.get("missing_token_count"), 1)
        try:
            context = json.loads(raw["flow_context_json"])
            context["target_step"]
            answer = None
            if labelled:
                answer = json.loads(raw["answer_json"])["missing_tokens"]
                if not valid_tokens(answer, count):
                    raise ValueError("invalid labelled answer")
            records.append(
                {"id": row_id, "count": count, "context": context, "answer": answer}
            )
        except Exception as exc:
            skipped += 1
            if labelled:
                log(f"warning: skipped malformed training row {row_id}: {exc}")
            else:
                log(f"warning: malformed test row {row_id}; fallback retained")
                records.append(
                    {"id": row_id, "count": count, "context": None, "answer": None}
                )
    if skipped:
        log(f"parsed rows with {skipped} malformed record(s)")
    return records


def row_score(truth, prediction):
    n = len(truth)
    if n == 0:
        return float(prediction == truth)
    exact = float(truth == prediction)
    position = sum(a == b for a, b in zip(truth, prediction)) / n
    common = sum((Counter(truth) & Counter(prediction)).values())
    precision = common / len(prediction) if prediction else 0.0
    recall = common / n
    token_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    previous = [0] * (len(prediction) + 1)
    for true_token in truth:
        current = [0]
        for column, predicted_token in enumerate(prediction, 1):
            current.append(
                previous[column - 1] + 1
                if true_token == predicted_token
                else max(previous[column], current[-1])
            )
        previous = current
    token_lcs = previous[-1] / n
    return 0.55 * exact + 0.25 * position + 0.10 * token_f1 + 0.10 * token_lcs


def metric(truths, predictions):
    if not truths:
        return 0.0
    return sum(row_score(list(t), list(p)) for t, p in zip(truths, predictions)) / len(
        truths
    )


def bert_text(context):
    step = context["target_step"]
    target_index = int(step["target_gap_index"])
    parts = [
        "phase",
        str(step["phase"]),
        "flow",
        "position",
        str(step["flow_position"]),
        "target",
        "step",
    ]
    for segment in step["masked_segments"]:
        if segment.get("kind") == "visible":
            parts.extend(segment.get("tokens", []))
        elif int(segment.get("gap_index", -1)) == target_index:
            parts.extend(
                ["[MASK]"]
                * int(segment.get("missing_token_count", step["missing_token_count"]))
            )
        else:
            parts.extend(
                ["[unused1]", "gap", str(segment.get("missing_token_count", 0))]
            )
    parts.extend(["[SEP]", "phase", "sequence"])
    for phase in context.get("phase_sequence", []):
        parts.extend([str(phase["flow_position"]), str(phase["phase"])])
    parts.extend(["[SEP]", "prerequisite"])
    parts.extend(context.get("prerequisite_tokens", []))
    parts.extend(["[SEP]", "consequence"])
    parts.extend(context.get("consequence_tokens", []))
    parts.extend(["[SEP]", "procedure", "context"])
    parts.extend(context.get("pattern_context_tokens", []))
    return " ".join(parts)


def span_text(context, count, slot=0):
    if context is None:
        return f"slot_{slot} count_{count} malformed_context"
    step = context["target_step"]
    target_index = int(step["target_gap_index"])
    before = []
    after = []
    all_step = []
    target_seen = False
    for segment in step["masked_segments"]:
        if segment.get("kind") == "visible":
            tokens = segment.get("tokens", [])
            all_step.extend(tokens)
            (after if target_seen else before).extend(tokens)
        elif int(segment.get("gap_index", -1)) == target_index:
            target_seen = True
            all_step.append("targetgap")
        else:
            all_step.append("othergap")
            (after if target_seen else before).append("othergap")
    features = [
        f"slot_{slot}",
        f"count_{count}",
        f"phase_{step['phase']}",
        f"flowpos_{step['flow_position']}",
        f"gapindex_{target_index}",
    ]
    features.extend(
        f"left_{distance}_{token}"
        for distance, token in enumerate(reversed(before[-8:]), 1)
    )
    features.extend(
        f"right_{distance}_{token}"
        for distance, token in enumerate(after[:8], 1)
    )
    features.extend(f"step_{token}" for token in all_step)
    features.extend(all_step)
    for phase in context.get("phase_sequence", []):
        features.append(f"seq_{phase['flow_position']}_{phase['phase']}")
    features.extend(f"pre_{token}" for token in context.get("prerequisite_tokens", []))
    features.extend(f"con_{token}" for token in context.get("consequence_tokens", []))
    features.extend(f"ctx_{token}" for token in context.get("pattern_context_tokens", []))
    return " ".join(features)


def collect_normalized_tokens(records, indices):
    tokens = set()
    for index in indices:
        record = records[int(index)]
        tokens.update(record["answer"] or [])
        context = record["context"]
        if context is None:
            continue
        tokens.update(context.get("pattern_context_tokens", []))
        tokens.update(context.get("prerequisite_tokens", []))
        tokens.update(context.get("consequence_tokens", []))
        for segment in context["target_step"]["masked_segments"]:
            tokens.update(segment.get("tokens", []))
    return {
        token
        for token in tokens
        if isinstance(token, str) and TOKEN_PATTERN.fullmatch(token)
    }


def procedure_groups(records, np):
    keys = [
        json.dumps(
            record["context"].get("pattern_context_tokens", []),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for record in records
    ]
    mapping = {key: index for index, key in enumerate(sorted(set(keys)))}
    return np.asarray([mapping[key] for key in keys], dtype=np.int64)


class RowDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def build_tokenizer(records, indices):
    from transformers import AddedToken, AutoTokenizer

    normalized = collect_normalized_tokens(records, indices)
    tokenizer = AutoTokenizer.from_pretrained(
        BACKBONE, local_files_only=True, use_fast=True
    )
    base_piece_ids = {
        token: tokenizer.encode(token, add_special_tokens=False) for token in normalized
    }
    new_tokens = sorted(
        token for token in normalized if token not in tokenizer.get_vocab()
    )
    tokenizer.add_tokens(
        [AddedToken(token, single_word=True, normalized=True) for token in new_tokens]
    )
    for token in normalized:
        if len(tokenizer.encode(token, add_special_tokens=False)) != 1:
            raise ValueError(f"could not make normalized token atomic: {token}")
    answer_tokens = {
        token for index in indices for token in (records[int(index)]["answer"] or [])
    }
    return tokenizer, normalized, answer_tokens, new_tokens, base_piece_ids


def initialize_mask_model(tokenizer, new_tokens, base_piece_ids, torch):
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(BACKBONE, local_files_only=True)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight
        for token in new_tokens:
            pieces = torch.tensor(base_piece_ids[token], dtype=torch.long)
            embeddings[tokenizer.convert_tokens_to_ids(token)].copy_(
                embeddings[pieces].mean(dim=0)
            )
    model.tie_weights()
    model.to(torch.device("cpu"))
    return model


def encoded_rows(tokenizer, records, indices, labelled):
    rows = []
    mask_id = tokenizer.mask_token_id
    for index in indices:
        record = records[int(index)]
        encoded = tokenizer(
            bert_text(record["context"]),
            max_length=320,
            truncation=True,
            add_special_tokens=True,
        )
        mask_positions = [
            position
            for position, token_id in enumerate(encoded["input_ids"])
            if token_id == mask_id
        ]
        if len(mask_positions) != record["count"]:
            raise ValueError(f"mask count mismatch in {record['id']}")
        if labelled:
            labels = [-100] * len(encoded["input_ids"])
            for position, token in zip(mask_positions, record["answer"]):
                labels[position] = tokenizer.convert_tokens_to_ids(token)
            encoded["labels"] = labels
        rows.append(encoded)
    return rows


def candidate_scopes(tokenizer, normalized, answer_tokens, torch):
    def token_ids(tokens):
        return torch.tensor(
            sorted({tokenizer.convert_tokens_to_ids(token) for token in tokens}),
            dtype=torch.long,
        )

    return {
        "answer": token_ids(answer_tokens),
        "train_tokens": token_ids(normalized),
    }


def neural_log_probabilities(model, tokenizer, rows, scopes, torch, batch_size=8):
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    output = {name: [] for name in scopes}
    loader = DataLoader(
        RowDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
    )
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            active = batch["input_ids"].eq(tokenizer.mask_token_id)
            hidden = model.bert(**batch, return_dict=True).last_hidden_state
            for row_index in range(hidden.shape[0]):
                logits = model.cls(hidden[row_index, active[row_index]])
                for name, candidate_ids in scopes.items():
                    output[name].append(
                        logits[:, candidate_ids]
                        .log_softmax(dim=1)
                        .cpu()
                        .numpy()
                        .astype("float32")
                    )
    return output


def tokens_from_log_rows(rows, candidate_tokens):
    return [
        [candidate_tokens[int(index)] for index in row.argmax(axis=1)]
        for row in rows
    ]


def train_neural_fold(
    records,
    train_indices,
    validation_indices,
    test_records,
    fold_number,
    torch,
    np,
    start_time,
):
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader
    from transformers import (
        DataCollatorForTokenClassification,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(SEED)
    tokenizer, normalized, answer_tokens, new_tokens, base_piece_ids = build_tokenizer(
        records, train_indices
    )
    model = initialize_mask_model(tokenizer, new_tokens, base_piece_ids, torch)
    train_rows = encoded_rows(tokenizer, records, train_indices, True)
    validation_rows = encoded_rows(
        tokenizer, records, validation_indices, labelled=False
    )
    valid_test_indices = [
        index for index, record in enumerate(test_records) if record["context"] is not None
    ]
    test_rows = encoded_rows(tokenizer, test_records, valid_test_indices, labelled=False)
    loader = DataLoader(
        RowDataset(train_rows),
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=DataCollatorForTokenClassification(
            tokenizer, label_pad_token_id=-100, return_tensors="pt"
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, len(loader) // 5),
        num_training_steps=len(loader) * MAX_EPOCHS,
    )
    scopes = candidate_scopes(tokenizer, normalized, answer_tokens, torch)
    truths = [records[int(index)]["answer"] for index in validation_indices]
    best = None
    best_state = None
    epoch_durations = []
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.monotonic()
        model.train()
        running_loss = 0.0
        for batch in loader:
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            hidden = model.bert(**batch, return_dict=True).last_hidden_state
            active = labels.ne(-100)
            logits = model.cls(hidden[active])
            loss = functional.cross_entropy(logits, labels[active])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach())
        epoch_scores = {}
        if epoch >= FIRST_EVAL_EPOCH:
            scope_rows = neural_log_probabilities(
                model, tokenizer, validation_rows, scopes, torch
            )
            epoch_winner = None
            for scope_name, rows in scope_rows.items():
                candidate_tokens = tokenizer.convert_ids_to_tokens(
                    scopes[scope_name].tolist()
                )
                predictions = tokens_from_log_rows(rows, candidate_tokens)
                score = metric(truths, predictions)
                epoch_scores[scope_name] = score
                if epoch_winner is None or score > epoch_winner["score"]:
                    epoch_winner = {
                        "score": score,
                        "scope": scope_name,
                        "candidate_tokens": list(candidate_tokens),
                        "validation_rows": [row.copy() for row in rows],
                    }
            if best is None or epoch_winner["score"] > best["score"] + 1e-12:
                best = {
                    **epoch_winner,
                    "epoch": epoch,
                    "validation_indices": np.asarray(validation_indices).copy(),
                }
                del best_state
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
        duration = time.monotonic() - epoch_start
        epoch_durations.append(duration)
        score_text = ", ".join(
            f"{name}={score:.6f}" for name, score in epoch_scores.items()
        )
        log(
            f"fold {fold_number + 1}/{N_FOLDS} epoch {epoch}: "
            f"loss={running_loss / len(loader):.5f}"
            + (f", {score_text}" if score_text else "")
        )
        remaining_folds = N_FOLDS - fold_number - 1
        average_epoch = sum(epoch_durations[-2:]) / min(2, len(epoch_durations))
        if (
            best is not None
            and epoch < MAX_EPOCHS
            and time.monotonic() - start_time
            + remaining_folds * FIRST_EVAL_EPOCH * average_epoch
            + 240.0
            >= TRAINING_DEADLINE_SECONDS
        ):
            log("wall-clock guard: ending this fold before the next epoch")
            break
    if best is None or best_state is None:
        raise RuntimeError("neural fold did not reach an evaluation epoch")
    model.load_state_dict(best_state)
    chosen_scope = {best["scope"]: scopes[best["scope"]]}
    valid_test_rows = neural_log_probabilities(
        model, tokenizer, test_rows, chosen_scope, torch
    )[best["scope"]]
    test_log_rows = [None] * len(test_records)
    for index, row in zip(valid_test_indices, valid_test_rows):
        test_log_rows[index] = row
    best["test_rows"] = test_log_rows
    best["epoch_seconds"] = sum(epoch_durations) / len(epoch_durations)
    log(
        f"fold {fold_number + 1} selected epoch={best['epoch']}, "
        f"scope={best['scope']}, score={best['score']:.6f}"
    )
    del model, optimizer, scheduler, loader, best_state
    gc.collect()
    return best


def neural_oof_predictions(records, artifacts):
    predictions = [None] * len(records)
    for artifact in artifacts:
        fold_predictions = tokens_from_log_rows(
            artifact["validation_rows"], artifact["candidate_tokens"]
        )
        for index, prediction in zip(
            artifact["validation_indices"], fold_predictions
        ):
            predictions[int(index)] = prediction
    return predictions


def neural_test_ensemble(test_records, artifacts, np):
    candidate_tokens = sorted(
        {token for artifact in artifacts for token in artifact["candidate_tokens"]}
    )
    candidate_index = {token: index for index, token in enumerate(candidate_tokens)}
    mappings = [
        [candidate_index[token] for token in artifact["candidate_tokens"]]
        for artifact in artifacts
    ]
    predictions = []
    for row_index, record in enumerate(test_records):
        if record["context"] is None or record["count"] == 0:
            predictions.append([] if record["count"] == 0 else ["unknown"] * record["count"])
            continue
        combined = np.zeros(
            (record["count"], len(candidate_tokens)), dtype=np.float32
        )
        used = 0
        for artifact, mapping in zip(artifacts, mappings):
            row = artifact["test_rows"][row_index]
            if row is None:
                continue
            aligned = np.full_like(combined, -30.0)
            aligned[:, mapping] = row
            combined += aligned
            used += 1
        if used == 0:
            predictions.append(["unknown"] * record["count"])
        else:
            predictions.append(
                [
                    candidate_tokens[int(index)]
                    for index in (combined / used).argmax(axis=1)
                ]
            )
    return predictions


def span_feature_corpus(records, indices):
    texts = []
    for index in indices:
        record = records[int(index)]
        for slot in range(record["count"]):
            texts.append(span_text(record["context"], record["count"], slot))
    return texts


def span_details(log_probabilities, classes, counts, class_count, np):
    class_tokens = [value.split("\x1f") for value in classes]
    by_count = defaultdict(list)
    for class_index, tokens in enumerate(class_tokens):
        by_count[len(tokens)].append(class_index)
    predictions = []
    normalized_confidences = []
    for row, count in zip(log_probabilities, counts):
        allowed = np.asarray(by_count.get(int(count), []), dtype=np.int64)
        if allowed.size == 0:
            predictions.append(None)
            normalized_confidences.append(float("-inf"))
            continue
        values = row[allowed]
        chosen = int(allowed[int(values.argmax())])
        predictions.append(class_tokens[chosen])
        normalized_confidences.append(
            float(values.max()) + math.log(max(1, class_count))
        )
    return predictions, normalized_confidences


def fit_span_family(
    records,
    train_indices,
    prediction_records,
    prediction_indices,
    alphas,
    np,
):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w+\b",
    )
    vectorizer.fit(span_feature_corpus(records, train_indices))
    train_texts = [
        span_text(records[int(index)]["context"], records[int(index)]["count"])
        for index in train_indices
    ]
    prediction_texts = [
        span_text(
            prediction_records[int(index)]["context"],
            prediction_records[int(index)]["count"],
        )
        for index in prediction_indices
    ]
    train_matrix = vectorizer.transform(train_texts)
    prediction_matrix = vectorizer.transform(prediction_texts)
    labels = np.asarray(
        ["\x1f".join(records[int(index)]["answer"]) for index in train_indices]
    )
    counts = [prediction_records[int(index)]["count"] for index in prediction_indices]
    output = {}
    for alpha in alphas:
        model = SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            max_iter=120,
            tol=1e-4,
            random_state=SEED,
            average=True,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_matrix, labels)
        log_probability = np.nan_to_num(
            model.predict_log_proba(prediction_matrix),
            nan=-30.0,
            neginf=-30.0,
            posinf=0.0,
        ).astype(np.float32)
        predictions, confidences = span_details(
            log_probability, model.classes_, counts, len(model.classes_), np
        )
        output[alpha] = {
            "predictions": predictions,
            "confidences": confidences,
        }
        del model
    del vectorizer, train_matrix, prediction_matrix
    gc.collect()
    return output


def aggregate_span(predictions, confidences, statistic):
    candidates = defaultdict(list)
    for prediction, confidence in zip(predictions, confidences):
        if prediction is not None and math.isfinite(confidence):
            candidates[tuple(prediction)].append(float(confidence))
    if not candidates:
        return None, 0, float("-inf")
    if statistic == "max":
        reducer = max
    elif statistic == "min":
        reducer = min
    else:
        reducer = lambda values: sum(values) / len(values)
    ranked = [
        (len(values), float(reducer(values)), tokens)
        for tokens, values in candidates.items()
    ]
    ranked.sort(reverse=True)
    votes, confidence, tokens = ranked[0]
    return list(tokens), votes, confidence


def nested_span_hpo(records, folds, groups, neural_oof, np):
    nested = {
        alpha: {
            "predictions": [[] for _ in records],
            "confidences": [[] for _ in records],
        }
        for alpha in SPAN_ALPHAS
    }
    from sklearn.model_selection import GroupKFold

    for outer_number, (outer_train, outer_validation) in enumerate(folds):
        inner_splits = GroupKFold(n_splits=N_FOLDS).split(
            outer_train, groups=groups[outer_train]
        )
        for inner_number, (inner_train_local, _) in enumerate(inner_splits):
            inner_train = outer_train[inner_train_local]
            family = fit_span_family(
                records,
                inner_train,
                records,
                outer_validation,
                SPAN_ALPHAS,
                np,
            )
            for alpha in SPAN_ALPHAS:
                for local_index, row_index in enumerate(outer_validation):
                    nested[alpha]["predictions"][int(row_index)].append(
                        family[alpha]["predictions"][local_index]
                    )
                    nested[alpha]["confidences"][int(row_index)].append(
                        family[alpha]["confidences"][local_index]
                    )
            log(
                f"span HPO outer={outer_number + 1}, inner={inner_number + 1} complete"
            )
            del family
    truths = [record["answer"] for record in records]
    best = None
    for alpha in SPAN_ALPHAS:
        for statistic in ("max", "mean", "min"):
            aggregated = [
                aggregate_span(predictions, confidences, statistic)
                for predictions, confidences in zip(
                    nested[alpha]["predictions"], nested[alpha]["confidences"]
                )
            ]
            finite = np.asarray(
                [item[2] for item in aggregated if math.isfinite(item[2])],
                dtype=float,
            )
            thresholds = (
                np.unique(
                    np.r_[
                        float("-inf"),
                        np.quantile(finite, np.linspace(0.0, 1.0, 101)),
                        float("inf"),
                    ]
                )
                if finite.size
                else np.asarray([float("inf")])
            )
            for minimum_votes in (1, 2, 3):
                for threshold in thresholds:
                    predictions = []
                    selected = 0
                    for row_index, base_prediction in enumerate(neural_oof):
                        span_prediction, votes, confidence = aggregated[row_index]
                        if (
                            span_prediction is not None
                            and votes >= minimum_votes
                            and confidence >= threshold
                        ):
                            predictions.append(span_prediction)
                            selected += 1
                        else:
                            predictions.append(base_prediction)
                    score = metric(truths, predictions)
                    if best is None or score > best["score"] + 1e-12:
                        best = {
                            "score": score,
                            "alpha": alpha,
                            "statistic": statistic,
                            "minimum_votes": minimum_votes,
                            "threshold": float(threshold),
                            "selected_rows": selected,
                        }
    log(
        f"cross-fitted score={best['score']:.6f}; alpha={best['alpha']:.1e}; "
        f"span_stat={best['statistic']}; votes={best['minimum_votes']}; "
        f"selected_rows={best['selected_rows']}"
    )
    return best


def final_span_ensemble(records, test_records, folds, configuration, np):
    test_indices = np.arange(len(test_records), dtype=np.int64)
    fold_predictions = []
    fold_confidences = []
    for fold_number, (train_indices, _) in enumerate(folds):
        family = fit_span_family(
            records,
            train_indices,
            test_records,
            test_indices,
            (configuration["alpha"],),
            np,
        )[configuration["alpha"]]
        fold_predictions.append(family["predictions"])
        fold_confidences.append(family["confidences"])
        log(f"final span model {fold_number + 1}/{N_FOLDS} complete")
    output = []
    for row_index in range(len(test_records)):
        prediction, votes, confidence = aggregate_span(
            [fold[row_index] for fold in fold_predictions],
            [fold[row_index] for fold in fold_confidences],
            configuration["statistic"],
        )
        if (
            prediction is not None
            and votes >= configuration["minimum_votes"]
            and confidence >= configuration["threshold"]
        ):
            output.append(prediction)
        else:
            output.append(None)
    return output


def audit_submission(path, test_records):
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["id", "answer_json"]:
                return False, "wrong columns or order"
            rows = list(reader)
        if len(rows) != len(test_records):
            return False, "wrong row count"
        expected_ids = [record["id"] for record in test_records]
        actual_ids = [row["id"] for row in rows]
        if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
            return False, "ID mismatch or duplicate"
        for row, record in zip(rows, test_records):
            value = json.loads(row["answer_json"])
            if set(value) != {"missing_tokens"} or not valid_tokens(
                value["missing_tokens"], record["count"]
            ):
                return False, f"invalid answer object for {record['id']}"
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
    if not train_path.is_file() or not test_path.is_file():
        missing = [str(path) for path in (train_path, test_path) if not path.is_file()]
        raise FileNotFoundError("missing required input: " + ", ".join(missing))

    import pandas as pd

    test_frame = pd.read_csv(test_path, dtype={"id": str})
    test_ids = [str(value) for value in test_frame["id"].tolist()]
    test_counts = [
        safe_count(value, 1) for value in test_frame["missing_token_count"].tolist()
    ]
    write_predictions(
        submission_out,
        test_ids,
        [["unknown"] * count for count in test_counts],
    )
    log(f"wrote early schema-valid placeholder with {len(test_ids)} rows")

    try:
        import numpy as np
        import torch
        from sklearn.model_selection import GroupKFold

        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(min(10, max(1, (os.cpu_count() or 4) - 2)))
        train_frame = pd.read_csv(train_path, dtype={"id": str})
        records = parse_records(train_frame, labelled=True)
        test_records = parse_records(test_frame, labelled=False)
        if len(records) < N_FOLDS * 2:
            log("warning: too few valid training rows; placeholder retained")
            return
        groups = procedure_groups(records, np)
        indices = np.arange(len(records), dtype=np.int64)
        folds = list(GroupKFold(n_splits=N_FOLDS).split(indices, groups=groups))
        log(
            "three-fold procedure split: "
            + ", ".join(
                f"{len(train_indices)}/{len(validation_indices)}"
                for train_indices, validation_indices in folds
            )
        )

        neural_artifacts = []
        try:
            for fold_number, (train_indices, validation_indices) in enumerate(folds):
                neural_artifacts.append(
                    train_neural_fold(
                        records,
                        train_indices,
                        validation_indices,
                        test_records,
                        fold_number,
                        torch,
                        np,
                        start_time,
                    )
                )
        except Exception as exc:
            log(f"warning: neural cross-fit unavailable; span models retained: {exc}")
            neural_artifacts = []
            gc.collect()

        if neural_artifacts:
            neural_oof = neural_oof_predictions(records, neural_artifacts)
            neural_score = metric(
                [record["answer"] for record in records], neural_oof
            )
            neural_test = neural_test_ensemble(
                test_records, neural_artifacts, np
            )
            log(f"three-fold neural OOF score={neural_score:.6f}")
        else:
            neural_oof = [["unknown"] * record["count"] for record in records]
            neural_test = [["unknown"] * record["count"] for record in test_records]

        span_configuration = nested_span_hpo(
            records, folds, groups, neural_oof, np
        )
        span_test = final_span_ensemble(
            records, test_records, folds, span_configuration, np
        )
        predictions = []
        for record, base_prediction, span_prediction in zip(
            test_records, neural_test, span_test
        ):
            prediction = (
                span_prediction if span_prediction is not None else base_prediction
            )
            if not valid_tokens(prediction, record["count"]):
                prediction = (
                    base_prediction
                    if valid_tokens(base_prediction, record["count"])
                    else ["unknown"] * record["count"]
                )
            predictions.append(prediction)

        write_predictions(
            submission_out,
            [record["id"] for record in test_records],
            predictions,
        )
        valid, reason = audit_submission(submission_out, test_records)
        if not valid:
            log(f"warning: final audit failed ({reason}); rewriting valid fallback")
            write_predictions(
                submission_out,
                [record["id"] for record in test_records],
                [["unknown"] * record["count"] for record in test_records],
            )
            valid, reason = audit_submission(submission_out, test_records)
        log(
            f"submission audit={valid} ({reason}); rows={len(test_records)}; "
            f"elapsed={time.monotonic() - start_time:.1f}s"
        )
    except Exception as exc:
        log(f"warning: solver stopped after protected placeholder was written: {exc}")


if __name__ == "__main__":
    main()
