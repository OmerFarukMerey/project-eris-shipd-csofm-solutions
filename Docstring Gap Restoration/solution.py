#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import gc
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
CPU_DEVICE = torch.device("cpu")

SEED = 20260720
MODEL_NAME = "Salesforce/codet5-base"
INPUT_TOKENS = 128
LABEL_TOKENS = 24
MAX_NEW_TOKENS = 20
TRAINABLE_DECODER_BLOCKS = 4
TRAIN_BATCH_SIZE = 32
INFERENCE_BATCH_SIZE = 64
MAX_TRAIN_ROWS = 200_000
VALIDATION_GROUPS = 2_048
PILOT_TRAIN_ROWS = 1_500
TUNE_ROWS = 256
REPORT_ROWS = 1_024
MAX_EXPECTED_TEST_ROWS = 50_000
HARD_TOTAL_SECONDS = 5_000.0
INFERENCE_SAFETY = 1.4
INFERENCE_BUFFER_SECONDS = 260.0
MIN_TRAIN_DEADLINE_SECONDS = 900.0
MAX_TRAIN_DEADLINE_SECONDS = 3_300.0
FLUSH_EVERY_BATCHES = 60
FALLBACK_TEXT = "[MODEL_UNAVAILABLE]"
LR_CANDIDATES = (3e-5, 6e-5, 1e-4)
DECODE_CANDIDATES = ((1, 1.0), (2, 0.8), (2, 1.0), (2, 1.2))


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED_AT
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(10, os.cpu_count() or 10))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def write_submission(ids: list[str], predictions: list[str], output_path: Path) -> None:
    frame = pd.DataFrame({"id": ids, "prediction": predictions}, columns=["id", "prediction"])
    frame.to_csv(output_path, index=False)


def write_placeholder(test: pd.DataFrame, output_path: Path) -> None:
    ids = test["id"].astype(str).tolist()
    write_submission(ids, [FALLBACK_TEXT] * len(ids), output_path)
    log(f"wrote early placeholder with {len(ids)} rows")


def build_source(masked_docstring: str, code_context: str) -> str:
    masked = str(masked_docstring)
    if "[GAP]" in masked:
        masked = masked.replace("[GAP]", "<extra_id_0>", 1)
    else:
        masked = masked + " <extra_id_0>"
    return "restore docstring: " + masked + " code: " + str(code_context)[:1600]


def build_grouped_split(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    split_doc = train["masked_docstring"].str.split("[GAP]", n=1, expand=True, regex=False)
    if split_doc.shape[1] == 1:
        split_doc[1] = ""
    complete_doc = (
        split_doc[0].fillna("")
        + train["target_span"].astype(str)
        + split_doc[1].fillna("")
    )
    groups = complete_doc.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    group_count = int(groups.nunique())
    if group_count < 2:
        indices = np.arange(len(train), dtype=np.int64)
        cut = max(1, len(indices) // 20)
        return indices[cut:], indices[:cut]
    validation_groups = min(VALIDATION_GROUPS, max(1, group_count // 20))
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_groups,
        random_state=SEED,
    )
    fit_indices, valid_indices = next(splitter.split(train, groups=groups))
    return fit_indices.astype(np.int64), valid_indices.astype(np.int64)


def encode_training_frame(
    frame: pd.DataFrame,
    tokenizer,
    chunk_size: int = 4_096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count = len(frame)
    input_ids = np.full((row_count, INPUT_TOKENS), tokenizer.pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((row_count, INPUT_TOKENS), dtype=np.uint8)
    labels = np.full((row_count, LABEL_TOKENS), tokenizer.pad_token_id, dtype=np.int32)
    for start in range(0, row_count, chunk_size):
        stop = min(row_count, start + chunk_size)
        part = frame.iloc[start:stop]
        sources = [
            build_source(masked, code)
            for masked, code in zip(part["masked_docstring"], part["code_context"])
        ]
        targets = [
            "<extra_id_0>" + str(target) + "<extra_id_1>"
            for target in part["target_span"]
        ]
        encoded = tokenizer(
            sources,
            padding="max_length",
            truncation=True,
            max_length=INPUT_TOKENS,
            return_tensors="np",
        )
        encoded_targets = tokenizer(
            targets,
            padding="max_length",
            truncation=True,
            max_length=LABEL_TOKENS,
            return_tensors="np",
        )["input_ids"]
        input_ids[start:stop] = encoded["input_ids"]
        attention_mask[start:stop] = encoded["attention_mask"]
        labels[start:stop] = encoded_targets
    return input_ids, attention_mask, labels


def configure_for_partial_finetuning(model) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in model.decoder.block[-TRAINABLE_DECODER_BLOCKS:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.decoder.final_layer_norm.parameters():
        parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def length_sorted_buckets(attention_mask: np.ndarray, seed: int) -> list[np.ndarray]:
    order = np.argsort(attention_mask.sum(axis=1), kind="stable")
    buckets = [
        order[start : start + TRAIN_BATCH_SIZE]
        for start in range(0, len(order), TRAIN_BATCH_SIZE)
    ]
    permutation = np.random.default_rng(seed).permutation(len(buckets))
    return [buckets[index] for index in permutation]


def train_encoded(
    model,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    learning_rate: float,
    seed: int,
    deadline_seconds: float | None,
) -> tuple[int, float]:
    input_ids, attention_mask, labels_array = arrays
    trainable = configure_for_partial_finetuning(model)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    buckets = length_sorted_buckets(attention_mask, seed)
    losses: list[float] = []
    rows_trained = 0
    model.config.use_cache = False
    model.train()
    for step, batch_indices in enumerate(buckets):
        if (
            deadline_seconds is not None
            and time.monotonic() - STARTED_AT >= deadline_seconds
        ):
            log("wall-clock guard ended training; moving to inference")
            break
        max_input = int(attention_mask[batch_indices].sum(axis=1).max())
        label_slice = labels_array[batch_indices]
        max_label = int((label_slice != TOKENIZER_PAD_ID).sum(axis=1).max())
        max_input = max(1, max_input)
        max_label = max(1, max_label)
        batch_input = torch.from_numpy(input_ids[batch_indices, :max_input]).long()
        batch_mask = torch.from_numpy(attention_mask[batch_indices, :max_input]).long()
        batch_labels = torch.from_numpy(label_slice[:, :max_label]).long()
        batch_labels[batch_labels == TOKENIZER_PAD_ID] = -100

        optimizer.zero_grad(set_to_none=True)
        loss = model(
            input_ids=batch_input,
            attention_mask=batch_mask,
            labels=batch_labels,
        ).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        rows_trained += len(batch_indices)
        losses.append(float(loss.detach()))
        if (step + 1) % 200 == 0:
            recent = float(np.mean(losses[-200:]))
            log(f"trained {rows_trained} rows; recent loss={recent:.4f}")
    model.config.use_cache = True
    model.eval()
    del optimizer
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return rows_trained, mean_loss


def decode_span(token_ids, tokenizer) -> str:
    raw = tokenizer.decode(token_ids, skip_special_tokens=False)
    if "<extra_id_0>" in raw:
        text = raw.split("<extra_id_0>", 1)[1]
        endpoints = [
            position
            for marker in ("<extra_id_1>", "<extra_id_2>", "</s>")
            if (position := text.find(marker)) >= 0
        ]
        if endpoints:
            text = text[: min(endpoints)]
        prediction = text.replace("<pad>", "").replace("<s>", "").strip()
    else:
        prediction = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return prediction if prediction else FALLBACK_TEXT


def generate_batch(
    model,
    tokenizer,
    masked_docstrings: list[str],
    code_contexts: list[str],
    beams: int,
    length_penalty: float,
) -> list[str]:
    sources = [
        build_source(masked, code)
        for masked, code in zip(masked_docstrings, code_contexts)
    ]
    encoded = tokenizer(
        sources,
        padding=True,
        truncation=True,
        max_length=INPUT_TOKENS,
        return_tensors="pt",
    )
    encoded = {name: tensor.to(CPU_DEVICE) for name, tensor in encoded.items()}
    generation_args = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": beams,
        "do_sample": False,
    }
    if beams > 1:
        generation_args["length_penalty"] = length_penalty
        generation_args["early_stopping"] = True
    with torch.inference_mode():
        generated = model.generate(**encoded, **generation_args)
    predictions: list[str] = []
    for token_ids in generated:
        try:
            predictions.append(decode_span(token_ids, tokenizer))
        except Exception as error:
            log(f"row decode warning: {type(error).__name__}: {error}")
            predictions.append(FALLBACK_TEXT)
    return predictions


def predict_frame(
    model,
    tokenizer,
    frame: pd.DataFrame,
    beams: int,
    length_penalty: float,
    on_progress: Callable[[int, list[str]], None] | None = None,
) -> tuple[list[str], float]:
    started = time.monotonic()
    predictions: list[str] = []
    model.eval()
    total_batches = 0
    for start in range(0, len(frame), INFERENCE_BATCH_SIZE):
        part = frame.iloc[start : start + INFERENCE_BATCH_SIZE]
        masked = part["masked_docstring"].astype(str).tolist()
        code = part["code_context"].astype(str).tolist()
        try:
            batch_predictions = generate_batch(
                model, tokenizer, masked, code, beams, length_penalty
            )
        except Exception as batch_error:
            log(
                "batch inference warning: "
                f"{type(batch_error).__name__}: {batch_error}; retrying rows"
            )
            batch_predictions = []
            for one_masked, one_code in zip(masked, code):
                try:
                    one_prediction = generate_batch(
                        model,
                        tokenizer,
                        [one_masked],
                        [one_code],
                        1,
                        1.0,
                    )[0]
                except Exception as row_error:
                    log(f"row inference warning: {type(row_error).__name__}: {row_error}")
                    one_prediction = FALLBACK_TEXT
                batch_predictions.append(one_prediction)
        predictions.extend(batch_predictions)
        total_batches += 1
        if on_progress is not None and total_batches % FLUSH_EVERY_BATCHES == 0:
            on_progress(len(predictions), predictions)
        if total_batches % 100 == 0:
            log(f"generated {len(predictions)} of {len(frame)} rows")
    return predictions, time.monotonic() - started


def character_ngram_f_score(prediction: str, reference: str) -> float:
    if prediction == reference:
        return 1.0
    if not prediction or not reference:
        return 0.0
    overlap = 0
    predicted_total = 0
    reference_total = 0
    for order in range(1, 7):
        predicted = Counter(
            prediction[index : index + order]
            for index in range(max(0, len(prediction) - order + 1))
        )
        reference_ngrams = Counter(
            reference[index : index + order]
            for index in range(max(0, len(reference) - order + 1))
        )
        overlap += sum((predicted & reference_ngrams).values())
        predicted_total += sum(predicted.values())
        reference_total += sum(reference_ngrams.values())
    if overlap == 0 or predicted_total == 0 or reference_total == 0:
        return 0.0
    precision = overlap / predicted_total
    recall = overlap / reference_total
    return 2.0 * precision * recall / (precision + recall)


def score_predictions(predictions: list[str], references: pd.Series) -> float:
    scores = [
        character_ngram_f_score(prediction, str(reference))
        for prediction, reference in zip(predictions, references)
    ]
    return float(np.mean(scores)) if scores else 0.0


def load_backbone() -> torch.nn.Module:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return model.to(CPU_DEVICE)


def search_learning_rate(
    tokenizer,
    pilot_train: pd.DataFrame,
    tune_frame: pd.DataFrame,
) -> tuple[float, list[tuple[float, float]], float]:
    arrays = encode_training_frame(pilot_train, tokenizer)
    results: list[tuple[float, float]] = []
    inference_ms_per_row = 0.0
    for learning_rate in LR_CANDIDATES:
        seed_everything(SEED)
        candidate_model = load_backbone()
        train_encoded(
            candidate_model,
            arrays,
            learning_rate,
            SEED,
            deadline_seconds=None,
        )
        predictions, seconds = predict_frame(
            candidate_model,
            tokenizer,
            tune_frame,
            beams=1,
            length_penalty=1.0,
        )
        score = score_predictions(predictions, tune_frame["target_span"])
        results.append((learning_rate, score))
        inference_ms_per_row = max(
            inference_ms_per_row, 1000.0 * seconds / max(1, len(tune_frame))
        )
        log(f"learning-rate candidate {learning_rate:g}: validation={score:.6f}")
        del candidate_model
        gc.collect()
    del arrays
    if not results:
        raise RuntimeError("learning-rate search could not evaluate a candidate")
    selected = max(results, key=lambda item: item[1])[0]
    return selected, results, inference_ms_per_row


def resolve_training_deadline(inference_ms_per_row: float) -> float:
    projected_test = inference_ms_per_row / 1000.0 * MAX_EXPECTED_TEST_ROWS
    reserved = projected_test * INFERENCE_SAFETY + INFERENCE_BUFFER_SECONDS
    deadline = HARD_TOTAL_SECONDS - reserved
    deadline = max(MIN_TRAIN_DEADLINE_SECONDS, min(MAX_TRAIN_DEADLINE_SECONDS, deadline))
    log(
        f"inference≈{inference_ms_per_row:.1f} ms/row, projected 50k test≈{projected_test:.0f}s, "
        f"training deadline set to {deadline:.0f}s"
    )
    return deadline


def search_decoding(
    model,
    tokenizer,
    tune_frame: pd.DataFrame,
) -> tuple[int, float, list[tuple[int, float, float, float]]]:
    results: list[tuple[int, float, float, float]] = []
    for beams, length_penalty in DECODE_CANDIDATES:
        predictions, seconds = predict_frame(
            model,
            tokenizer,
            tune_frame,
            beams=beams,
            length_penalty=length_penalty,
        )
        score = score_predictions(predictions, tune_frame["target_span"])
        projected_test_seconds = seconds * MAX_EXPECTED_TEST_ROWS / max(1, len(tune_frame))
        results.append((beams, length_penalty, score, projected_test_seconds))
        log(
            f"decode candidate beams={beams}, length_penalty={length_penalty:.1f}: "
            f"validation={score:.6f}, projected_test_seconds={projected_test_seconds:.1f}"
        )
    elapsed = time.monotonic() - STARTED_AT
    feasible = [
        result
        for result in results
        if elapsed + result[3] * INFERENCE_SAFETY <= HARD_TOTAL_SECONDS
    ]
    pool = feasible if feasible else [min(results, key=lambda item: item[3])]
    selected = max(pool, key=lambda item: item[2])
    return selected[0], selected[1], results




def complete_predictions(predictions: list[str], row_count: int) -> list[str]:
    completed: list[str] = []
    for row_index in range(row_count):
        if row_index < len(predictions):
            value = str(predictions[row_index]).strip()
        else:
            value = ""
        completed.append(value if value else FALLBACK_TEXT)
    return completed


def solve(train_path: Path, test: pd.DataFrame, output_path: Path) -> None:
    global TOKENIZER_PAD_ID

    train = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    require_columns(
        train,
        ["id", "code_context", "masked_docstring", "target_span"],
        "train.csv",
    )
    log(f"loaded train={len(train)} and test={len(test)}")

    fit_indices, valid_indices = build_grouped_split(train)
    rng = np.random.default_rng(SEED)
    fit_indices = rng.permutation(fit_indices)
    valid_indices = rng.permutation(valid_indices)
    valid = train.iloc[valid_indices].reset_index(drop=True)

    lr_tune = valid.iloc[: min(TUNE_ROWS, len(valid))].reset_index(drop=True)
    decode_start = len(lr_tune)
    decode_tune = valid.iloc[
        decode_start : decode_start + min(TUNE_ROWS, max(0, len(valid) - decode_start))
    ].reset_index(drop=True)
    if decode_tune.empty:
        decode_tune = lr_tune
    report_start = decode_start + len(decode_tune)
    report = valid.iloc[report_start : report_start + REPORT_ROWS].reset_index(drop=True)
    if report.empty:
        report = valid

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    TOKENIZER_PAD_ID = int(tokenizer.pad_token_id)
    pilot_count = min(PILOT_TRAIN_ROWS, len(fit_indices))
    pilot_train = train.iloc[fit_indices[:pilot_count]].reset_index(drop=True)
    selected_lr, lr_results, inference_ms_per_row = search_learning_rate(
        tokenizer, pilot_train, lr_tune
    )
    log(f"selected learning rate={selected_lr:g} from {lr_results}")
    training_deadline = resolve_training_deadline(inference_ms_per_row)

    main_count = min(MAX_TRAIN_ROWS, len(fit_indices))
    main_train = train.iloc[fit_indices[:main_count]].reset_index(drop=True)
    del pilot_train
    log(f"encoding {len(main_train)} train-only rows")
    main_arrays = encode_training_frame(main_train, tokenizer)
    del main_train

    model = load_backbone()
    try:
        rows_trained, mean_loss = train_encoded(
            model,
            main_arrays,
            selected_lr,
            SEED + 1,
            deadline_seconds=training_deadline,
        )
        log(f"main fine-tuning rows={rows_trained}, mean_loss={mean_loss:.6f}")
    except Exception as training_error:
        log(
            f"training warning: {type(training_error).__name__}: {training_error}; "
            "continuing with the available model"
        )
    del main_arrays
    gc.collect()

    try:
        beams, length_penalty, decode_results = search_decoding(
            model,
            tokenizer,
            decode_tune,
        )
        log(
            f"selected decode beams={beams}, length_penalty={length_penalty:.1f} "
            f"from {decode_results}"
        )
    except Exception as decode_error:
        log(
            f"decode-search warning: {type(decode_error).__name__}: {decode_error}; "
            "using greedy generation"
        )
        beams, length_penalty = 1, 1.0

    try:
        report_predictions, _ = predict_frame(
            model,
            tokenizer,
            report,
            beams=beams,
            length_penalty=length_penalty,
        )
        report_score = score_predictions(report_predictions, report["target_span"])
        exact = float(
            np.mean(
                np.asarray(report_predictions, dtype=object)
                == report["target_span"].astype(str).to_numpy()
            )
        )
        log(
            f"group-held-out validation rows={len(report)}, "
            f"character_ngram_f={report_score:.6f}, exact_match={exact:.6f}"
        )
    except Exception as validation_error:
        log(f"validation-report warning: {type(validation_error).__name__}: {validation_error}")

    del train, valid, report, lr_tune, decode_tune
    gc.collect()

    ids = test["id"].astype(str).tolist()

    def flush(done_count: int, partial: list[str]) -> None:
        current = complete_predictions(partial, len(ids))
        write_submission(ids, current, output_path)
        log(f"flushed submission with {done_count} real predictions")

    predictions, inference_seconds = predict_frame(
        model,
        tokenizer,
        test,
        beams=beams,
        length_penalty=length_penalty,
        on_progress=flush,
    )
    log(f"test inference completed in {inference_seconds:.1f}s")

    completed = complete_predictions(predictions, len(ids))
    write_submission(ids, completed, output_path)

    written = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    schema_ok = written.columns.tolist() == ["id", "prediction"]
    row_count_ok = len(written) == len(test)
    ids_ok = written["id"].astype(str).tolist() == ids
    nonempty_ok = all(bool(str(value).strip()) for value in written["prediction"])
    log(
        "submission checks: "
        f"schema={schema_ok}, rows={row_count_ok}, ids={ids_ok}, nonempty={nonempty_ok}"
    )
    if not (schema_ok and row_count_ok and ids_ok and nonempty_ok):
        repaired = complete_predictions(completed, len(ids))
        write_submission(ids, repaired, output_path)
        log("rewrote submission after completeness repair")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    public_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    train_path = public_dir / "train.csv"
    test_path = public_dir / "test.csv"
    for required_path in (train_path, test_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"required input not found: {required_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
    require_columns(test, ["id", "code_context", "masked_docstring"], "test.csv")
    write_placeholder(test, output_path)

    try:
        solve(train_path, test, output_path)
    except Exception as error:
        log(
            f"solver warning: {type(error).__name__}: {error}; "
            "preserving the schema-valid placeholder"
        )


STARTED_AT = time.monotonic()
TOKENIZER_PAD_ID = 0
seed_everything(SEED)

if __name__ == "__main__":
    main()
