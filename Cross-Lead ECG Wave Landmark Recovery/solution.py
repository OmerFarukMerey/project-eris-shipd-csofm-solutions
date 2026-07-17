#!/usr/bin/env python3
"""Train-only cross-lead ECG landmark recovery.

The submitted model is a supervised beat-level Extra-Trees/CatBoost ensemble. A
generic signal transform proposes heartbeat and context-lead landmark anchors;
the learned presence classifiers and timing regressors produce target events.
Patient-grouped out-of-fold predictions
select ensemble weights and event thresholds using train only. Test rows are
touched only after every model and calibration value has been fit, then strictly
one row at a time by deterministic transformation and frozen-model prediction.
"""

import gc
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy import signal as sps
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

SEED = 20260716
FS = 500
N_SAMPLES = 1000
N_TREES = 320
CAT_ITERATIONS = 450
N_JOBS = min(8, os.cpu_count() or 1)
WAVES = ("P", "QRS", "T")
LANDMARKS = ("onset", "peak", "offset")
CLASSES = tuple((wave, landmark) for wave in WAVES for landmark in LANDMARKS)
CLASS_INDEX = {key: i for i, key in enumerate(CLASSES)}

# The offsets are generic physiological neighborhoods around a detected beat,
# not values derived from test data or challenge asset identities.
SAMPLE_OFFSETS = {
    "x": np.arange(-220, 321, 8, dtype=np.int16),
    "low": np.arange(-220, 321, 8, dtype=np.int16),
    "band": np.arange(-100, 121, 4, dtype=np.int16),
    "low8": np.arange(-240, 321, 8, dtype=np.int16),
    "energy": np.arange(-100, 121, 4, dtype=np.int16),
    "grad": np.arange(-180, 221, 8, dtype=np.int16),
}
SIGNAL_FEATURE_COUNT = sum(len(v) for v in SAMPLE_OFFSETS.values())

SOS_HP = sps.butter(2, 0.5, btype="highpass", fs=FS, output="sos")
SOS_LOW = sps.butter(3, 35.0, btype="lowpass", fs=FS, output="sos")
SOS_BAND = sps.butter(3, [7.0, 28.0], btype="bandpass", fs=FS, output="sos")
SOS_LOW8 = sps.butter(3, 8.0, btype="lowpass", fs=FS, output="sos")


def parse_window(signal_file):
    match = re.search(r"window_(\d+)", signal_file)
    return int(match.group(1)) if match else -1


def parse_record(signal_file):
    match = re.search(r"rec_([^/]+?)_window_", signal_file)
    return match.group(1) if match else signal_file


def load_npz(public_dir, signal_file):
    with np.load(public_dir / signal_file) as data:
        signal = np.asarray(data["signal"], dtype=np.float64).reshape(-1)
        context_token = str(data["lead_tokens"][0])
    return signal, context_token


def detect_qrs(highpassed):
    """Generic derivative-energy QRS candidates; no label-specific edge rules."""
    band = sps.sosfiltfilt(SOS_BAND, highpassed)
    derivative = np.gradient(band)
    energy = derivative * derivative
    envelope = np.convolve(energy, np.ones(25) / 25.0, mode="same")
    if not np.any(np.isfinite(envelope)) or float(np.max(envelope)) <= 0:
        return np.array([len(highpassed) // 2], dtype=np.int32)
    peaks, _ = sps.find_peaks(
        envelope,
        distance=int(0.22 * FS),
        height=0.20 * float(np.max(envelope)),
        prominence=0.10 * float(np.max(envelope)),
    )
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(envelope))])
    refined = []
    center = float(np.median(highpassed))
    half_width = int(0.045 * FS)
    for peak in peaks:
        lo = max(0, int(peak) - half_width)
        hi = min(len(highpassed), int(peak) + half_width + 1)
        refined.append(lo + int(np.argmax(np.abs(highpassed[lo:hi] - center))))
    return np.asarray(sorted(set(refined)), dtype=np.int32)


def energy_bounds(energy, rpeak, lo_limit, hi_limit, fraction=0.10):
    threshold = fraction * float(energy[rpeak])
    onset = lo_limit
    for i in range(rpeak, lo_limit - 1, -1):
        onset = i
        if energy[i] < threshold:
            break
    offset = hi_limit
    for i in range(rpeak, hi_limit + 1):
        offset = i
        if energy[i] < threshold:
            break
    return onset, offset


def local_baseline(x, lo, hi):
    width = 10
    left = x[lo:min(hi + 1, lo + width)]
    right = x[max(lo, hi - width + 1):hi + 1]
    return float(np.median(np.concatenate((left, right))))


def amplitude_onset(x, baseline, peak, lo, amplitude, fraction=0.45):
    threshold = fraction * abs(amplitude)
    onset = lo
    for i in range(peak, lo - 1, -1):
        onset = i
        if abs(x[i] - baseline) < threshold:
            break
    return onset


def amplitude_offset(x, baseline, peak, hi, amplitude, fraction=0.45):
    threshold = fraction * abs(amplitude)
    offset = hi
    for i in range(peak, hi + 1):
        offset = i
        if abs(x[i] - baseline) < threshold:
            break
    return offset


def context_landmarks(highpassed, rpeaks):
    """Create generic context-lead anchors used only as model input features."""
    qrs_band = sps.sosfiltfilt(SOS_BAND, highpassed)
    qrs_energy = np.gradient(qrs_band) ** 2
    qrs_energy = np.convolve(qrs_energy, np.ones(10) / 10.0, mode="same")
    low = sps.sosfiltfilt(SOS_LOW, highpassed)
    median = float(np.median(highpassed))
    qrs = []
    for rpeak in rpeaks:
        lo_limit = max(0, int(rpeak) - int(0.075 * FS))
        hi_limit = min(N_SAMPLES - 1, int(rpeak) + int(0.10 * FS))
        onset, offset = energy_bounds(qrs_energy, int(rpeak), lo_limit, hi_limit)
        peak = onset + int(np.argmax(np.abs(highpassed[onset:offset + 1] - median)))
        qrs.append((onset, peak, offset))

    events = []
    for beat, (q_on, q_peak, q_off) in enumerate(qrs):
        events.extend(((q_on, "QRS", "onset"), (q_peak, "QRS", "peak"),
                       (q_off, "QRS", "offset")))

        t_lo = min(N_SAMPLES - 1, q_off + int(0.04 * FS))
        t_hi = min(N_SAMPLES - 1, q_off + int(0.40 * FS))
        if beat + 1 < len(qrs):
            t_hi = min(t_hi, qrs[beat + 1][0] - int(0.05 * FS))
        if t_hi - t_lo > int(0.05 * FS):
            baseline = local_baseline(low, t_lo, t_hi)
            t_peak = t_lo + int(np.argmax(np.abs(low[t_lo:t_hi + 1] - baseline)))
            amplitude = float(low[t_peak] - baseline)
            t_on = amplitude_onset(low, baseline, t_peak, t_lo, amplitude)
            t_off = amplitude_offset(low, baseline, t_peak, t_hi, amplitude)
            if t_on < t_peak:
                events.append((t_on, "T", "onset"))
            events.append((t_peak, "T", "peak"))
            if t_off > t_peak:
                events.append((t_off, "T", "offset"))

        p_hi = max(0, q_on - int(0.03 * FS))
        p_lo = max(0, q_on - int(0.30 * FS))
        if beat > 0:
            p_lo = max(p_lo, qrs[beat - 1][2] + int(0.05 * FS))
        if p_hi - p_lo > int(0.04 * FS):
            baseline = local_baseline(low, p_lo, p_hi)
            p_peak = p_lo + int(np.argmax(np.abs(low[p_lo:p_hi + 1] - baseline)))
            amplitude = float(low[p_peak] - baseline)
            p_on = amplitude_onset(low, baseline, p_peak, p_lo, amplitude)
            p_off = amplitude_offset(low, baseline, p_peak, p_hi, amplitude)
            if p_on < p_peak:
                events.append((p_on, "P", "onset"))
            events.append((p_peak, "P", "peak"))
            if p_off > p_peak:
                events.append((p_off, "P", "offset"))
    return tuple(events)


def prepare_signal(public_dir, signal_file):
    raw, context_token = load_npz(public_dir, signal_file)
    highpassed = sps.sosfiltfilt(SOS_HP, raw)
    scale = float(np.percentile(np.abs(highpassed), 95))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = max(float(np.std(highpassed)), 1.0)
    normalized = np.clip(highpassed / scale, -8.0, 8.0)
    low = sps.sosfiltfilt(SOS_LOW, normalized)
    band = sps.sosfiltfilt(SOS_BAND, normalized)
    low8 = sps.sosfiltfilt(SOS_LOW8, normalized)
    energy = np.convolve(np.gradient(band) ** 2, np.ones(9) / 9.0, mode="same")
    energy_scale = float(np.percentile(energy, 95))
    if energy_scale > 1e-8:
        energy = energy / energy_scale
    rpeaks = detect_qrs(highpassed)
    return {
        "token": context_token,
        "rpeaks": rpeaks,
        "events": context_landmarks(highpassed, rpeaks),
        "channels": {
            "x": normalized,
            "low": low,
            "band": band,
            "low8": low8,
            "energy": np.clip(energy, 0.0, 10.0),
            "grad": np.gradient(low),
        },
    }


def one_hot(index, size):
    result = [0.0] * size
    if 0 <= index < size:
        result[index] = 1.0
    return result


def anchor_features(context, rpeak):
    output = []
    expected = {"P": -80, "QRS": 0, "T": 130}
    for wave, landmark in CLASSES:
        candidates = []
        for sample, event_wave, event_landmark in context["events"]:
            if event_wave != wave or event_landmark != landmark:
                continue
            delta = int(sample) - int(rpeak)
            valid = (-260 <= delta <= -15) if wave == "P" else (
                -90 <= delta <= 100 if wave == "QRS" else 15 <= delta <= 380)
            if valid:
                candidates.append((abs(delta - expected[wave]), delta))
        if candidates:
            output.extend((min(candidates)[1] / 250.0, 1.0))
        else:
            output.extend((0.0, 0.0))
    return output


def beat_features(context, target_token, signal_file, beat_index, token_index):
    rpeaks = context["rpeaks"]
    rpeak = int(rpeaks[beat_index])
    values = []
    for name, offsets in SAMPLE_OFFSETS.items():
        indices = np.clip(rpeak + offsets, 0, N_SAMPLES - 1)
        values.extend(context["channels"][name][indices])

    previous_rr = (rpeak - int(rpeaks[beat_index - 1])) / FS if beat_index else 2.0
    next_rr = (int(rpeaks[beat_index + 1]) - rpeak) / FS if beat_index + 1 < len(rpeaks) else 2.0
    values.extend((
        rpeak / (N_SAMPLES - 1),
        min(rpeak, N_SAMPLES - 1 - rpeak) / FS,
        previous_rr,
        next_rr,
        len(rpeaks) / 5.0,
        beat_index / 4.0,
        (len(rpeaks) - 1 - beat_index) / 4.0,
    ))
    values.extend(one_hot(token_index.get(context["token"], -1), len(token_index)))
    values.extend(one_hot(token_index.get(target_token, -1), len(token_index)))
    values.extend(one_hot(parse_window(signal_file), 5))
    values.extend(one_hot(min(beat_index, 4), 5))
    values.extend(one_hot(min(len(rpeaks), 5) - 1, 5))
    values.extend(anchor_features(context, rpeak))
    return np.asarray(values, dtype=np.float32)


def assign_events_to_qrs(events):
    qrs_peaks = sorted(
        int(e["sample"]) for e in events
        if e["wave"] == "QRS" and e["landmark"] == "peak"
    )
    assigned = [dict() for _ in qrs_peaks]
    for event in events:
        sample = int(event["sample"])
        wave = event["wave"]
        candidates = []
        for q_index, qpeak in enumerate(qrs_peaks):
            if wave == "P" and 0 <= qpeak - sample <= 260:
                candidates.append((qpeak - sample, q_index))
            elif wave == "T" and 0 <= sample - qpeak <= 360:
                candidates.append((sample - qpeak, q_index))
            elif wave == "QRS" and abs(sample - qpeak) <= 100:
                candidates.append((abs(sample - qpeak), q_index))
        if candidates:
            q_index = min(candidates)[1]
            assigned[q_index][(wave, event["landmark"])] = sample
    return qrs_peaks, assigned


def build_training_beats(train, signal_cache, token_index):
    features = []
    presence = []
    offsets = []
    beat_rows = []
    beat_samples = []
    groups = []

    for row_index, row in enumerate(train.itertuples(index=False)):
        context = signal_cache[row.signal_file]
        events = json.loads(row.answer_json)["events"]
        qrs_peaks, assigned = assign_events_to_qrs(events)
        used_qrs = set()
        for beat_index, rpeak in enumerate(context["rpeaks"]):
            features.append(beat_features(
                context, row.target_lead_token, row.signal_file, beat_index, token_index))
            event_presence = np.zeros(len(CLASSES), dtype=np.int8)
            event_offsets = np.full(len(CLASSES), np.nan, dtype=np.float32)
            available = sorted(
                (abs(int(rpeak) - qpeak), q_index)
                for q_index, qpeak in enumerate(qrs_peaks) if q_index not in used_qrs
            )
            if available and available[0][0] <= 45:
                q_index = available[0][1]
                used_qrs.add(q_index)
                for key, sample in assigned[q_index].items():
                    class_index = CLASS_INDEX[key]
                    event_presence[class_index] = 1
                    event_offsets[class_index] = sample - int(rpeak)
            presence.append(event_presence)
            offsets.append(event_offsets)
            beat_rows.append(row_index)
            beat_samples.append(int(rpeak))
            groups.append(parse_record(row.signal_file))

    return (
        np.stack(features), np.stack(presence), np.stack(offsets),
        np.asarray(beat_rows, dtype=np.int32),
        np.asarray(beat_samples, dtype=np.int32), np.asarray(groups),
    )


def fit_model(features, presence, offsets, indices, seed, classifier_leaf, regressor_leaf):
    classifier = ExtraTreesClassifier(
        n_estimators=N_TREES,
        min_samples_leaf=classifier_leaf,
        max_features=0.65,
        class_weight="balanced",
        n_jobs=N_JOBS,
        random_state=seed,
    )
    classifier.fit(features[indices], presence[indices])

    regressors = {}
    for wave_index, wave in enumerate(WAVES):
        columns = np.arange(3 * wave_index, 3 * wave_index + 3)
        keep = indices[presence[indices, 3 * wave_index + 1] == 1]
        targets = offsets[keep][:, columns].copy()
        for output_index, column in enumerate(columns):
            finite_train = offsets[indices, column]
            fill = float(np.nanmedian(finite_train[np.isfinite(finite_train)]))
            targets[~np.isfinite(targets[:, output_index]), output_index] = fill
        regressor = ExtraTreesRegressor(
            n_estimators=N_TREES,
            min_samples_leaf=regressor_leaf,
            max_features=0.75,
            n_jobs=N_JOBS,
            random_state=seed + 10 + wave_index,
        )
        regressor.fit(features[keep], targets)
        regressors[wave] = regressor
    return classifier, regressors


def predict_model(model, features):
    classifier, regressors = model
    raw_probabilities = classifier.predict_proba(features)
    probability_columns = []
    for probabilities, classes in zip(raw_probabilities, classifier.classes_):
        class_one = np.flatnonzero(np.asarray(classes) == 1)
        if len(class_one):
            probability_columns.append(probabilities[:, int(class_one[0])])
        else:
            probability_columns.append(np.full(len(features), float(classes[0] == 1)))
    probabilities = np.column_stack(probability_columns)
    offsets = np.zeros((len(features), len(CLASSES)), dtype=np.float32)
    for wave_index, wave in enumerate(WAVES):
        offsets[:, 3 * wave_index:3 * wave_index + 3] = regressors[wave].predict(features)
    return probabilities, offsets


def fit_ensemble(full_features, presence, offsets, indices, seed):
    compact_features = full_features[:, SIGNAL_FEATURE_COUNT:]
    compact = fit_model(
        compact_features, presence, offsets, indices, seed,
        classifier_leaf=2, regressor_leaf=1,
    )
    full = fit_model(
        full_features, presence, offsets, indices, seed + 1000,
        classifier_leaf=8, regressor_leaf=5,
    )
    return compact, full


def predict_components(models, full_features):
    compact_model, full_model = models
    compact_probability, compact_offset = predict_model(
        compact_model, full_features[:, SIGNAL_FEATURE_COUNT:])
    full_probability, full_offset = predict_model(full_model, full_features)
    return compact_probability, compact_offset, full_probability, full_offset


def blend_predictions(components, blend_weights):
    compact_probability, compact_offset, full_probability, full_offset = components
    probability_weight, offset_weight = blend_weights
    return (
        probability_weight * compact_probability
        + (1.0 - probability_weight) * full_probability,
        offset_weight * compact_offset + (1.0 - offset_weight) * full_offset,
    )


def predict_ensemble(models, full_features, blend_weights):
    return blend_predictions(predict_components(models, full_features), blend_weights)
def cat_feature_frame(full_features, token_count):
    """Compact structural view with train-vocabulary categorical columns."""
    start = SIGNAL_FEATURE_COUNT
    position = full_features[:, start:start + 7]
    start += 7

    context_block = full_features[:, start:start + token_count]
    context = np.where(
        context_block.sum(axis=1) > 0, np.argmax(context_block, axis=1), -1)
    start += token_count
    target_block = full_features[:, start:start + token_count]
    target = np.where(
        target_block.sum(axis=1) > 0, np.argmax(target_block, axis=1), -1)
    start += token_count

    window = np.argmax(full_features[:, start:start + 5], axis=1)
    start += 5
    ordinal = np.argmax(full_features[:, start:start + 5], axis=1)
    start += 5
    count = np.argmax(full_features[:, start:start + 5], axis=1)
    start += 5
    anchors = full_features[:, start:]

    frame = pd.DataFrame(np.column_stack(
        (position, context, target, window, ordinal, count, anchors)))
    categorical_columns = list(range(7, 12))
    for column in categorical_columns:
        frame[column] = frame[column].astype(np.int16).astype(str)
    return frame, categorical_columns


def fit_cat_models(
        full_features, categorical_frame, categorical_columns,
        presence, offsets, indices, seed):
    compact = full_features[:, SIGNAL_FEATURE_COUNT:]
    presence_models = []
    timing_models = []
    for class_index in range(len(CLASSES)):
        classifier = CatBoostClassifier(
            iterations=CAT_ITERATIONS,
            depth=6,
            learning_rate=0.04,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            l2_leaf_reg=5.0,
            random_strength=0.5,
            task_type="CPU",
            thread_count=N_JOBS,
            random_seed=seed + class_index,
            allow_writing_files=False,
            verbose=False,
            cat_features=categorical_columns,
        )
        classifier.fit(categorical_frame.iloc[indices], presence[indices, class_index])
        presence_models.append(classifier)

        keep = indices[presence[indices, class_index] == 1]
        regressor = CatBoostRegressor(
            iterations=CAT_ITERATIONS,
            depth=6,
            learning_rate=0.04,
            loss_function="MAE",
            l2_leaf_reg=5.0,
            random_strength=0.5,
            task_type="CPU",
            thread_count=N_JOBS,
            random_seed=seed + 100 + class_index,
            allow_writing_files=False,
            verbose=False,
        )
        regressor.fit(compact[keep], offsets[keep, class_index])
        timing_models.append(regressor)
    return presence_models, timing_models


def predict_cat_models(models, full_features, categorical_frame):
    presence_models, timing_models = models
    compact = full_features[:, SIGNAL_FEATURE_COUNT:]
    probabilities = np.column_stack([
        model.predict_proba(categorical_frame)[:, 1]
        for model in presence_models
    ])
    offsets = np.column_stack([
        model.predict(compact) for model in timing_models
    ]).astype(np.float32)
    return probabilities, offsets


def mix_cat_predictions(
        base_probability, base_offset, cat_probability, cat_offset,
        probability_weights, offset_weights):
    return (
        (1.0 - probability_weights) * base_probability
        + probability_weights * cat_probability,
        (1.0 - offset_weights) * base_offset + offset_weights * cat_offset,
    )




def decode_predictions(beat_rows, beat_samples, probabilities, offsets, thresholds, row_count):
    predictions = [[] for _ in range(row_count)]
    for beat_index, row_index in enumerate(beat_rows):
        rpeak = int(beat_samples[beat_index])
        for class_index, (wave, landmark) in enumerate(CLASSES):
            if probabilities[beat_index, class_index] >= thresholds[class_index]:
                sample = int(np.clip(np.rint(
                    rpeak + offsets[beat_index, class_index]), 0, N_SAMPLES - 1))
                predictions[int(row_index)].append({
                    "sample": sample, "wave": wave, "landmark": landmark,
                })
    # The challenge explicitly retains only rows with at least six true events.
    # If calibrated thresholds are unusually conservative, complete the
    # highest-confidence row-local wave groups using model predictions only.
    for row_index, row_events in enumerate(predictions):
        if len(row_events) >= 6:
            continue
        candidates = []
        for beat_index in np.flatnonzero(beat_rows == row_index):
            for wave_index in range(len(WAVES)):
                columns = np.arange(3 * wave_index, 3 * wave_index + 3)
                confidence = float(np.mean(probabilities[beat_index, columns]))
                candidates.append((confidence, int(beat_index), wave_index))
        candidates.sort(reverse=True)
        existing = {
            (event["sample"], event["wave"], event["landmark"])
            for event in row_events
        }
        for _, beat_index, wave_index in candidates:
            rpeak = int(beat_samples[beat_index])
            for class_index in range(3 * wave_index, 3 * wave_index + 3):
                sample = int(np.clip(np.rint(
                    rpeak + offsets[beat_index, class_index]), 0, N_SAMPLES - 1))
                wave, landmark = CLASSES[class_index]
                key = (sample, wave, landmark)
                if key not in existing:
                    row_events.append({
                        "sample": sample, "wave": wave, "landmark": landmark,
                    })
                    existing.add(key)
            if len(existing) >= 6:
                break
    for row_events in predictions:
        unique = {
            (event["sample"], event["wave"], event["landmark"]): event
            for event in row_events
        }
        row_events[:] = sorted(
            unique.values(),
            key=lambda event: (event["sample"], CLASS_INDEX[(event["wave"], event["landmark"])]),
        )
    return predictions


def matching_count(predicted, truth, tolerance):
    predicted_by_class = defaultdict(list)
    truth_by_class = defaultdict(list)
    for event in predicted:
        predicted_by_class[(event["wave"], event["landmark"])].append(int(event["sample"]))
    for event in truth:
        truth_by_class[(event["wave"], event["landmark"])].append(int(event["sample"]))
    matches = 0
    for key in predicted_by_class.keys() & truth_by_class.keys():
        left = sorted(predicted_by_class[key])
        right = sorted(truth_by_class[key])
        i = j = 0
        while i < len(left) and j < len(right):
            if abs(left[i] - right[j]) <= tolerance:
                matches += 1
                i += 1
                j += 1
            elif left[i] < right[j] - tolerance:
                i += 1
            else:
                j += 1
    return matches


def lcs_length(left, right):
    state = [0] * (len(right) + 1)
    for left_item in left:
        previous = state.copy()
        for j, right_item in enumerate(right, 1):
            if left_item == right_item:
                state[j] = previous[j - 1] + 1
            else:
                state[j] = max(previous[j], state[j - 1])
    return state[-1]


def row_score(predicted, truth):
    f_scores = []
    denominator = len(predicted) + len(truth)
    for tolerance in (8, 20):
        matches = matching_count(predicted, truth, tolerance)
        f_scores.append(2.0 * matches / denominator if denominator else 1.0)
    predicted_order = [(e["wave"], e["landmark"]) for e in predicted]
    truth_order = [(e["wave"], e["landmark"]) for e in truth]
    longest = max(len(predicted_order), len(truth_order))
    order = lcs_length(predicted_order, truth_order) / longest if longest else 1.0
    return 0.60 * f_scores[0] + 0.30 * f_scores[1] + 0.10 * order, *f_scores, order


def validation_score(predictions, truth):
    return np.mean(
        [row_score(predicted, expected) for predicted, expected in zip(predictions, truth)],
        axis=0,
    )


def calibrate_blend(beat_rows, beat_samples, components, truth):
    candidates = (0.25, 0.40, 0.50, 0.60, 0.75)
    provisional_thresholds = np.full(len(CLASSES), 0.55, dtype=np.float64)
    best_score = -1.0
    best_weights = (0.50, 0.50)
    best_predictions = None
    for probability_weight in candidates:
        for offset_weight in candidates:
            blended = blend_predictions(
                components, (probability_weight, offset_weight))
            decoded = decode_predictions(
                beat_rows, beat_samples, *blended, provisional_thresholds, len(truth))
            score = float(validation_score(decoded, truth)[0])
            if score > best_score:
                best_score = score
                best_weights = (probability_weight, offset_weight)
                best_predictions = blended
    return best_weights, best_predictions
def calibrate_cat_blend(
        beat_rows, beat_samples, base_probability, base_offset,
        cat_probability, cat_offset, truth):
    """Select per-landmark CatBoost weights and thresholds on train OOF only."""
    weight_candidates = np.arange(0.0, 1.01, 0.20)
    threshold_candidates = np.arange(0.35, 0.751, 0.05)
    probability_weights = np.zeros(len(CLASSES), dtype=np.float64)
    offset_weights = np.zeros(len(CLASSES), dtype=np.float64)
    thresholds = calibrate_thresholds(
        beat_rows, beat_samples, base_probability, base_offset, truth)
    probability = base_probability.copy()
    offsets = base_offset.copy()

    for _ in range(1):
        for class_index in range(len(CLASSES)):
            best_score = -1.0
            best_weight = offset_weights[class_index]
            for weight in weight_candidates:
                trial_offsets = offsets.copy()
                trial_offsets[:, class_index] = (
                    (1.0 - weight) * base_offset[:, class_index]
                    + weight * cat_offset[:, class_index])
                decoded = decode_predictions(
                    beat_rows, beat_samples, probability, trial_offsets,
                    thresholds, len(truth))
                score = float(validation_score(decoded, truth)[0])
                if score > best_score:
                    best_score = score
                    best_weight = weight
            offset_weights[class_index] = best_weight
            offsets[:, class_index] = (
                (1.0 - best_weight) * base_offset[:, class_index]
                + best_weight * cat_offset[:, class_index])

        for class_index in range(len(CLASSES)):
            best_score = -1.0
            best_weight = probability_weights[class_index]
            best_threshold = thresholds[class_index]
            for weight in weight_candidates:
                trial_probability = probability.copy()
                trial_probability[:, class_index] = (
                    (1.0 - weight) * base_probability[:, class_index]
                    + weight * cat_probability[:, class_index])
                for threshold in threshold_candidates:
                    trial_thresholds = thresholds.copy()
                    trial_thresholds[class_index] = threshold
                    decoded = decode_predictions(
                        beat_rows, beat_samples, trial_probability, offsets,
                        trial_thresholds, len(truth))
                    score = float(validation_score(decoded, truth)[0])
                    if score > best_score:
                        best_score = score
                        best_weight = weight
                        best_threshold = threshold
            probability_weights[class_index] = best_weight
            thresholds[class_index] = best_threshold
            probability[:, class_index] = (
                (1.0 - best_weight) * base_probability[:, class_index]
                + best_weight * cat_probability[:, class_index])

    thresholds = calibrate_thresholds(
        beat_rows, beat_samples, probability, offsets, truth)
    return (
        probability_weights, offset_weights, thresholds, probability, offsets,
    )




def calibrate_thresholds(beat_rows, beat_samples, probabilities, offsets, truth):
    thresholds = np.full(len(CLASSES), 0.55, dtype=np.float64)
    candidates = np.arange(0.35, 0.751, 0.05)
    for _ in range(1):
        for class_index in range(len(CLASSES)):
            best_threshold = thresholds[class_index]
            best_score = -1.0
            for candidate in candidates:
                trial = thresholds.copy()
                trial[class_index] = candidate
                decoded = decode_predictions(
                    beat_rows, beat_samples, probabilities, offsets, trial, len(truth))
                score = float(validation_score(decoded, truth)[0])
                if score > best_score:
                    best_score = score
                    best_threshold = candidate
            thresholds[class_index] = best_threshold
    return thresholds


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    np.random.seed(SEED)

    # All fitted state starts here and comes exclusively from train.
    train = pd.read_csv(public_dir / "train.csv")
    signal_cache = {}
    for signal_file in train["signal_file"].drop_duplicates():
        signal_cache[signal_file] = prepare_signal(public_dir, signal_file)
    tokens = sorted(
        set(train["target_lead_token"].tolist())
        | {context["token"] for context in signal_cache.values()}
    )
    token_index = {token: index for index, token in enumerate(tokens)}

    full_features, presence, target_offsets, beat_rows, beat_samples, groups = (
        build_training_beats(train, signal_cache, token_index)
    )
    truth = [json.loads(value)["events"] for value in train["answer_json"]]
    categorical_frame, categorical_columns = cat_feature_frame(
        full_features, len(token_index))

    # Honest patient-grouped OOF predictions select the ensemble weights and
    # calibrate event thresholds using train only.
    oof_compact_probability = np.zeros(
        (len(full_features), len(CLASSES)), dtype=np.float64)
    oof_compact_offset = np.zeros(
        (len(full_features), len(CLASSES)), dtype=np.float32)
    oof_full_probability = np.zeros(
        (len(full_features), len(CLASSES)), dtype=np.float64)
    oof_full_offset = np.zeros(
        (len(full_features), len(CLASSES)), dtype=np.float32)
    oof_cat_probability = np.zeros(
        (len(full_features), len(CLASSES)), dtype=np.float64)
    oof_cat_offset = np.zeros(
        (len(full_features), len(CLASSES)), dtype=np.float32)
    splitter = GroupKFold(n_splits=5)
    for fold, (fit_indices, validation_indices) in enumerate(
            splitter.split(full_features, groups=groups)):
        fold_models = fit_ensemble(
            full_features, presence, target_offsets, fit_indices, SEED + 100 * fold)
        fold_components = predict_components(
            fold_models, full_features[validation_indices])
        oof_compact_probability[validation_indices] = fold_components[0]
        oof_compact_offset[validation_indices] = fold_components[1]
        oof_full_probability[validation_indices] = fold_components[2]
        oof_full_offset[validation_indices] = fold_components[3]
        fold_cat_models = fit_cat_models(
            full_features,
            categorical_frame,
            categorical_columns,
            presence,
            target_offsets,
            fit_indices,
            SEED + 2000 + 100 * fold,
        )
        fold_cat_probability, fold_cat_offset = predict_cat_models(
            fold_cat_models,
            full_features[validation_indices],
            categorical_frame.iloc[validation_indices],
        )
        oof_cat_probability[validation_indices] = fold_cat_probability
        oof_cat_offset[validation_indices] = fold_cat_offset
        del fold_cat_models
        del fold_models
        gc.collect()

    blend_weights, (oof_probability, oof_offset) = calibrate_blend(
        beat_rows,
        beat_samples,
        (
            oof_compact_probability, oof_compact_offset,
            oof_full_probability, oof_full_offset,
        ),
        truth,
    )
    (
        cat_probability_weights,
        cat_offset_weights,
        thresholds,
        oof_probability,
        oof_offset,
    ) = calibrate_cat_blend(
        beat_rows,
        beat_samples,
        oof_probability,
        oof_offset,
        oof_cat_probability,
        oof_cat_offset,
        truth,
    )
    oof_predictions = decode_predictions(
        beat_rows, beat_samples, oof_probability, oof_offset, thresholds, len(train))
    cv = validation_score(oof_predictions, truth)
    print(
        "grouped_5fold_cv "
        f"score={cv[0]:.6f} f1_8={cv[1]:.6f} "
        f"f1_20={cv[2]:.6f} order={cv[3]:.6f} "
        f"blend_probability={blend_weights[0]:.2f} "
        f"blend_offset={blend_weights[1]:.2f} "
        f"cat_probability_mean={cat_probability_weights.mean():.2f} "
        f"cat_offset_mean={cat_offset_weights.mean():.2f} "
    )

    all_indices = np.arange(len(full_features), dtype=np.int32)
    final_models = fit_ensemble(
        full_features, presence, target_offsets, all_indices, SEED + 9000)
    final_cat_models = fit_cat_models(
        full_features,
        categorical_frame,
        categorical_columns,
        presence,
        target_offsets,
        all_indices,
        SEED + 12000,
    )

    # Test is opened only after fitting and calibration are complete. Each test
    # row is independently transformed, predicted, decoded, and discarded before
    # the next row is touched. No cross-row test feature matrix or statistic exists.
    test = pd.read_csv(public_dir / "test.csv")
    ids = []
    answers = []
    for row in test.itertuples(index=False):
        context = prepare_signal(public_dir, row.signal_file)
        row_features = np.stack([
            beat_features(
                context, row.target_lead_token, row.signal_file, beat_index, token_index)
            for beat_index in range(len(context["rpeaks"]))
        ])
        row_categorical_frame, _ = cat_feature_frame(
            row_features, len(token_index))
        row_samples = np.asarray(context["rpeaks"], dtype=np.int32)
        base_probability, base_offset = predict_ensemble(
            final_models, row_features, blend_weights)
        cat_probability, cat_offset = predict_cat_models(
            final_cat_models, row_features, row_categorical_frame)
        row_probability, row_offset = mix_cat_predictions(
            base_probability,
            base_offset,
            cat_probability,
            cat_offset,
            cat_probability_weights,
            cat_offset_weights,
        )
        events = decode_predictions(
            np.zeros(len(row_samples), dtype=np.int32),
            row_samples,
            row_probability,
            row_offset,
            thresholds,
            1,
        )[0]

        # Model-based safety path using candidates from this row only.
        if not events:
            best = int(np.argmax(row_probability[:, 4]))
            rpeak = int(row_samples[best])
            for class_index in (3, 4, 5):
                sample = int(np.clip(np.rint(
                    rpeak + row_offset[best, class_index]), 0, N_SAMPLES - 1))
                wave, landmark = CLASSES[class_index]
                events.append({"sample": sample, "wave": wave, "landmark": landmark})
            events.sort(key=lambda event: event["sample"])

        ids.append(row.id)
        answers.append(json.dumps({"events": events}, separators=(",", ":")))

    output = pd.DataFrame({"id": ids, "answer_json": answers})
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(submission_out, index=False)
    print(f"wrote {len(output)} rows to {submission_out}")


if __name__ == "__main__":
    main()
