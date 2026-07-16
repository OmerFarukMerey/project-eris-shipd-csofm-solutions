#!/usr/bin/env python3
"""Candidate-anchored historical OCR verification and repair.

The solver learns the released corruption channel and a character language model,
enumerates only one/two-edit repairs of the supplied candidate, and ranks them.
A candidate-conditioned visual matcher is added below the language baseline.
"""
from __future__ import annotations

import sys
import difflib
import heapq
import math
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SEED = 20260714
TYPES = ["none", "substitution", "deletion", "insertion", "spacing", "punctuation", "diacritic", "mixed"]
WORD_WEIGHTS = {
    "none": 0.0, "substitution": 0.5, "deletion": 1.0, "insertion": 0.1,
    "spacing": 0.0, "punctuation": 0.1, "diacritic": 0.5, "mixed": 0.3,
}
BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset" / "public"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def edit_ops(candidate: str, corrected: str):
    sm = difflib.SequenceMatcher(None, candidate, corrected, autojunk=False)
    return [(tag, candidate[i1:i2], corrected[j1:j2], i1)
            for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


class CharLM:
    """Interpolated 2--5 gram model over trusted released transcriptions."""
    def __init__(self):
        self.ng = {n: Counter() for n in range(2, 6)}
        self.ctx = {n: Counter() for n in range(2, 6)}
        self.vocab = 128
        self.cache = {}

    def fit(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
            z = "^^^^" + text + "$$$$"
            for n in range(2, 6):
                self.ng[n].update(z[i:i+n] for i in range(len(z) - n + 1))
                self.ctx[n].update(z[i:i+n-1] for i in range(len(z) - n + 1))
        self.vocab = max(32, len(chars) + 8)
        self.cache.clear()
        return self

    def _contribution(self, z: str, i: int) -> float:
        prob = 0.0
        for n, weight in zip(range(2, 6), (0.10, 0.15, 0.25, 0.50)):
            gram = z[i-n+1:i+1]
            context = gram[:-1]
            prob += weight * (self.ng[n][gram] + 0.12) / (self.ctx[n][context] + 0.12 * self.vocab)
        return math.log(prob + 1e-15)

    def score(self, text: str) -> float:
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        z = "^^^^" + text + "$$$$"
        total = sum(self._contribution(z, i) for i in range(4, len(z)))
        if len(self.cache) < 200_000:
            self.cache[text] = total
        return total

    def edit_score(self, text: str, hypothesis: str, op) -> float:
        """Score a one-edit hypothesis by recomputing only affected n-grams."""
        if op is None:
            return self.score(text)
        if isinstance(op, tuple) and len(op) == 2 and isinstance(op[0], tuple):
            return self.score(hypothesis)
        _, wrong, right, pos = op
        old_z = "^^^^" + text + "$$$$"
        new_z = "^^^^" + hypothesis + "$$$$"
        old_start = 4 + pos
        new_start = 4 + pos
        old_stop = min(len(old_z), old_start + len(wrong) + 4)
        new_stop = min(len(new_z), new_start + len(right) + 4)
        removed = sum(self._contribution(old_z, i) for i in range(old_start, old_stop))
        added = sum(self._contribution(new_z, i) for i in range(new_start, new_stop))
        return self.score(text) - removed + added


class WordLM:
    """Released-corpus word unigram/bigram evidence for local reranking."""
    def __init__(self):
        self.words = Counter()
        self.bigrams = Counter()
        self.contexts = Counter()
        self.total = 0
        self.vocab = 10_000
        self.cache = {}

    @staticmethod
    def tokenize(text):
        return re.findall(r"[^\W_]+(?:[’'-][^\W_]+)*|[^\w\s]", text, flags=re.UNICODE)

    def fit(self, texts):
        for text in texts:
            tokens = ["<s>"] + self.tokenize(text) + ["</s>"]
            lexical = [token for token in tokens[1:-1] if any(c.isalnum() for c in token)]
            self.words.update(lexical)
            self.bigrams.update(zip(tokens[:-1], tokens[1:]))
            self.contexts.update(tokens[:-1])
        self.total = sum(self.words.values())
        self.vocab = max(10_000, len(self.words))
        self.cache.clear()
        return self

    def score(self, text):
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        tokens = self.tokenize(text)
        value = 0.0
        for token in tokens:
            if any(c.isalnum() for c in token):
                value += math.log((self.words[token] + .08) /
                                  (self.total + .08*self.vocab))
        sequence = ["<s>"] + tokens + ["</s>"]
        for left, right in zip(sequence[:-1], sequence[1:]):
            value += .18*math.log((self.bigrams[(left, right)] + .05) /
                                  (self.contexts[left] + .05*self.vocab))
        if len(self.cache) < 300_000:
            self.cache[text] = value
        return value


class CorruptionChannel:
    """Observed candidate->truth edit inventory; never generates a free transcript."""
    def __init__(self):
        self.maps = defaultdict(Counter)
        self.type_ops = defaultdict(Counter)
        self.mixed_pairs = Counter()

    def fit(self, frame: pd.DataFrame):
        for row in frame.itertuples(index=False):
            ops = edit_ops(row.candidate_text, row.corrected_text)
            kinds = []
            for tag, wrong, right, _ in ops:
                self.maps[(row.error_type, tag, wrong)][right] += 1
                self.type_ops[row.error_type][(tag, wrong, right)] += 1
                kinds.append(self.semantic_type(tag, wrong, right))
            if row.error_type == "mixed" and len(kinds) >= 2:
                self.mixed_pairs[tuple(sorted(kinds[:2]))] += 1
        return self

    @staticmethod
    def semantic_type(tag: str, wrong: str, right: str) -> str:
        punct = set(".,:;!?—–-’'\"„“”«»()[]")
        if wrong == " " or right == " ":
            return "spacing"
        if (wrong and all(c in punct for c in wrong)) or (right and all(c in punct for c in right)):
            return "punctuation"
        stripped_wrong = "".join(c for c in unicodedata.normalize("NFD", wrong) if not unicodedata.combining(c))
        stripped_right = "".join(c for c in unicodedata.normalize("NFD", right) if not unicodedata.combining(c))
        if wrong != right and stripped_wrong == stripped_right:
            return "diacritic"
        return {"replace": "substitution", "insert": "deletion", "delete": "insertion"}.get(tag, "substitution")

    def _push(self, heap, seen, text, op, score, topk):
        if not text or text in seen:
            return
        seen.add(text)
        item = (score, text, op)
        if len(heap) < topk:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)

    def candidates(self, text: str, typ: str, lm: CharLM, topk: int = 4):
        if typ == "none":
            return [(lm.score(text), text, None)]
        heap, seen = [], set()
        entries = []
        if typ in ("substitution", "diacritic"):
            for (t, tag, wrong), rights in self.maps.items():
                if t != typ or not wrong:
                    continue
                total = sum(rights.values())
                start = 0
                while True:
                    pos = text.find(wrong, start)
                    if pos < 0:
                        break
                    for right, count in rights.items():
                        hypothesis = text[:pos] + right + text[pos+len(wrong):]
                        prior = math.log((count + .2) / (total + .2 * len(rights)))
                        entries.append((hypothesis, (tag, wrong, right, pos), prior))
                    start = pos + 1
            # Rare combining-mark insertion/deletion cases.
            for (t, tag, wrong), rights in self.maps.items():
                if t != typ or tag == "replace":
                    continue
                if tag == "delete" and wrong:
                    for pos in range(len(text)):
                        if text.startswith(wrong, pos):
                            entries.append((text[:pos] + text[pos+len(wrong):], (tag, wrong, "", pos), 0.0))
                elif tag == "insert":
                    for pos in range(len(text) + 1):
                        for right, count in rights.items():
                            entries.append((text[:pos] + right + text[pos:], (tag, "", right, pos), math.log(count + 1)))
        elif typ == "deletion":
            chars = Counter()
            for (t, tag, _), rights in self.maps.items():
                if t == typ and tag == "insert":
                    chars.update(rights)
            total = sum(chars.values())
            for pos in range(len(text) + 1):
                for right, count in chars.items():
                    entries.append((text[:pos] + right + text[pos:], ("insert", "", right, pos), math.log((count + .2)/(total + .2*len(chars)))))
        elif typ == "insertion":
            wrongs = Counter()
            for (t, tag, wrong), rights in self.maps.items():
                if t == typ and tag == "delete":
                    wrongs[wrong] += sum(rights.values())
            total = sum(wrongs.values())
            for pos in range(len(text)):
                for wrong, count in wrongs.items():
                    if wrong and text.startswith(wrong, pos):
                        entries.append((text[:pos] + text[pos+len(wrong):], ("delete", wrong, "", pos), math.log((count+.2)/(total+.2*len(wrongs)))))
        elif typ in ("spacing", "punctuation"):
            for (t, tag, wrong), rights in self.maps.items():
                if t != typ:
                    continue
                total = sum(rights.values())
                if tag == "insert":
                    for pos in range(len(text) + 1):
                        for right, count in rights.items():
                            entries.append((text[:pos] + right + text[pos:], (tag, "", right, pos), math.log((count+.2)/(total+.2*len(rights)))))
                elif wrong:
                    for pos in range(len(text)):
                        if text.startswith(wrong, pos):
                            for right, count in rights.items():
                                entries.append((text[:pos] + right + text[pos+len(wrong):], (tag, wrong, right, pos), math.log((count+.2)/(total+.2*len(rights)))))
        elif typ == "mixed":
            # Mixed feature generation is deferred: composing all possible pairs
            # is quadratic. The two strongest distinct single-edit signals below
            # are used to detect mixed rows; repairs are composed only if selected.
            return []
        for hypothesis, op, prior in entries:
            score = lm.edit_score(text, hypothesis, op) + 0.12 * prior
            self._push(heap, seen, hypothesis, op, score, topk)
        return sorted(heap, reverse=True, key=lambda x: x[0])


def operation_features(op, text_len):
    if op is None:
        return [0.0] * 8
    if isinstance(op, tuple) and len(op) == 2 and isinstance(op[0], tuple):
        a = operation_features(op[0], text_len)
        b = operation_features(op[1], text_len)
        return [(x+y)/2 for x, y in zip(a, b)]
    tag, wrong, right, pos = op
    return [pos/max(1, text_len), float(pos == 0), float(pos >= text_len-1),
            len(wrong), len(right), float(" " in wrong+right),
            float(any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", wrong+right))),
            {"replace": 1, "insert": 2, "delete": 3}.get(tag, 0)]


def compose_mixed_options(text, row_hyp, lm, topk):
    """Compose two non-overlapping edits from distinct documented categories."""
    heap, seen = [], set()
    for first_index, first_type in enumerate(TYPES[1:7]):
        for second_type in TYPES[first_index+2:7]:
            for _, _, op1 in row_hyp.get(first_type, [])[:2]:
                for _, _, op2 in row_hyp.get(second_type, [])[:2]:
                    if op1 is None or op2 is None:
                        continue
                    _, wrong1, right1, pos1 = op1
                    _, wrong2, right2, pos2 = op2
                    end1 = pos1 + max(1, len(wrong1))
                    end2 = pos2 + max(1, len(wrong2))
                    if not (end1 <= pos2 or end2 <= pos1):
                        continue
                    hypothesis = text
                    valid = True
                    for _, wrong, right, pos in sorted((op1, op2), key=lambda x: x[3], reverse=True):
                        if wrong and hypothesis[pos:pos+len(wrong)] != wrong:
                            valid = False
                            break
                        hypothesis = hypothesis[:pos] + right + hypothesis[pos+len(wrong):]
                    if not valid or not hypothesis or hypothesis in seen:
                        continue
                    seen.add(hypothesis)
                    item = (lm.score(hypothesis), hypothesis, (op1, op2))
                    if len(heap) < topk:
                        heapq.heappush(heap, item)
                    elif item[0] > heap[0][0]:
                        heapq.heapreplace(heap, item)
    return sorted(heap, reverse=True, key=lambda x: x[0])


def make_features(frame, lm, channel, word_lm=None, topk=8, progress=False):
    records, hypotheses = [], []
    for number, row in enumerate(frame.itertuples(index=False), 1):
        base_score = lm.score(row.candidate_text)
        base_word = word_lm.score(row.candidate_text) if word_lm else 0.0
        rec = {
            "lm_per_char": base_score/max(1, len(row.candidate_text)),
            "word_per_char": base_word/max(1, len(row.candidate_text)),
            "length": len(row.candidate_text),
            "spaces": row.candidate_text.count(" "),
            "punct": sum(not c.isalnum() and not c.isspace() for c in row.candidate_text),
            "nonascii": sum(ord(c) > 127 for c in row.candidate_text),
            "upper": sum(c.isupper() for c in row.candidate_text),
            "digits": sum(c.isdigit() for c in row.candidate_text),
        }
        row_hyp = {}
        for typ in TYPES[:7]:
            type_topk = topk + 6 if typ == "deletion" else topk
            raw_options = channel.candidates(row.candidate_text, typ, lm, topk=type_topk)
            weight = WORD_WEIGHTS[typ] if word_lm else 0.0
            options = sorted(
                [(score + weight*word_lm.score(text) if word_lm else score, text, op)
                 for score, text, op in raw_options],
                reverse=True, key=lambda item: item[0],
            )
            row_hyp[typ] = options
            if options:
                score, best_text, op = options[0]
                type_base = base_score + weight*base_word
                rec[f"delta_{typ}"] = score - type_base
                rec[f"char_delta_{typ}"] = lm.score(best_text) - base_score
                rec[f"word_delta_{typ}"] = ((word_lm.score(best_text)-base_word) if word_lm else 0.0)
                rec[f"margin_{typ}"] = score - options[1][0] if len(options) > 1 else 0.0
                rec[f"available_{typ}"] = len(options)
                for j, value in enumerate(operation_features(op, len(row.candidate_text))):
                    rec[f"op{j}_{typ}"] = value
            else:
                rec[f"delta_{typ}"] = -50.0
                rec[f"char_delta_{typ}"] = -50.0
                rec[f"word_delta_{typ}"] = -50.0
                rec[f"margin_{typ}"] = 0.0
                rec[f"available_{typ}"] = 0
                for j in range(8):
                    rec[f"op{j}_{typ}"] = 0.0
        mixed_raw = compose_mixed_options(row.candidate_text, row_hyp, lm, topk)
        mixed_weight = WORD_WEIGHTS["mixed"] if word_lm else 0.0
        mixed = sorted(
            [(score + mixed_weight*word_lm.score(text) if word_lm else score, text, op)
             for score, text, op in mixed_raw],
            reverse=True, key=lambda item: item[0],
        )
        row_hyp["mixed"] = mixed
        if mixed:
            score, best_text, op = mixed[0]
            rec["delta_mixed"] = score - (base_score + mixed_weight*base_word)
            rec["char_delta_mixed"] = lm.score(best_text) - base_score
            rec["word_delta_mixed"] = ((word_lm.score(best_text)-base_word) if word_lm else 0.0)
            rec["margin_mixed"] = score - mixed[1][0] if len(mixed) > 1 else 0.0
            rec["available_mixed"] = len(mixed)
            for j, value in enumerate(operation_features(op, len(row.candidate_text))):
                rec[f"op{j}_mixed"] = value
        else:
            rec["delta_mixed"], rec["char_delta_mixed"], rec["word_delta_mixed"] = -50.0, -50.0, -50.0
            rec["margin_mixed"], rec["available_mixed"] = 0.0, 0
            for j in range(8):
                rec[f"op{j}_mixed"] = 0.0
        records.append(rec)
        hypotheses.append(row_hyp)
        if progress and number % 500 == 0:
            print(f"  generated {number}/{len(frame)} rows", flush=True)
    return pd.DataFrame(records).replace([np.inf, -np.inf], [-50, -50]), hypotheses


class BinaryTypeEnsemble:
    """Independent rare-class detectors with a common quota assignment layer."""
    def __init__(self, models):
        self.models = models
        self.classes_ = np.asarray(TYPES)

    def predict_proba(self, features):
        return np.column_stack([model.predict_proba(features)[:, 1] for model in self.models])


def fit_binary_type_ensemble(features, labels, sample_weights):
    models = []
    labels = np.asarray(labels)
    for index, typ in enumerate(TYPES):
        model = CatBoostClassifier(
            iterations=450, depth=6, learning_rate=.06, loss_function="Logloss",
            random_seed=SEED+index, verbose=False, thread_count=10, l2_leaf_reg=6,
            auto_class_weights="SqrtBalanced", allow_writing_files=False,
        )
        model.fit(features, (labels == typ).astype(int), sample_weight=sample_weights)
        models.append(model)
    return BinaryTypeEnsemble(models)


def quota_type_predictions(classifier, features):
    """Calibrate to the released generation mix, then satisfy integer quotas."""
    probabilities = np.clip(classifier.predict_proba(features), 1e-9, 1)
    classes = [str(x) for x in classifier.classes_]
    proportions = {
        "none": .40, "substitution": .18, "deletion": .10, "insertion": .10,
        "spacing": .08, "punctuation": .07, "diacritic": .04, "mixed": .03,
    }
    raw_targets = np.asarray([proportions[c]*len(features) for c in classes])
    targets = np.floor(raw_targets).astype(int)
    for index in np.argsort(-(raw_targets-targets))[:len(features)-targets.sum()]:
        targets[index] += 1
    logp = np.log(probabilities)
    bias = np.zeros(len(classes))
    best_pred, best_error = None, 10**9
    for _ in range(500):
        pred = np.argmax(logp+bias, axis=1)
        counts = np.bincount(pred, minlength=len(classes))
        error = int(np.abs(counts-targets).sum())
        if error < best_error:
            best_pred, best_error = pred.copy(), error
        if error == 0:
            break
        bias += .08*np.log((targets+.5)/(counts+.5))
    pred = best_pred
    counts = np.bincount(pred, minlength=len(classes))
    # Bias iteration is usually exact. Repair any residual with the least-cost moves.
    while np.any(counts != targets):
        over = np.where(counts > targets)[0]
        under = np.where(counts < targets)[0]
        best = None
        for destination in under:
            rows = np.where(np.isin(pred, over))[0]
            gains = logp[rows, destination] - logp[rows, pred[rows]]
            position = int(np.argmax(gains))
            candidate = (float(gains[position]), int(rows[position]), int(destination))
            if best is None or candidate[0] > best[0]:
                best = candidate
        _, row, destination = best
        counts[pred[row]] -= 1
        pred[row] = destination
        counts[destination] += 1
    return [classes[i] for i in pred]


def predict_rows(classifier, features, hypotheses, candidates, visual_rows=None,
                 visual_weight=0.35, enforce_quotas=False):
    pred_type = (quota_type_predictions(classifier, features) if enforce_quotas else
                 [str(x) for x in classifier.predict(features).reshape(-1)])
    corrected = []
    for row_index, (typ, row_hyp, candidate) in enumerate(zip(pred_type, hypotheses, candidates)):
        options = row_hyp.get(typ, [])
        if typ == "none" or not options:
            corrected.append(candidate)
            continue
        if visual_rows is None:
            choice = 0
        else:
            visual = visual_rows[row_index][typ]
            base_lm = options[0][0]
            utilities = [visual[i] + visual_weight*(option[0]-base_lm)
                         for i, option in enumerate(options)]
            choice = int(np.argmax(utilities))
        corrected.append(options[choice][1])
    return pred_type, corrected


def load_line_image(path: Path, target_height: int = 32) -> torch.Tensor:
    """Contrast-normalize without recognizing text, then preserve aspect ratio."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape
    new_width = max(32, min(960, int(round(width * target_height / height))))
    image = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    low, high = np.percentile(image, (2, 98))
    darkness = np.clip((high - image.astype(np.float32)) / max(12.0, high-low), 0, 1)
    return torch.from_numpy(darkness[None])


def pad_images(images):
    widths = torch.tensor([x.shape[-1] for x in images], dtype=torch.long)
    out = torch.zeros(len(images), 1, images[0].shape[-2], int(widths.max()), dtype=torch.float32)
    for i, image in enumerate(images):
        out[i, :, :, :image.shape[-1]] = image
    return out, widths


def mutate_correct_text(text: str, rng: random.Random, reverse_confusions) -> str:
    """One candidate-like corruption for visually matched negative training."""
    if len(text) < 2:
        return text + "e"
    action = rng.choices(("sub", "drop", "add", "space", "punct"), (35, 18, 18, 16, 13))[0]
    positions = [i for i, c in enumerate(text) if not c.isspace()]
    if action == "sub" and positions:
        valid = [i for i in positions if reverse_confusions.get(text[i])]
        if valid:
            pos = rng.choice(valid)
            wrong = rng.choice(reverse_confusions[text[pos]])
            return text[:pos] + wrong + text[pos+1:]
    if action == "drop" and positions:
        pos = rng.choice(positions)
        return text[:pos] + text[pos+1:]
    if action == "add":
        pos = rng.randrange(len(text)+1)
        char = rng.choice("eeeennrrttiiaasslloucdfm")
        return text[:pos] + char + text[pos:]
    if action == "space":
        spaces = [i for i, c in enumerate(text) if c == " "]
        if spaces and rng.random() < .75:
            pos = rng.choice(spaces)
            return text[:pos] + text[pos+1:]
        pos = rng.randrange(1, len(text))
        return text[:pos] + " " + text[pos:]
    punct = [i for i, c in enumerate(text) if c in ".,:;!?—-’"]
    if punct and rng.random() < .6:
        pos = rng.choice(punct)
        return text[:pos] + text[pos+1:]
    pos = rng.randrange(len(text)+1)
    return text[:pos] + rng.choice(".,;:—-") + text[pos:]


class VisualPairDataset(Dataset):
    """Positive truth, actual candidate negative, and a one-edit hard negative."""
    def __init__(self, frame, data_root, channel, augment=True):
        self.frame = frame.reset_index(drop=True)
        self.root = data_root
        self.augment = augment
        self.epoch = 0
        self.reverse = defaultdict(list)
        for (typ, tag, wrong), rights in channel.maps.items():
            if typ == "substitution" and tag == "replace" and len(wrong) == 1:
                for right in rights:
                    if len(right) == 1 and wrong != right:
                        self.reverse[right].append(wrong)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        positive = row.corrected_text
        rng = random.Random(SEED + index * 1009 + self.epoch * 7919)
        if row.candidate_text != positive:
            negative = row.candidate_text
        else:
            negative = mutate_correct_text(positive, rng, self.reverse)
        hard = mutate_correct_text(positive, rng, self.reverse)
        image = load_line_image(self.root / row.image)
        if self.augment:
            gain = .90 + .2*rng.random()
            image = (image*gain + .01*torch.randn(image.shape,
                     generator=torch.Generator().manual_seed(rng.randrange(1 << 31)))).clamp(0, 1)
        return image, positive, negative, hard


def pair_collate(batch):
    images, positive, negative, hard = zip(*batch)
    padded, widths = pad_images(images)
    return padded, widths, list(positive), list(negative), list(hard)



class CandidateVisualMatcher(nn.Module):
    """Scores a supplied text hypothesis against a line image.

    It has no decoder and cannot emit an unrestricted transcript. Character
    locations are derived from the supplied hypothesis, making this a
    candidate-conditioned monotonic alignment model.
    """
    def __init__(self, chars):
        super().__init__()
        self.chars = sorted(set(chars))
        self.char_to_id = {c: i+1 for i, c in enumerate(self.chars)}
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, 5, stride=2, padding=2), nn.GroupNorm(3, 12), nn.SiLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.GroupNorm(6, 24), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=(2, 1), padding=1), nn.GroupNorm(6, 48), nn.SiLU(),
            nn.Conv2d(48, 48, 3, stride=1, padding=1), nn.GroupNorm(6, 48), nn.SiLU(),
        )
        self.embedding = nn.Embedding(len(self.chars)+1, 24, padding_idx=0)
        self.token = nn.Sequential(nn.Linear(48*7 + 24*3, 128), nn.SiLU(),
                                   nn.Dropout(.10), nn.Linear(128, 48), nn.SiLU(),
                                   nn.Linear(48, 1))
        self.row = nn.Sequential(nn.Linear(5, 24), nn.SiLU(), nn.Linear(24, 1))

    @staticmethod
    def char_width(char):
        if char.isspace():
            return .55
        if char in "ilI1.,:;!|'`’()[]":
            return .42
        if char in "mwMW—":
            return 1.30
        if char in "ABCDEFGHJKLMNOPQRSTUVXYZÄÖÜ":
            return 1.02
        if unicodedata.combining(char):
            return .05
        return .82

    def encode(self, images, widths):
        features = self.encoder(images).mean(dim=2)
        # Two horizontal stride-2 convolutions use ceil division.
        feature_lengths = torch.div(widths + 3, 4, rounding_mode="floor").clamp(max=features.shape[-1])
        return features, feature_lengths

    def text_batch(self, texts, device):
        length = max(len(x) for x in texts)
        ids = torch.zeros(len(texts), length, dtype=torch.long, device=device)
        centers = torch.zeros(len(texts), length, dtype=torch.float32, device=device)
        mask = torch.zeros(len(texts), length, dtype=torch.bool, device=device)
        for row, text in enumerate(texts):
            values = [self.char_to_id.get(c, 0) for c in text]
            weights = np.asarray([self.char_width(c) for c in text], np.float32)
            cumulative = np.cumsum(weights) - weights/2
            cumulative /= max(float(weights.sum()), 1e-4)
            ids[row, :len(text)] = torch.tensor(values, device=device)
            centers[row, :len(text)] = torch.tensor(cumulative, device=device)
            mask[row, :len(text)] = True
        return ids, centers, mask

    def token_scores_encoded(self, features, feature_lengths, texts):
        ids, centers, mask = self.text_batch(texts, features.device)
        batch, length = ids.shape
        base = torch.round(centers * (feature_lengths[:, None]-1)).long()
        offsets = torch.arange(-3, 4, device=features.device)
        indices = (base[:, :, None] + offsets).clamp(min=0)
        indices = torch.minimum(indices, (feature_lengths[:, None, None]-1))
        expanded = features[:, :, None, :].expand(-1, -1, length, -1)
        visual = torch.gather(expanded, 3, indices[:, None].expand(-1, features.shape[1], -1, -1))
        visual = visual.permute(0, 2, 1, 3).reshape(batch, length, -1)
        emb = self.embedding(ids)
        previous = F.pad(emb[:, :-1], (0, 0, 1, 0))
        following = F.pad(emb[:, 1:], (0, 0, 0, 1))
        token_scores = self.token(torch.cat((visual, previous, emb, following), dim=-1)).squeeze(-1)
        return token_scores, mask

    def aggregate_token_scores(self, token_scores, mask):
        masked_low = token_scores.masked_fill(~mask, 1e4)
        lengths = mask.sum(1).clamp(min=1)
        mean = token_scores.masked_fill(~mask, 0).sum(1) / lengths
        minimum = masked_low.min(1).values
        sorted_scores = masked_low.sort(1).values
        bottom3 = sorted_scores[:, :min(3, token_scores.shape[1])].mean(1)
        maximum = token_scores.masked_fill(~mask, -1e4).max(1).values
        variance = ((token_scores.masked_fill(~mask, 0)-mean[:, None])**2 * mask).sum(1)/lengths
        summary = torch.stack((mean, minimum, bottom3, maximum, torch.sqrt(variance+1e-5)), 1)
        return self.row(summary).squeeze(1)

    def score_encoded(self, features, feature_lengths, texts):
        token_scores, mask = self.token_scores_encoded(features, feature_lengths, texts)
        return self.aggregate_token_scores(token_scores, mask)

    def forward(self, images, widths, texts):
        features, lengths = self.encode(images, widths)
        return self.score_encoded(features, lengths, texts)


def localized_mismatch_mask(negative, positive, length, device):
    """Mark candidate character positions implicated by the known local edit."""
    result = torch.zeros(len(negative), length, dtype=torch.bool, device=device)
    for row, (candidate, truth) in enumerate(zip(negative, positive)):
        matcher = difflib.SequenceMatcher(None, candidate, truth, autojunk=False)
        for tag, i1, i2, _, _ in matcher.get_opcodes():
            if tag == "equal":
                continue
            if i2 > i1:
                result[row, i1:i2] = True
            else:
                if i1 > 0:
                    result[row, i1-1] = True
                if i1 < len(candidate):
                    result[row, i1] = True
        if not result[row].any() and candidate:
            result[row, min(len(candidate)-1, length-1)] = True
    return result


def train_visual_matcher(frame, channel, epochs=14):
    torch.set_num_threads(min(10, os.cpu_count() or 10))
    chars = "".join(frame.corrected_text) + "".join(frame.candidate_text)
    model = CandidateVisualMatcher(chars)
    dataset = VisualPairDataset(frame, DATA, channel)
    loader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=0,
                        collate_fn=pair_collate, generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-4)
    model.train()
    for epoch in range(epochs):
        dataset.epoch = epoch
        losses = []
        for images, widths, positive, negative, hard in loader:
            features, lengths = model.encode(images, widths)
            pos_tokens, pos_mask = model.token_scores_encoded(features, lengths, positive)
            neg_tokens, neg_mask = model.token_scores_encoded(features, lengths, negative)
            hard_tokens, hard_mask = model.token_scores_encoded(features, lengths, hard)
            pos = model.aggregate_token_scores(pos_tokens, pos_mask)
            neg = model.aggregate_token_scores(neg_tokens, neg_mask)
            hardneg = model.aggregate_token_scores(hard_tokens, hard_mask)
            mismatch = localized_mismatch_mask(
                negative, positive, neg_tokens.shape[1], neg_tokens.device
            ) & neg_mask
            hard_mismatch = localized_mismatch_mask(
                hard, positive, hard_tokens.shape[1], hard_tokens.device
            ) & hard_mask
            matching_negative = neg_mask & ~mismatch
            loss = F.softplus(neg-pos).mean() + .5*F.softplus(hardneg-pos).mean()
            loss += .12*(F.softplus(-pos).mean()
                         + .65*F.softplus(neg).mean() + .35*F.softplus(hardneg).mean())
            loss += .30*(F.softplus(neg_tokens)*mismatch).sum()/mismatch.sum().clamp(min=1)
            loss += .15*(F.softplus(hard_tokens)*hard_mismatch).sum()/hard_mismatch.sum().clamp(min=1)
            loss += .04*(F.softplus(-pos_tokens)*pos_mask).sum()/pos_mask.sum().clamp(min=1)
            loss += .02*(F.softplus(-neg_tokens)*matching_negative).sum()/matching_negative.sum().clamp(min=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        print(f"  visual epoch {epoch+1}/{epochs} loss={np.mean(losses):.4f}", flush=True)
    model.eval()
    return model


@torch.inference_mode()
def augment_visual_features(features, hypotheses, frame, model, batch_size=20):
    """Score all anchored alternatives while encoding each image only once."""
    features = features.copy()
    visual_rows = [None] * len(frame)
    for begin in range(0, len(frame), batch_size):
        end = min(len(frame), begin + batch_size)
        images = [load_line_image(DATA / path) for path in frame.image.iloc[begin:end]]
        padded, widths = pad_images(images)
        encoded, enc_lengths = model.encode(padded, widths)
        flat_texts, owners, labels = [], [], []
        for local, row_hyp in enumerate(hypotheses[begin:end]):
            for typ in TYPES:
                for option_index, (_, text, _) in enumerate(row_hyp[typ]):
                    flat_texts.append(text)
                    owners.append(local)
                    labels.append((local, typ, option_index))
        owners_t = torch.tensor(owners, dtype=torch.long)
        token_scores, token_mask = model.token_scores_encoded(
            encoded.index_select(0, owners_t), enc_lengths.index_select(0, owners_t), flat_texts)
        scores = model.aggregate_token_scores(token_scores, token_mask).cpu().numpy()
        masked_low = token_scores.masked_fill(~token_mask, 1e4)
        token_min = masked_low.min(1).values.cpu().numpy()
        token_bottom3 = masked_low.sort(1).values[:, :min(3, token_scores.shape[1])].mean(1).cpu().numpy()
        per_row = [{t: [] for t in TYPES} for _ in range(end-begin)]
        per_row_stats = [{t: [] for t in TYPES} for _ in range(end-begin)]
        for score, minimum, bottom3, (local, typ, _) in zip(scores, token_min, token_bottom3, labels):
            per_row[local][typ].append(float(score))
            per_row_stats[local][typ].append((float(minimum), float(bottom3)))
        for local, visual in enumerate(per_row):
            global_row = begin + local
            candidate_score = visual["none"][0]
            candidate_min, candidate_bottom3 = per_row_stats[local]["none"][0]
            features.loc[global_row, "vis_tokmin_none"] = candidate_min
            features.loc[global_row, "vis_tokbot3_none"] = candidate_bottom3
            for typ in TYPES:
                values = visual[typ]
                if values:
                    order = int(np.argmax(values))
                    ordered = sorted(values, reverse=True)
                    features.loc[global_row, f"vis_best_{typ}"] = ordered[0]
                    features.loc[global_row, f"vis_delta_{typ}"] = ordered[0] - candidate_score
                    features.loc[global_row, f"vis_margin_{typ}"] = ordered[0] - ordered[1] if len(ordered)>1 else 0
                    features.loc[global_row, f"vis_lmfirst_{typ}"] = values[0]
                    features.loc[global_row, f"vis_minfix_{typ}"] = per_row_stats[local][typ][order][0] - candidate_min
                else:
                    features.loc[global_row, f"vis_best_{typ}"] = -20
                    features.loc[global_row, f"vis_delta_{typ}"] = -20
                    features.loc[global_row, f"vis_margin_{typ}"] = 0
                    features.loc[global_row, f"vis_lmfirst_{typ}"] = -20
                    features.loc[global_row, f"vis_minfix_{typ}"] = 0
            visual_rows[global_row] = visual
        if end % 500 == 0 or end == len(frame):
            print(f"  visually scored {end}/{len(frame)} rows", flush=True)
    return features, visual_rows

def score_report(frame, pred_type, corrected, title="validation"):
    from sklearn.metrics import accuracy_score, f1_score
    def distance(a, b):
        previous = list(range(len(b)+1))
        for i, char_a in enumerate(a, 1):
            current = [i]
            for j, char_b in enumerate(b, 1):
                current.append(min(current[-1]+1, previous[j]+1,
                                   previous[j-1] + (char_a != char_b)))
            previous = current
        return previous[-1]
    truth_type = frame.error_type.to_numpy()
    truth_correct = frame.corrected_text.to_numpy()
    pred_type = np.asarray(pred_type)
    corrected = np.asarray(corrected)
    pred_correct = (pred_type == "none").astype(int)
    exact_text = corrected == truth_correct
    exact_row = exact_text & (pred_type == truth_type)
    correctness_f1 = f1_score(frame.is_correct, pred_correct, average="macro")
    type_f1 = f1_score(truth_type, pred_type, labels=TYPES, average="macro")
    character_rows = np.asarray([1/(1+distance(p, t))**2
                                 for p, t in zip(corrected, truth_correct)])
    balanced_character, balanced_text, balanced_row = [], [], []
    for typ in TYPES:
        mask = truth_type == typ
        if mask.any():
            balanced_character.append(character_rows[mask].mean())
            balanced_text.append(exact_text[mask].mean())
            balanced_row.append(exact_row[mask].mean())
    components = [correctness_f1, type_f1, np.mean(balanced_character),
                  np.mean(balanced_text), np.mean(balanced_row)]
    core = .15*components[0] + .20*components[1] + .20*components[2] + .25*components[3] + .20*components[4]
    final = .75*core + .25*min(components)
    print(f"{title}: type_acc={accuracy_score(truth_type,pred_type):.4f} "
          f"type_macro_f1={type_f1:.4f} correctness_f1={correctness_f1:.4f} "
          f"exact_text={exact_text.mean():.4f} exact_row={exact_row.mean():.4f} "
          f"challenge_score={final:.4f}")
    print(f"  balanced components: char={components[2]:.4f} text={components[3]:.4f} row={components[4]:.4f}; "
          f"predicted={dict(Counter(pred_type))}")
    for typ in TYPES:
        mask = truth_type == typ
        if mask.any():
            conditional = (corrected[mask] == truth_correct[mask]).mean()
            print(f"  {typ:12s} n={mask.sum():4d} type={(pred_type[mask]==typ).mean():.3f} text={conditional:.3f}")


def language_validation(train):
    from sklearn.model_selection import train_test_split
    build_idx, rest_idx = train_test_split(np.arange(len(train)), test_size=.40, random_state=SEED, stratify=train.error_type)
    cal_idx, val_idx = train_test_split(rest_idx, test_size=.50, random_state=SEED+1, stratify=train.iloc[rest_idx].error_type)
    build = train.iloc[build_idx]
    calibration = train.iloc[cal_idx].reset_index(drop=True)
    validation = train.iloc[val_idx].reset_index(drop=True)
    supplemental = build.groupby("error_type", group_keys=False).sample(
        n=200, replace=True, random_state=SEED
    ).reset_index(drop=True)
    lm = CharLM().fit(build.corrected_text)
    word_lm = WordLM().fit(build.corrected_text)
    channel = CorruptionChannel().fit(build)
    print("Generating calibration features")
    x_cal, h_cal = make_features(calibration, lm, channel, word_lm, progress=True)
    print("Generating validation features")
    x_val, h_val = make_features(validation, lm, channel, word_lm, progress=True)
    print("Generating supplemental classifier features")
    x_sup, h_sup = make_features(supplemental, lm, channel, word_lm, progress=True)
    print("Training candidate-conditioned visual matcher")
    visual_model = train_visual_matcher(build, channel)
    print("Scoring calibration hypotheses")
    x_cal, v_cal = augment_visual_features(x_cal, h_cal, calibration, visual_model)
    print("Scoring validation hypotheses")
    x_val, v_val = augment_visual_features(x_val, h_val, validation, visual_model)
    print("Scoring supplemental hypotheses")
    x_sup, v_sup = augment_visual_features(x_sup, h_sup, supplemental, visual_model)
    combined_x = pd.concat((x_cal, x_sup), ignore_index=True)
    combined_y = pd.concat((calibration.error_type, supplemental.error_type), ignore_index=True)
    for supplemental_weight in (0.0, 0.15, 0.35, 0.70):
        model = CatBoostClassifier(iterations=550, depth=7, learning_rate=.055, loss_function="MultiClass",
                                   random_seed=SEED, verbose=False, thread_count=10, l2_leaf_reg=5,
                                   auto_class_weights="SqrtBalanced", allow_writing_files=False)
        weights = np.r_[np.ones(len(x_cal)), np.full(len(x_sup), supplemental_weight)]
        model.fit(combined_x, combined_y, sample_weight=weights)
        pred_type, corrected = predict_rows(model, x_val, h_val, validation.candidate_text,
                                            v_val, visual_weight=2.0, enforce_quotas=True)
        score_report(validation, pred_type, corrected,
                     f"supplemental_weight={supplemental_weight}")
    binary_weights = np.r_[np.ones(len(x_cal)), np.full(len(x_sup), .35)]
    binary_model = fit_binary_type_ensemble(combined_x, combined_y, binary_weights)
    pred_type, corrected = predict_rows(
        binary_model, x_val, h_val, validation.candidate_text,
        v_val, visual_weight=2.0, enforce_quotas=True,
    )
    score_report(validation, pred_type, corrected, "binary type ensemble")


def train_and_submit(train, test, submission_out):
    from sklearn.model_selection import train_test_split
    build_idx, cal_idx = train_test_split(np.arange(len(train)), test_size=.22, random_state=SEED, stratify=train.error_type)
    build = train.iloc[build_idx]
    calibration = train.iloc[cal_idx].reset_index(drop=True)
    test = test.reset_index(drop=True)
    supplemental = build.groupby("error_type", group_keys=False).sample(
        n=200, replace=True, random_state=SEED
    ).reset_index(drop=True)
    print(f"Fitting language/channel models on {len(build_idx)} rows")
    lm = CharLM().fit(build.corrected_text)
    word_lm = WordLM().fit(build.corrected_text)
    channel = CorruptionChannel().fit(build)
    print("Generating classifier calibration features")
    x_cal, h_cal = make_features(calibration, lm, channel, word_lm, progress=True)
    print("Generating supplemental classifier features")
    x_sup, h_sup = make_features(supplemental, lm, channel, word_lm, progress=True)
    print("Generating test hypotheses")
    x_test, h_test = make_features(test, lm, channel, word_lm, progress=True)
    print("Training candidate-conditioned visual matcher")
    visual_model = train_visual_matcher(build, channel)
    print("Scoring calibration hypotheses")
    x_cal, v_cal = augment_visual_features(x_cal, h_cal, calibration, visual_model)
    print("Scoring supplemental hypotheses")
    x_sup, v_sup = augment_visual_features(x_sup, h_sup, supplemental, visual_model)
    print("Scoring test hypotheses")
    x_test, v_test = augment_visual_features(x_test, h_test, test, visual_model)
    combined_x = pd.concat((x_cal, x_sup), ignore_index=True)
    combined_y = pd.concat((calibration.error_type, supplemental.error_type), ignore_index=True)
    sample_weights = np.r_[np.ones(len(x_cal)), np.full(len(x_sup), .35)]
    model = fit_binary_type_ensemble(combined_x, combined_y, sample_weights)
    pred_type, corrected = predict_rows(model, x_test, h_test, test.candidate_text,
                                        v_test, visual_weight=2.0, enforce_quotas=True)
    out = pd.DataFrame({
        "id": test.id,
        "is_correct": [int(t == "none") for t in pred_type],
        "error_type": pred_type,
        "corrected_text": corrected,
    })
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(submission_out, index=False)
    print(out.error_type.value_counts().to_dict())
    print(f"Wrote {submission_out}")


def main():
    global DATA
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    DATA = public_dir
    set_seed()
    train = pd.read_csv(public_dir / "train.csv", keep_default_na=False)
    test = pd.read_csv(public_dir / "test.csv", keep_default_na=False)
    train_and_submit(train, test, submission_out)


if __name__ == "__main__":
    main()
