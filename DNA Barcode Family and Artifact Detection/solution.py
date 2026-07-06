"""DNA Barcode Family / NOVEL / ARTIFACT classifier.

Reads dataset/public/{train,test}.csv (raw nucleotide barcode text), trains a
model from scratch, and writes predictions to ./working/submission.csv.

Two complementary signal sources are combined:
  - Engineered reading-frame / codon-usage features: a 6-frame scan under the
    invertebrate mitochondrial genetic code (the correct code for insect COI
    barcodes) finds the cleanest reading frame. A genuine barcode has a clean
    frame; a pseudogene-like ARTIFACT usually does not, even in its best
    frame. This is the primary ARTIFACT signal.
  - A 1D residual CNN over the canonicalized nucleotide sequence, which learns
    the family / NOVEL distinctions (NOVEL has thousands of labeled examples,
    so it is an ordinary supervised class, not an open-set/anomaly problem).

If torch is unavailable or the neural path fails for any reason, the script
falls back to a scikit-learn classifier trained on the engineered features
alone, so a valid submission is always produced.
"""

import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
MAX_LEN = 702
N_FOLDS = 5
MAX_TRAIN_SECONDS = 3000
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 6
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.08
WIDTH = 96
N_BLOCKS = 4
KERNELS = [3, 9, 15, 21]
HIDDEN = 128
DROPOUT_BLOCK = 0.1
DROPOUT_HEAD = 0.3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "public")
OUT_DIR = os.path.join(BASE_DIR, "working")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

CLASSES = ["ARTIFACT"] + [f"F{i:02d}" for i in range(1, 21)] + ["NOVEL"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    USE_TORCH = True
except Exception:
    USE_TORCH = False

if USE_TORCH:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    if USE_TORCH:
        torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Genetic code tables
# ---------------------------------------------------------------------------
STOP_INVMITO = frozenset({"TAA", "TAG"})
STOP_STANDARD = frozenset({"TAA", "TAG", "TGA"})
COMPLEMENT = str.maketrans("ACGT", "TGCA")

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
CODON_TO_IDX = {codon: i for i, codon in enumerate(CODONS)}
DINUCS = [a + b for a in "ACGT" for b in "ACGT"]
DINUC_TO_IDX = {d: i for i, d in enumerate(DINUCS)}
VALID_BASES = set("ACGT")

# Standard genetic code (NCBI translation table 1), then patched to the
# invertebrate mitochondrial code (table 5) -- the correct code for insect
# COI barcodes: AGA/AGG Arg->Ser, ATA Ile->Met, TGA Stop->Trp.
_STANDARD_CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
INVMITO_CODON_TABLE = dict(_STANDARD_CODON_TABLE)
INVMITO_CODON_TABLE.update({"AGA": "S", "AGG": "S", "ATA": "M", "TGA": "W"})

_BASE_LOOKUP = np.full(256, 4, dtype=np.int8)  # default 4 = "other"
_BASE_LOOKUP[ord("A")] = 0
_BASE_LOOKUP[ord("C")] = 1
_BASE_LOOKUP[ord("G")] = 2
_BASE_LOOKUP[ord("T")] = 3
PAD_IDX = 5
OTHER_IDX = 4


# ---------------------------------------------------------------------------
# Sequence utility functions
# ---------------------------------------------------------------------------
def revcomp(seq):
    return seq.translate(COMPLEMENT)[::-1]


def count_stops(seq, stop_set):
    n_codons = len(seq) // 3
    stops = 0
    for i in range(n_codons):
        if seq[i * 3 : i * 3 + 3] in stop_set:
            stops += 1
    return stops, n_codons


def scan_six_frames(seq, stop_set):
    rc = revcomp(seq)
    results = []
    for strand, s in (("+", seq), ("-", rc)):
        for frame in range(3):
            stops, n_codons = count_stops(s[frame:], stop_set)
            results.append(
                {"strand": strand, "frame": frame, "stops": stops, "n_codons": n_codons}
            )
    return results


def best_two_frames(scan_results):
    ordered = sorted(scan_results, key=lambda r: r["stops"])
    return ordered[0], ordered[1]


def gc_content(seq):
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def gc_by_codon_position(oriented_seq):
    n_codons = len(oriented_seq) // 3
    if n_codons == 0:
        return 0.0, 0.0, 0.0
    end = n_codons * 3
    return (
        gc_content(oriented_seq[0:end:3]),
        gc_content(oriented_seq[1:end:3]),
        gc_content(oriented_seq[2:end:3]),
    )


def codon_usage_vector(oriented_seq):
    vec = np.zeros(64, dtype=np.float32)
    n_codons = len(oriented_seq) // 3
    for i in range(n_codons):
        idx = CODON_TO_IDX.get(oriented_seq[i * 3 : i * 3 + 3])
        if idx is not None:
            vec[idx] += 1
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def dinucleotide_vector(seq):
    vec = np.zeros(16, dtype=np.float32)
    n = len(seq) - 1
    for i in range(n):
        idx = DINUC_TO_IDX.get(seq[i : i + 2])
        if idx is not None:
            vec[idx] += 1
    total = vec.sum()
    if total > 0:
        vec /= total
    return vec


def count_non_acgt(seq):
    return sum(1 for ch in seq if ch not in VALID_BASES)


FRAME_INDEX_MAP = {
    ("+", 0): 0,
    ("+", 1): 1,
    ("+", 2): 2,
    ("-", 0): 3,
    ("-", 1): 4,
    ("-", 2): 5,
}


def canonicalize(seq, invmito_scan=None):
    """Reorient/trim seq to its cleanest invertebrate-mito reading frame."""
    scan = invmito_scan if invmito_scan is not None else scan_six_frames(seq, STOP_INVMITO)
    best, second = best_two_frames(scan)
    oriented = revcomp(seq) if best["strand"] == "-" else seq
    canon = oriented[best["frame"] :]
    return canon, best, second


def build_features_and_canonical(raw_seq):
    """Single pass: compute the FEAT_DIM-dim engineered feature vector and the
    canonicalized (oriented, frame-trimmed, untruncated) sequence used for
    the neural net's input tensor."""
    seq = raw_seq.upper()

    scan_im = scan_six_frames(seq, STOP_INVMITO)
    canon_seq, best_im, second_im = canonicalize(seq, scan_im)

    length = len(seq)
    length_mod3 = np.zeros(3, dtype=np.float32)
    length_mod3[length % 3] = 1.0
    non_acgt = float(count_non_acgt(seq))
    gc_overall = gc_content(seq)

    min_stop_im = float(best_im["stops"])
    second_stop_im = float(second_im["stops"])
    gap_im = second_stop_im - min_stop_im

    frame_onehot = np.zeros(6, dtype=np.float32)
    frame_onehot[FRAME_INDEX_MAP[(best_im["strand"], best_im["frame"])]] = 1.0

    gc1, gc2, gc3 = gc_by_codon_position(canon_seq)
    codon_usage = codon_usage_vector(canon_seq)
    dinuc = dinucleotide_vector(seq)

    scan_std = scan_six_frames(seq, STOP_STANDARD)
    best_std, second_std = best_two_frames(scan_std)
    min_stop_std = float(best_std["stops"])
    second_stop_std = float(second_std["stops"])
    gap_std = second_stop_std - min_stop_std

    feats = np.concatenate(
        [
            [length / 700.0],
            length_mod3,
            [non_acgt],
            [gc_overall],
            [min_stop_im],
            [second_stop_im],
            [gap_im],
            frame_onehot,
            [gc1, gc2, gc3],
            codon_usage,
            dinuc,
            [min_stop_std],
            [second_stop_std],
            [gap_std],
        ]
    ).astype(np.float32)

    return feats, canon_seq


FEAT_DIM = 101


def build_feature_and_canonical_tables(sequences):
    n = len(sequences)
    feats = np.zeros((n, FEAT_DIM), dtype=np.float32)
    canon_seqs = [None] * n
    for i, seq in enumerate(sequences):
        f, c = build_features_and_canonical(seq)
        feats[i] = f
        canon_seqs[i] = c
        if (i + 1) % 5000 == 0:
            print(f"  processed {i + 1}/{n} sequences")
    return feats, canon_seqs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), dtype=str)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), dtype=str)
    train_df["sequence"] = train_df["sequence"].str.upper().str.strip()
    test_df["sequence"] = test_df["sequence"].str.upper().str.strip()

    unknown_labels = set(train_df["label"]) - set(CLASSES)
    if unknown_labels:
        raise ValueError(f"Unexpected labels in train data: {unknown_labels}")

    return train_df, test_df


def encode_and_pad(canon_seq):
    seq = canon_seq[:MAX_LEN]
    b = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
    arr = np.full(MAX_LEN, PAD_IDX, dtype=np.int8)
    arr[: len(b)] = _BASE_LOOKUP[b]
    mask = np.zeros(MAX_LEN, dtype=bool)
    mask[: len(b)] = True
    return arr, mask


def build_encoded_arrays(canon_seqs):
    n = len(canon_seqs)
    X = np.full((n, MAX_LEN), PAD_IDX, dtype=np.int8)
    mask = np.zeros((n, MAX_LEN), dtype=bool)
    for i, seq in enumerate(canon_seqs):
        arr, m = encode_and_pad(seq)
        X[i] = arr
        mask[i] = m
    return X, mask


# Synthetic minority-class augmentation was tried (twice): first mutating
# only the CNN's tensor input (regressed 0.745 -> 0.622 OOF macro-F1 from
# feature/sequence inconsistency), then properly fixed to re-run the full
# feature pipeline on each mutated sequence so both branches stayed
# consistent (still regressed to 0.7365, with ARTIFACT recall dropping and
# F19 getting worse, while F16/F20 only moved to ~0.03-0.04 F1). Classes
# this rare (100-200 examples) appear to be a genuine data-floor problem,
# not fixable via resampling, reweighting, or synthetic augmentation -- all
# three mechanisms tried cost more on the classes that matter (ARTIFACT,
# F19) than they gained on F16/F20. Not reattempting.


def class_weights(y_fold, n_classes):
    """Inverse-sqrt-frequency loss weighting. A stronger effective-number-of-
    samples variant (Cui et al., CVPR 2019, beta=0.999) was tried and came
    out flat-to-slightly-worse overall (0.7443 vs 0.7453 OOF macro-F1) while
    barely moving the smallest classes (F16/F20 rarely predicted either way),
    so this simpler, empirically-best formula is kept.
    """
    counts = np.maximum(np.bincount(y_fold, minlength=n_classes).astype(np.float64), 1.0)
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return w.astype(np.float32)


def report_per_class_f1(y_true, y_pred):
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(N_CLASSES)),
            target_names=CLASSES,
            zero_division=0,
            digits=3,
        )
    )


# ---------------------------------------------------------------------------
# Torch model / training (only defined if torch import succeeded)
# ---------------------------------------------------------------------------
if USE_TORCH:

    class BarcodeDataset(Dataset):
        def __init__(self, X_int, mask, feats, labels=None):
            self.X_int = X_int
            self.mask = mask
            self.feats = feats
            self.labels = labels

        def __len__(self):
            return len(self.X_int)

        def __getitem__(self, idx):
            x = torch.from_numpy(self.X_int[idx].astype(np.int64))
            m = torch.from_numpy(self.mask[idx])
            f = torch.from_numpy(self.feats[idx].astype(np.float32))
            if self.labels is not None:
                y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
            else:
                y = torch.tensor(-1, dtype=torch.long)
            return x, m, f, y

    class ResBlock(nn.Module):
        def __init__(self, width, kernel, dropout):
            super().__init__()
            pad = kernel // 2
            self.conv1 = nn.Conv1d(width, width, kernel, padding=pad)
            self.bn1 = nn.BatchNorm1d(width)
            self.conv2 = nn.Conv1d(width, width, kernel, padding=pad)
            self.bn2 = nn.BatchNorm1d(width)
            self.dropout = nn.Dropout(dropout)
            self.act = nn.GELU()

        def forward(self, x):
            residual = x
            out = self.act(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            out = out + residual
            return self.act(out)

    class AttnPool(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.proj = nn.Linear(width, width)
            self.query = nn.Parameter(torch.randn(width) * 0.02)

        def forward(self, x, mask):
            h = x.transpose(1, 2)  # (B, L, width)
            proj = torch.tanh(self.proj(h))
            scores = proj @ self.query  # (B, L)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            return (h * weights.unsqueeze(-1)).sum(dim=1)

    class BarcodeCNN(nn.Module):
        def __init__(self, feat_dim, n_classes):
            super().__init__()
            self.stem = nn.Conv1d(4, WIDTH, kernel_size=15, padding=7)
            self.stem_bn = nn.BatchNorm1d(WIDTH)
            self.act = nn.GELU()
            self.blocks = nn.ModuleList(
                [
                    ResBlock(WIDTH, KERNELS[i % len(KERNELS)], DROPOUT_BLOCK)
                    for i in range(N_BLOCKS)
                ]
            )
            self.pool = AttnPool(WIDTH)
            self.head = nn.Sequential(
                nn.Linear(WIDTH + feat_dim, HIDDEN),
                nn.GELU(),
                nn.Dropout(DROPOUT_HEAD),
                nn.Linear(HIDDEN, n_classes),
            )

        def forward(self, x_int, mask, feats):
            x = torch.clamp(x_int, max=4)
            onehot = F.one_hot(x, num_classes=5).float()[..., :4]  # (B, L, 4)
            onehot = onehot.transpose(1, 2)  # (B, 4, L)
            h = self.act(self.stem_bn(self.stem(onehot)))
            for block in self.blocks:
                h = block(h)
            pooled = self.pool(h, mask)
            combined = torch.cat([pooled, feats], dim=1)
            return self.head(combined)

    def train_one_fold(fold_idx, train_idx, val_idx, X_int, mask_arr, feats_raw, y, start_time):
        scaler = StandardScaler()
        feats_train = scaler.fit_transform(feats_raw[train_idx]).astype(np.float32)
        feats_val = scaler.transform(feats_raw[val_idx]).astype(np.float32)
        y_train, y_val = y[train_idx], y[val_idx]

        train_ds = BarcodeDataset(X_int[train_idx], mask_arr[train_idx], feats_train, y_train)
        val_ds = BarcodeDataset(X_int[val_idx], mask_arr[val_idx], feats_val, y_val)

        # Class-balanced batch sampling (both full inverse-frequency and a
        # milder sqrt-based version) was tried here and consistently made
        # overall OOF macro-F1 worse (0.745 baseline -> 0.709 full inverse-freq),
        # driven mainly by degraded ARTIFACT recall: oversampling the smallest
        # classes (F16=162, F20=123 examples) means the same handful of
        # sequences get replayed many times per epoch, and crowds out batch
        # composition for every other class, which hurts more broadly than it
        # helps the tiny classes it targets. Natural sampling plus the
        # per-sample inverse-sqrt-frequency loss weight below is the
        # empirically best configuration measured; F16/F20 remain hard given
        # they have only ~100-150 total examples, but that is a real data
        # limitation, not something batch resampling fixes here.
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False)

        torch.manual_seed(SEED + fold_idx)
        model = BarcodeCNN(feats_raw.shape[1], N_CLASSES).to(DEVICE)
        w = class_weights(y_train, N_CLASSES)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(w, device=DEVICE), label_smoothing=LABEL_SMOOTH
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
        )

        best_state, best_f1, epochs_no_improve = None, -1.0, 0

        for epoch in range(MAX_EPOCHS):
            if time.time() - start_time > MAX_TRAIN_SECONDS:
                print(f"  [fold {fold_idx}] time budget exceeded, stopping at epoch {epoch}")
                break

            model.train()
            for x_b, m_b, f_b, y_b in train_loader:
                x_b, m_b, f_b, y_b = (
                    x_b.to(DEVICE),
                    m_b.to(DEVICE),
                    f_b.to(DEVICE),
                    y_b.to(DEVICE),
                )
                optimizer.zero_grad()
                logits = model(x_b, m_b, f_b)
                loss = criterion(logits, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            model.eval()
            val_preds = []
            with torch.no_grad():
                for x_b, m_b, f_b, _ in val_loader:
                    x_b, m_b, f_b = x_b.to(DEVICE), m_b.to(DEVICE), f_b.to(DEVICE)
                    logits = model(x_b, m_b, f_b)
                    val_preds.append(logits.argmax(dim=1).cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            scheduler.step(val_f1)

            if val_f1 > best_f1 + 1e-5:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f"  [fold {fold_idx}] epoch {epoch + 1}: val macro-F1={val_f1:.4f} best={best_f1:.4f}")

            if epochs_no_improve >= PATIENCE:
                print(f"  [fold {fold_idx}] early stopping at epoch {epoch + 1}")
                break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_f1 = 0.0

        model.load_state_dict(best_state)
        model.eval()
        oof_probs = np.zeros((len(val_idx), N_CLASSES), dtype=np.float32)
        offset = 0
        with torch.no_grad():
            for x_b, m_b, f_b, _ in val_loader:
                x_b, m_b, f_b = x_b.to(DEVICE), m_b.to(DEVICE), f_b.to(DEVICE)
                probs = torch.softmax(model(x_b, m_b, f_b), dim=1).cpu().numpy()
                oof_probs[offset : offset + len(probs)] = probs
                offset += len(probs)

        return best_state, best_f1, oof_probs, scaler

    def run_cv_training(train_df, feats_raw, X_int, mask_arr):
        y = train_df["label"].map(CLASS_TO_IDX).values.astype(np.int64)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_models = []
        oof_probs = np.zeros((len(y), N_CLASSES), dtype=np.float32)
        oof_mask = np.zeros(len(y), dtype=bool)
        start_time = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            print(f"=== Fold {fold_idx + 1}/{N_FOLDS} ===")
            if time.time() - start_time > MAX_TRAIN_SECONDS:
                print("Global time budget exceeded, stopping fold loop.")
                break
            best_state, best_f1, fold_oof_probs, scaler = train_one_fold(
                fold_idx, train_idx, val_idx, X_int, mask_arr, feats_raw, y, start_time
            )
            fold_models.append((best_state, scaler))
            oof_probs[val_idx] = fold_oof_probs
            oof_mask[val_idx] = True
            print(f"Fold {fold_idx + 1} best val macro-F1: {best_f1:.4f}")

        if not fold_models:
            raise RuntimeError("No folds completed training within the time budget.")

        if oof_mask.any():
            oof_pred = oof_probs[oof_mask].argmax(axis=1)
            oof_f1 = f1_score(y[oof_mask], oof_pred, average="macro", zero_division=0)
            print(f"Overall OOF macro-F1 (torch CNN): {oof_f1:.4f}")
            report_per_class_f1(y[oof_mask], oof_pred)

        return fold_models

    def predict_test_torch(fold_models, X_int_test, mask_test, feats_raw_test):
        probs_sum = np.zeros((len(X_int_test), N_CLASSES), dtype=np.float64)
        for state, scaler in fold_models:
            model = BarcodeCNN(feats_raw_test.shape[1], N_CLASSES).to(DEVICE)
            model.load_state_dict(state)
            model.eval()
            feats_scaled = scaler.transform(feats_raw_test).astype(np.float32)
            ds = BarcodeDataset(X_int_test, mask_test, feats_scaled, None)
            loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False)
            probs_list = []
            with torch.no_grad():
                for x_b, m_b, f_b, _ in loader:
                    x_b, m_b, f_b = x_b.to(DEVICE), m_b.to(DEVICE), f_b.to(DEVICE)
                    probs_list.append(torch.softmax(model(x_b, m_b, f_b), dim=1).cpu().numpy())
            probs_sum += np.concatenate(probs_list)
        probs_avg = probs_sum / len(fold_models)
        pred_idx = probs_avg.argmax(axis=1)
        return [CLASSES[i] for i in pred_idx]


# ---------------------------------------------------------------------------
# Fallback path: engineered features + sklearn gradient boosting
# ---------------------------------------------------------------------------
def fit_predict_fallback(train_df, test_df, feats_train, feats_test):
    from sklearn.ensemble import HistGradientBoostingClassifier

    y = train_df["label"].map(CLASS_TO_IDX).values.astype(np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    probs_sum = np.zeros((len(test_df), N_CLASSES), dtype=np.float64)
    oof_probs = np.zeros((len(y), N_CLASSES), dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(feats_train[train_idx])
        X_val = scaler.transform(feats_train[val_idx])
        X_test = scaler.transform(feats_test)

        y_train = y[train_idx]
        w = class_weights(y_train, N_CLASSES)
        sample_weight = w[y_train]

        clf = HistGradientBoostingClassifier(random_state=SEED + fold_idx, max_iter=300)
        clf.fit(X_train, y_train, sample_weight=sample_weight)

        val_probs = clf.predict_proba(X_val)
        full_val = np.zeros((len(val_idx), N_CLASSES), dtype=np.float32)
        full_val[:, clf.classes_] = val_probs
        oof_probs[val_idx] = full_val

        test_probs = clf.predict_proba(X_test)
        full_test = np.zeros((len(test_df), N_CLASSES), dtype=np.float64)
        full_test[:, clf.classes_] = test_probs
        probs_sum += full_test

        fold_f1 = f1_score(y[val_idx], full_val.argmax(axis=1), average="macro", zero_division=0)
        print(f"[fallback] fold {fold_idx + 1} val macro-F1: {fold_f1:.4f}")

    oof_f1 = f1_score(y, oof_probs.argmax(axis=1), average="macro", zero_division=0)
    print(f"[fallback] Overall OOF macro-F1: {oof_f1:.4f}")
    report_per_class_f1(y, oof_probs.argmax(axis=1))

    pred_idx = (probs_sum / N_FOLDS).argmax(axis=1)
    return [CLASSES[i] for i in pred_idx]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_submission(test_ids, pred_labels, test_df):
    df = pd.DataFrame({"id": test_ids, "label": pred_labels})[["id", "label"]]
    assert len(df) == len(test_df), "row count mismatch"
    assert set(df["id"]) == set(test_df["id"]), "id set mismatch"
    assert df["id"].duplicated().sum() == 0, "duplicate ids"
    assert df["label"].isin(CLASSES).all(), "invalid label found"
    assert df["label"].notna().all(), "NaN label found"
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


def main():
    t0 = time.time()
    set_all_seeds(SEED)

    print("Loading data...")
    train_df, test_df = load_data()
    print(f"train: {len(train_df)} rows, test: {len(test_df)} rows")

    print("Building engineered features (train)...")
    feats_train, canon_train = build_feature_and_canonical_tables(train_df["sequence"].tolist())
    print("Building engineered features (test)...")
    feats_test, canon_test = build_feature_and_canonical_tables(test_df["sequence"].tolist())

    pred_labels = None

    if USE_TORCH:
        try:
            print(f"Using device: {DEVICE}")
            print("Encoding sequences for CNN...")
            X_int_train, mask_train = build_encoded_arrays(canon_train)
            X_int_test, mask_test = build_encoded_arrays(canon_test)

            fold_models = run_cv_training(train_df, feats_train, X_int_train, mask_train)
            pred_labels = predict_test_torch(fold_models, X_int_test, mask_test, feats_test)
            print("Torch CNN path succeeded.")
        except Exception as e:
            print(f"Torch path failed with error: {e!r}. Falling back to sklearn.")
            pred_labels = None
    else:
        print("torch not available; using sklearn fallback path.")

    if pred_labels is None:
        print("Running sklearn fallback path...")
        pred_labels = fit_predict_fallback(train_df, test_df, feats_train, feats_test)

    write_submission(test_df["id"].tolist(), pred_labels, test_df)
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
