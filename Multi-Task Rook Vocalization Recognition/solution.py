#!/usr/bin/env python3
"""Train a session-robust multi-task rook-call recognizer and write predictions."""

import csv
import io
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SEED = 20260809
NUM_INDIVIDUALS = 15
NUM_CALL_TYPES = 8
COLUMNS = (
    ["id", "det_score"]
    + [f"ind_{i}" for i in range(NUM_INDIVIDUALS)]
    + [f"ct_{i}" for i in range(NUM_CALL_TYPES)]
)
TRAINING_DEADLINE_SECONDS = 3000.0
MAX_EPOCHS = 18
PATIENCE = 5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def neutral_scores():
    return [0.5] + [1.0 / NUM_INDIVIDUALS] * NUM_INDIVIDUALS + [1.0 / NUM_CALL_TYPES] * NUM_CALL_TYPES


def write_placeholder(path, n_rows):
    """Write the required early, schema-valid safety submission."""
    neutral = neutral_scores()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for i in range(n_rows):
            writer.writerow([f"t{i:05d}"] + neutral)


def write_predictions(path, det, ind, ct):
    """Write one independently checked row at a time."""
    n_rows = len(det)
    if ind.shape != (n_rows, NUM_INDIVIDUALS) or ct.shape != (n_rows, NUM_CALL_TYPES):
        print("WARNING: prediction shapes are invalid; retaining placeholder", flush=True)
        return False

    buffer = io.StringIO(newline="")
    with buffer as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for i in range(n_rows):
            values = [float(det[i])] + [float(x) for x in ind[i]] + [float(x) for x in ct[i]]
            if not all(math.isfinite(x) for x in values):
                values = neutral_scores()
            writer.writerow([f"t{i:05d}"] + [format(x, ".9g") for x in values])
        contents = buffer.getvalue()
    path.write_text(contents, encoding="utf-8")
    return True


class RookDataset(Dataset):
    def __init__(self, waveforms, is_call, individual, call_type, indices):
        self.waveforms = waveforms
        self.is_call = is_call
        self.individual = individual
        self.call_type = call_type
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        # float32 conversion makes a writable, finite-safe copy of the mmap row.
        wave = np.array(self.waveforms[idx], dtype=np.float32, copy=True)
        np.nan_to_num(wave, copy=False)
        return (
            torch.from_numpy(wave),
            int(self.is_call[idx]),
            int(self.individual[idx]),
            int(self.call_type[idx]),
        )


def hz_to_mel(freq):
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def make_mel_filter(sample_rate=16000, n_fft=512, n_mels=80, f_min=80.0, f_max=7900.0):
    frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1, dtype=np.float32)
    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = np.asarray([mel_to_hz(x) for x in mel_points], dtype=np.float32)
    bank = np.zeros((n_mels, len(frequencies)), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = hz_points[m : m + 3]
        bank[m] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / max(center - left, 1e-8),
                (right - frequencies) / max(right - center, 1e-8),
            ),
        )
    # Area normalization prevents high-frequency filters from dominating by width alone.
    bank /= np.maximum(bank.sum(axis=1, keepdims=True), 1e-8)
    return torch.from_numpy(bank)


class MelFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = 512
        self.hop_length = 160
        self.win_length = 400
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer("mel_filter", make_mel_filter(), persistent=False)

    def forward(self, wave):
        spectrum = torch.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spectrum.real.square() + spectrum.imag.square()
        mel = torch.matmul(self.mel_filter, power)
        log_mel = torch.log(mel.clamp_min(1e-7)).clamp_(-16.0, 8.0)
        row_mean = log_mel.mean(dim=(1, 2), keepdim=True)
        row_scale = log_mel.var(dim=(1, 2), keepdim=True, unbiased=False).add(1e-4).sqrt()
        normalized = (log_mel - row_mean) / row_scale
        return torch.stack((log_mel, normalized), dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(drop) if drop else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.silu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(self.drop(x)))
        return F.silu(x + residual)


class AttentivePoolHead(nn.Module):
    """Learn task evidence locations and an angularly discriminative embedding."""

    def __init__(self, channels, outputs, cosine=False):
        super().__init__()
        self.outputs = outputs
        self.cosine = cosine
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 2, 1, 1),
        )
        feature_dim = channels * 2
        self.embedding = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 192),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.LayerNorm(192),
        )
        self.classifier = nn.Linear(192, outputs, bias=not cosine)
        if cosine:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def scale(self):
        return self.logit_scale.clamp(max=math.log(50.0)).exp()

    def forward(self, x, return_embedding=False):
        values = x.flatten(2)
        weights = torch.softmax(self.attention(x).flatten(2), dim=-1)
        mean = (values * weights).sum(dim=-1)
        second_moment = (values.square() * weights).sum(dim=-1)
        std = (second_moment - mean.square()).clamp_min(1e-5).sqrt()
        embedding = self.embedding(torch.cat((mean, std), dim=1))
        if self.cosine:
            embedding = F.normalize(embedding, dim=1)
            weights = F.normalize(self.classifier.weight, dim=1)
            logits = self.scale() * F.linear(embedding, weights)
        else:
            logits = self.classifier(embedding)
        return (logits, embedding) if return_embedding else logits


class MultiTaskRookNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = MelFrontend()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.shared_body = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2, drop=0.05),
            ResidualBlock(64, 64),
            ResidualBlock(64, 96, stride=2, drop=0.08),
            ResidualBlock(96, 96),
        )
        self.det_branch = self.make_branch()
        self.ind_branch = self.make_branch()
        self.ct_branch = self.make_branch()
        self.det_head = AttentivePoolHead(160, 1)
        self.ind_head = AttentivePoolHead(160, NUM_INDIVIDUALS, cosine=True)
        self.ct_head = AttentivePoolHead(160, NUM_CALL_TYPES)

    @staticmethod
    def make_branch():
        return nn.Sequential(
            ResidualBlock(96, 160, stride=2, drop=0.10),
            ResidualBlock(160, 160, drop=0.10),
        )

    def augment_waveform(self, wave):
        gain = torch.empty((wave.shape[0], 1), device=wave.device).uniform_(0.45, 1.50)
        polarity = torch.where(
            torch.rand((wave.shape[0], 1), device=wave.device) < 0.5,
            -torch.ones((), device=wave.device),
            torch.ones((), device=wave.device),
        )
        wave = wave * gain * polarity
        shift = int(torch.randint(-1600, 1601, (1,), device=wave.device).item())
        return torch.roll(wave, shifts=shift, dims=1)

    @staticmethod
    def augment_spectrogram(spectrogram):
        _, _, frequencies, frames = spectrogram.shape
        freq_width = int(torch.randint(0, 9, (1,), device=spectrogram.device).item())
        time_width = int(torch.randint(0, 13, (1,), device=spectrogram.device).item())
        freq_start = int(torch.randint(0, frequencies - freq_width + 1, (1,), device=spectrogram.device).item())
        time_start = int(torch.randint(0, frames - time_width + 1, (1,), device=spectrogram.device).item())
        keep = torch.ones((1, 1, frequencies, frames), dtype=torch.bool, device=spectrogram.device)
        if freq_width:
            keep[:, :, freq_start : freq_start + freq_width, :] = False
        if time_width:
            keep[:, :, :, time_start : time_start + time_width] = False
        fill = spectrogram.mean(dim=(2, 3), keepdim=True)
        return torch.where(keep, spectrogram, fill)

    def forward(self, wave, augment=False, ind_mask=None, ct_mask=None, return_embeddings=False):
        if augment:
            wave = self.augment_waveform(wave)
        # Fixed signal processing stays in float32; FFT power can overflow float16.
        with torch.no_grad(), torch.autocast(device_type=wave.device.type, enabled=False):
            spectrogram = self.frontend(wave.float())
            if augment:
                spectrogram = self.augment_spectrogram(spectrogram)
        shared = self.shared_body(self.stem(spectrogram))
        det_logits = self.det_head(self.det_branch(shared)).squeeze(1)

        ind_input = shared if ind_mask is None else shared[ind_mask]
        ct_input = shared if ct_mask is None else shared[ct_mask]
        if len(ind_input):
            ind_result = self.ind_head(self.ind_branch(ind_input), return_embedding=return_embeddings)
        else:
            empty_logits = shared.new_empty((0, NUM_INDIVIDUALS))
            empty_embedding = shared.new_empty((0, 192))
            ind_result = (empty_logits, empty_embedding) if return_embeddings else empty_logits
        if len(ct_input):
            ct_result = self.ct_head(self.ct_branch(ct_input), return_embedding=return_embeddings)
        else:
            empty_logits = shared.new_empty((0, NUM_CALL_TYPES))
            empty_embedding = shared.new_empty((0, 192))
            ct_result = (empty_logits, empty_embedding) if return_embeddings else empty_logits
        if return_embeddings:
            ind_logits, ind_embedding = ind_result
            ct_logits, ct_embedding = ct_result
            return det_logits, ind_logits, ct_logits, ind_embedding, ct_embedding
        return det_logits, ind_result, ct_result


def choose_group_split(recording, is_call, individual, call_type, seed, val_fraction=0.18):
    """Search deterministic train-only group splits for label coverage and balance."""
    groups = np.unique(recording)
    rng = np.random.default_rng(seed)
    target_size = val_fraction * len(recording)
    global_ind = np.bincount(individual[individual >= 0], minlength=NUM_INDIVIDUALS).astype(np.float64)
    global_ct = np.bincount(call_type[call_type >= 0], minlength=NUM_CALL_TYPES).astype(np.float64)
    global_ind /= max(global_ind.sum(), 1.0)
    global_ct /= max(global_ct.sum(), 1.0)
    best = None

    for _ in range(768):
        selected = []
        selected_size = 0
        for group in rng.permutation(groups):
            selected.append(group)
            selected_size += int(np.count_nonzero(recording == group))
            if selected_size >= target_size:
                break
        val_mask = np.isin(recording, selected)
        train_mask = ~val_mask
        train_ind = np.bincount(individual[train_mask & (individual >= 0)], minlength=NUM_INDIVIDUALS)
        val_ind = np.bincount(individual[val_mask & (individual >= 0)], minlength=NUM_INDIVIDUALS)
        train_ct = np.bincount(call_type[train_mask & (call_type >= 0)], minlength=NUM_CALL_TYPES)
        val_ct = np.bincount(call_type[val_mask & (call_type >= 0)], minlength=NUM_CALL_TYPES)
        missing_train = int(np.count_nonzero(train_ind == 0) + np.count_nonzero(train_ct == 0))
        missing_val = int(np.count_nonzero(val_ind == 0) + np.count_nonzero(val_ct == 0))
        val_ind_dist = val_ind / max(val_ind.sum(), 1)
        val_ct_dist = val_ct / max(val_ct.sum(), 1)
        distribution_error = np.abs(val_ind_dist - global_ind).mean() + np.abs(val_ct_dist - global_ct).mean()
        size_error = abs(val_mask.mean() - val_fraction)
        call_error = abs(is_call[val_mask].mean() - is_call.mean())
        score = (missing_train, missing_val, distribution_error + size_error + call_error)
        if best is None or score < best[0]:
            best = (score, np.flatnonzero(train_mask), np.flatnonzero(val_mask), selected)

    return best[1], best[2], best[3]


def reciprocal_rank(scores, labels):
    if len(labels) == 0:
        return float("nan")
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    true_scores = scores[np.arange(len(labels)), labels]
    above = (scores > true_scores[:, None]).sum(axis=1)
    tied = (scores == true_scores[:, None]).sum(axis=1) - 1
    ranks = 1.0 + above + 0.5 * tied
    return float(np.mean(1.0 / ranks))


def validation_metrics(det_scores, ind_scores, ct_scores, is_call, individual, call_type):
    det_scores = np.nan_to_num(det_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    backgrounds = det_scores[is_call == 0]
    calls = det_scores[is_call == 1]
    efficiencies = []
    for false_alarm_rate in (0.05, 0.10):
        threshold = np.quantile(backgrounds, 1.0 - false_alarm_rate, method="higher")
        efficiencies.append(float(np.mean(calls > threshold)))
    detection = float(np.mean(efficiencies))
    call_mask = is_call == 1
    individual_mrr = reciprocal_rank(ind_scores[call_mask], individual[call_mask])
    ct_mask = call_type >= 0
    call_type_mrr = reciprocal_rank(ct_scores[ct_mask], call_type[ct_mask])
    score = float(np.mean((detection, individual_mrr, call_type_mrr)))
    return score, detection, individual_mrr, call_type_mrr


def make_loader(dataset, batch_size, shuffle, device):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
        drop_last=False,
    )


def evaluate(model, loader, device, labels):
    model.eval()
    det_parts, ind_parts, ct_parts = [], [], []
    with torch.inference_mode():
        for wave, _, _, _ in loader:
            wave = wave.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                det, ind, ct = model(wave)
            det_parts.append(det.float().cpu().numpy())
            ind_parts.append(ind.float().cpu().numpy())
            ct_parts.append(ct.float().cpu().numpy())
    return validation_metrics(
        np.concatenate(det_parts),
        np.concatenate(ind_parts),
        np.concatenate(ct_parts),
        labels[0],
        labels[1],
        labels[2],
    )


def supervised_contrastive_loss(embedding, labels, scale):
    """Pull same-label calls together across sessions within each training batch."""
    if len(embedding) < 2:
        return embedding.sum() * 0.0
    same = labels[:, None] == labels[None, :]
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = same & ~diagonal
    positive_count = positives.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embedding.sum() * 0.0
    similarity = scale * (embedding @ embedding.T)
    similarity = similarity.masked_fill(diagonal, -1e4)
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    per_anchor = -(log_probability * positives).sum(dim=1) / positive_count.clamp_min(1)
    return per_anchor[valid].mean()


def run_training_epoch(model, loader, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    seen = 0
    skipped_batches = 0
    for wave, det_target, ind_target, ct_target in loader:
        wave = wave.to(device, non_blocking=True)
        det_target = det_target.to(device, dtype=torch.float32, non_blocking=True)
        ind_target = ind_target.to(device, non_blocking=True)
        ct_target = ct_target.to(device, non_blocking=True)
        call_mask = ind_target >= 0
        type_mask = ct_target >= 0
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            det_logits, ind_logits, ct_logits, ind_embedding, _ = model(
                wave,
                augment=True,
                ind_mask=call_mask,
                ct_mask=type_mask,
                return_embeddings=True,
            )
            loss = F.binary_cross_entropy_with_logits(det_logits, det_target)
            if call_mask.any():
                call_labels = ind_target[call_mask]
                loss = loss + F.cross_entropy(ind_logits, call_labels, label_smoothing=0.04)
                loss = loss + 0.08 * supervised_contrastive_loss(
                    ind_embedding,
                    call_labels,
                    model.ind_head.scale(),
                )
            if type_mask.any():
                type_labels = ct_target[type_mask]
                loss = loss + F.cross_entropy(ct_logits, type_labels, label_smoothing=0.04)
        if not torch.isfinite(loss).item():
            skipped_batches += 1
            continue
        loss_value = float(loss.detach().cpu())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad_norm).item():
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss_value * len(wave)
        seen += len(wave)
    return running_loss / max(seen, 1), skipped_batches


def train_model(waveforms, is_call, individual, call_type, recording, device, seed, started_at):
    seed_everything(seed)
    train_idx, val_idx, val_groups = choose_group_split(recording, is_call, individual, call_type, seed)
    print(
        f"split seed={seed}: train={len(train_idx)} val={len(val_idx)} "
        f"held-out sessions={len(val_groups)}",
        flush=True,
    )
    train_set = RookDataset(waveforms, is_call, individual, call_type, train_idx)
    val_set = RookDataset(waveforms, is_call, individual, call_type, val_idx)
    batch_size = 128 if device.type != "cpu" else 48
    train_loader = make_loader(train_set, batch_size, True, device)
    val_loader = make_loader(val_set, batch_size * 2, False, device)
    labels = (is_call[val_idx], individual[val_idx], call_type[val_idx])

    model = MultiTaskRookNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=3e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_score = -math.inf
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    previous_epoch_seconds = 0.0

    for epoch in range(1, MAX_EPOCHS + 1):
        elapsed = time.monotonic() - started_at
        if elapsed + 1.25 * previous_epoch_seconds >= TRAINING_DEADLINE_SECONDS:
            print("wall-clock guard: stopping before next epoch", flush=True)
            break
        epoch_started = time.monotonic()
        running_loss, skipped_batches = run_training_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
        )
        scheduler.step()
        metrics = evaluate(model, val_loader, device, labels)
        previous_epoch_seconds = time.monotonic() - epoch_started
        print(
            f"epoch={epoch:02d} loss={running_loss:.4f} "
            f"RookScore={metrics[0]:.4f} det={metrics[1]:.4f} "
            f"ind_mrr={metrics[2]:.4f} ct_mrr={metrics[3]:.4f} "
            f"seconds={previous_epoch_seconds:.1f} skipped={skipped_batches}",
            flush=True,
        )
        if metrics[0] > best_score:
            best_score = metrics[0]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            print("early stopping", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training deadline reached before completing one epoch")
    model.load_state_dict(best_state)
    model.eval()
    print(f"selected epoch={best_epoch} validation RookScore={best_score:.4f}", flush=True)
    return (
        model,
        best_score,
        best_epoch,
        epoch,
        time.monotonic() - started_at,
        len(train_idx) / len(waveforms),
    )


def train_full_model(
    waveforms,
    is_call,
    individual,
    call_type,
    device,
    seed,
    epochs,
    started_at,
):
    """Retrain on every training session for the train-only selected epoch count."""
    seed_everything(seed)
    indices = np.arange(len(waveforms), dtype=np.int64)
    dataset = RookDataset(waveforms, is_call, individual, call_type, indices)
    batch_size = 128 if device.type != "cpu" else 48
    loader = make_loader(dataset, batch_size, True, device)
    model = MultiTaskRookNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=3e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    previous_epoch_seconds = 0.0

    for epoch in range(1, epochs + 1):
        elapsed = time.monotonic() - started_at
        if elapsed + 1.25 * previous_epoch_seconds + 300.0 >= TRAINING_DEADLINE_SECONDS:
            print("wall-clock guard: abandoning incomplete full-data member", flush=True)
            return None
        epoch_started = time.monotonic()
        loss, skipped = run_training_epoch(model, loader, optimizer, scaler, device)
        scheduler.step()
        previous_epoch_seconds = time.monotonic() - epoch_started
        print(
            f"full-data seed={seed} epoch={epoch:02d}/{epochs:02d} "
            f"loss={loss:.4f} seconds={previous_epoch_seconds:.1f} skipped={skipped}",
            flush=True,
        )
    model.eval()
    return model


def predict_batch(models, wave, device):
    wave = torch.nan_to_num(wave.to(device, non_blocking=True))
    det_sum = None
    ind_sum = None
    ct_sum = None
    with torch.inference_mode():
        for model in models:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                det_logits, ind_logits, ct_logits = model(wave)
                det = torch.sigmoid(det_logits).float()
                ind = torch.softmax(ind_logits, dim=1).float()
                ct = torch.softmax(ct_logits, dim=1).float()
            det_sum = det if det_sum is None else det_sum + det
            ind_sum = ind if ind_sum is None else ind_sum + ind
            ct_sum = ct if ct_sum is None else ct_sum + ct
    scale = 1.0 / len(models)
    return (
        (det_sum * scale).cpu().numpy(),
        (ind_sum * scale).cpu().numpy(),
        (ct_sum * scale).cpu().numpy(),
    )


def infer(models, test_waveforms, device):
    n_rows = len(test_waveforms)
    neutral = neutral_scores()
    det_out = np.full(n_rows, neutral[0], dtype=np.float32)
    ind_out = np.full((n_rows, NUM_INDIVIDUALS), neutral[1], dtype=np.float32)
    ct_out = np.full((n_rows, NUM_CALL_TYPES), neutral[-1], dtype=np.float32)
    batch_size = 192 if device.type != "cpu" else 48

    for start in range(0, n_rows, batch_size):
        stop = min(start + batch_size, n_rows)
        batch_array = np.array(test_waveforms[start:stop], dtype=np.float32, copy=True)
        batch = torch.from_numpy(batch_array)
        try:
            det, ind, ct = predict_batch(models, batch, device)
            det_out[start:stop] = np.nan_to_num(det, nan=neutral[0], posinf=1.0, neginf=0.0)
            ind_out[start:stop] = np.nan_to_num(ind, nan=neutral[1], posinf=neutral[1], neginf=neutral[1])
            ct_out[start:stop] = np.nan_to_num(ct, nan=neutral[-1], posinf=neutral[-1], neginf=neutral[-1])
        except Exception as batch_error:
            print(f"WARNING: batch {start}:{stop} failed ({batch_error}); retrying row-wise", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for offset in range(stop - start):
                try:
                    det, ind, ct = predict_batch(models, batch[offset : offset + 1], device)
                    det_out[start + offset] = np.nan_to_num(det[0], nan=neutral[0], posinf=1.0, neginf=0.0)
                    ind_out[start + offset] = np.nan_to_num(ind[0], nan=neutral[1], posinf=neutral[1], neginf=neutral[1])
                    ct_out[start + offset] = np.nan_to_num(ct[0], nan=neutral[-1], posinf=neutral[-1], neginf=neutral[-1])
                except Exception as row_error:
                    print(f"WARNING: row {start + offset} failed ({row_error}); using neutral scores", flush=True)
    return det_out, ind_out, ct_out


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out>")
    started_at = time.monotonic()
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    required = [
        "train_X.npy",
        "train_iscall.npy",
        "train_ind.npy",
        "train_ct.npy",
        "train_rec.npy",
        "test_X.npy",
    ]
    missing = [name for name in required if not (public_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {', '.join(missing)}")

    submission_out.parent.mkdir(parents=True, exist_ok=True)
    test_waveforms = np.load(public_dir / "test_X.npy", mmap_mode="r")
    if test_waveforms.ndim != 2:
        raise ValueError("test_X.npy must be a two-dimensional waveform array")
    write_placeholder(submission_out, len(test_waveforms))
    print(f"wrote early placeholder with {len(test_waveforms)} rows", flush=True)

    try:
        waveforms = np.load(public_dir / "train_X.npy", mmap_mode="r")
        is_call = np.load(public_dir / "train_iscall.npy")
        individual = np.load(public_dir / "train_ind.npy")
        call_type = np.load(public_dir / "train_ct.npy")
        recording = np.load(public_dir / "train_rec.npy")
        lengths = [len(waveforms), len(is_call), len(individual), len(call_type), len(recording)]
        if len(set(lengths)) != 1:
            print(f"WARNING: inconsistent training lengths {lengths}; using shortest", flush=True)
            usable = min(lengths)
            waveforms = waveforms[:usable]
            is_call = is_call[:usable]
            individual = individual[:usable]
            call_type = call_type[:usable]
            recording = recording[:usable]

        seed_everything(SEED)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device={device} train_rows={len(waveforms)}", flush=True)

        (
            validation_model,
            validation_score,
            selected_epochs,
            trained_epochs,
            validation_duration,
            fold_fraction,
        ) = train_model(
            waveforms,
            is_call,
            individual,
            call_type,
            recording,
            device,
            SEED,
            started_at,
        )
        models = [validation_model]
        print(f"validation score={validation_score:.4f}", flush=True)

        estimated_full_duration = (
            validation_duration
            / max(trained_epochs, 1)
            * selected_epochs
            / fold_fraction
        )
        full_models = []
        for full_seed in (SEED + 101, SEED + 211):
            elapsed = time.monotonic() - started_at
            if elapsed + 1.20 * estimated_full_duration + 300.0 >= TRAINING_DEADLINE_SECONDS:
                print("wall-clock guard: skipping remaining full-data members", flush=True)
                break
            try:
                full_model = train_full_model(
                    waveforms,
                    is_call,
                    individual,
                    call_type,
                    device,
                    full_seed,
                    selected_epochs,
                    started_at,
                )
                if full_model is None:
                    break
                full_models.append(full_model)
            except Exception as training_error:
                print(f"WARNING: full-data model failed: {training_error}", flush=True)
                break
        if full_models:
            models = full_models
        print(f"inference ensemble members={len(models)}", flush=True)
        det, ind, ct = infer(models, test_waveforms, device)
        if write_predictions(submission_out, det, ind, ct):
            print(f"wrote final submission: {submission_out} rows={len(test_waveforms)}", flush=True)
    except Exception as error:
        # Heavy-work failures must leave the already-written valid placeholder in place.
        print(f"WARNING: pipeline failed after placeholder creation: {error}", flush=True)


if __name__ == "__main__":
    main()
