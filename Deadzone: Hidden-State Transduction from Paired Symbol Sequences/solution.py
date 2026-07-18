"""
Deadzone: Hidden-State Transduction from Paired Symbol Sequences
================================================================
Sequence tagging / transduction. Each record is two parallel opaque symbol
strings -- actor_codes and foe_codes over a 254-symbol alphabet (c000..c253)
plus a blank marker [X] -- and the target is an 8-class `states` string
(rsp,gnd,air,atk,shd,grb,hit,evd) aligned one-to-one with the actor's scored
positions. A hidden automaton makes the symbol->class map one-to-many, ~28% of
outputs are forced by the foe, 22% of inputs are blanked, and a 60-symbol
unscored prefix is provided to warm up the latent state.

Model (a single genuinely-trained neural sequence tagger, from scratch, no
pretraining -- the alphabet is salted so no external weights could help):

  * INPUT per position: a SHARED symbol embedding applied to the actor symbol,
    the foe symbol, and each stream's forward-filled last-non-blank symbol
    (imputes the [X] holes); the actor(x)foe Hadamard interaction (targets the
    foe-forced hit/evd/rsp states); explicit blank flags; and record-level
    categorical embeddings for arena / actor_kind / foe_kind plus a learned
    matchup vector MLP([actor_kind, foe_kind]).  actor_kind matters a lot: the
    class prior is strongly archetype-dependent (atk share ranges 8%..82%).
  * ENCODER: a local conv front-end + a 2-layer BiLSTM/BiGRU that reads the
    WHOLE sequence (the prefix warms the hidden state) and tracks the latent
    automaton state bidirectionally.  A per-position classifier on the current
    symbol alone is one-to-many and caps ~43%; carrying state is the whole game
    (knowing the true previous state gives ~96% oracle accuracy).
  * HEADS: (a) a linear-chain CRF (learned 8x8 transition matrix) whose Viterbi
    decode emits coherent runs and places the rare (4.7%) boundaries -> the CRF's
    diagonal learns the ~95% self-transition; (b) an auxiliary boundary head
    predicting P(state changes at k), which both regularizes the encoder to
    localize transitions and provides a decode-time signal.
  * TRAINING: CRF negative log-likelihood + auxiliary class-weighted
    cross-entropy (gradient for rare classes) + boundary BCE; EMA of weights;
    blank-augmentation for robustness to the lossy inputs.  An ensemble of seeds
    (BiLSTM + BiGRU) is averaged in log-probability / transition space.
  * DECODE CALIBRATION: a boundary-modulated Viterbi with knobs -- transition
    scale `alpha`, switch penalty `lam0`, boundary-head modulation (`gamma`,`b0`)
    that locally eases the switch penalty where a boundary is likely, and
    per-class additive gains `g[8]`.  All knobs are tuned by coordinate ascent on
    a held-out split to DIRECTLY maximize the exact competition metric
    (0.70*MCC8 + 0.30*BoundaryF1).  MCC8 is chance-corrected, so nudging the rare
    classes up recovers diagonal mass and lifts the score.  These knobs are small
    decode-time support transforms fit on train only; predictions come from the
    neural CRF, never a lookup/frequency table.

Compliance: a real model is trained inside this script on every run.  Every
vocabulary / statistic / calibration is fit on TRAIN only; test rows are used
for inference (transform + predict) only.  No train+test concatenation, no
test-derived statistics, no pretrained weights.  Seeds are fixed.  A schema-valid
non-constant placeholder submission is written immediately after reading test,
and a wall-clock guard always leaves time to decode + write the real submission.

Runtime contract: python3 solution.py <public_dir> <submission_out>
"""
import os, sys, time, math, random, csv
import numpy as np

# ----------------------------------------------------------------------------
CFG = dict(
    e_sym       = 64,
    e_cat       = 16,
    hidden      = 256,     # per direction
    layers      = 2,
    dropout     = 0.30,
    lr          = 2.0e-3,
    weight_decay= 1e-4,
    batch       = 64,
    epochs      = 20,
    class_weight_pow = 0.5,
    aux_ce_w    = 0.5,
    bnd_w       = 0.3,     # boundary BCE weight
    bnd_pos_w   = 8.0,     # boundary BCE positive weight (boundaries ~4.7%)
    label_smooth= 0.02,
    ema_decay   = 0.999,
    blank_aug   = 0.10,    # extra fraction of non-blank symbols masked to [X] while training
    grad_clip   = 1.0,
    # ensemble members: (rnn_type, seed) -- BiLSTM + BiGRU for cross-family diversity
    ensemble    = [("lstm", 0), ("gru", 1), ("lstm", 2), ("gru", 3), ("lstm", 4)],
    holdout_frac= 0.10,
    time_soft   = 3000.0,  # stop launching new seeds past this
    time_hard   = 3300.0,  # stop training epochs past this
    max_full    = 320,
    max_sc      = 260,
)

CLASSES = ["rsp", "gnd", "air", "atk", "shd", "grb", "hit", "evd"]
CI = {c: i for i, c in enumerate(CLASSES)}
PAD, UNK, BLK = 0, 1, 2
START_TIME = time.time()


def log(*a):
    print(f"[{time.time()-START_TIME:7.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------
# exact competition metric (used only for decode-knob tuning on the TRAIN holdout)
# ----------------------------------------------------------------------------
def mcc8(conf):
    conf = conf.astype(np.float64)
    N = conf.sum()
    if N == 0:
        return 0.0
    c = np.trace(conf)
    t = conf.sum(1); p = conf.sum(0)
    S_tp = float((t * p).sum()); S_pp = float((p * p).sum()); S_tt = float((t * t).sum())
    den = np.sqrt(N * N - S_pp) * np.sqrt(N * N - S_tt)
    return 0.0 if den == 0 else (c * N - S_tp) / den


def metric_score(y_true, y_pred):
    conf = np.zeros((8, 8), np.int64)
    TP = FP = FN = 0
    for yt, yp in zip(y_true, y_pred):
        yt = np.asarray(yt); yp = np.asarray(yp)
        np.add.at(conf, (yt, yp), 1)
        if len(yt) >= 2:
            Bt = (yt[1:] != yt[:-1]).astype(np.int64)
            Dt = (yp[1:] != yp[:-1]).astype(np.int64)
            TP += int((Bt & Dt).sum()); FP += int(((1 - Bt) & Dt).sum()); FN += int((Bt & (1 - Dt)).sum())
    mcc = mcc8(conf)
    bf1 = 0.0 if TP == 0 else 2 * TP / (2 * TP + FP + FN)
    return {"mcc": mcc, "boundary_f1": bf1, "score": 100.0 * float(np.clip(0.70 * mcc + 0.30 * bf1, 0, 1))}


# ----------------------------------------------------------------------------
# vocab (TRAIN only) + encoding
# ----------------------------------------------------------------------------
def build_sym():
    v = {"<pad>": 0, "<unk>": 1, "[X]": 2}
    for i in range(254):
        v[f"c{i:03d}"] = 3 + i
    return v


def build_cat(values):
    v = {"<unk>": 0}
    for x in sorted(set(values)):
        v[x] = len(v)
    return v


def read_csv_dicts(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def encode(rows, sym, arena_v, ak_v, fk_v, has_states):
    n = len(rows); MF, MS = CFG["max_full"], CFG["max_sc"]
    A = np.zeros((n, MF), np.int64); Fo = np.zeros((n, MF), np.int64)
    Af = np.zeros((n, MF), np.int64); Ff = np.zeros((n, MF), np.int64)
    full_len = np.zeros(n, np.int64); nsc = np.zeros(n, np.int64); ctx = np.zeros(n, np.int64)
    arena = np.zeros(n, np.int64); ak = np.zeros(n, np.int64); fk = np.zeros(n, np.int64)
    Y = np.full((n, MS), -100, np.int64); sid = np.zeros(n, np.int64)
    for r, row in enumerate(rows):
        aa = row["actor_codes"].split(); ff = row["foe_codes"].split()
        L = min(len(aa), MF); full_len[r] = L
        ctx[r] = int(row["context_len"]); nsc[r] = int(row["n_scored"]); sid[r] = int(row["seq_id"])
        la = UNK; lf = UNK
        for j in range(L):
            a = sym.get(aa[j], UNK); f = sym.get(ff[j], UNK)
            A[r, j] = a; Fo[r, j] = f
            if a != BLK:
                la = a
            if f != BLK:
                lf = f
            Af[r, j] = la; Ff[r, j] = lf
        arena[r] = arena_v.get(row["arena"], 0); ak[r] = ak_v.get(row["actor_kind"], 0); fk[r] = fk_v.get(row["foe_kind"], 0)
        if has_states:
            ss = row["states"].split()
            for k in range(min(len(ss), MS)):
                Y[r, k] = CI[ss[k]]
    return dict(A=A, Fo=Fo, Af=Af, Ff=Ff, full_len=full_len, nsc=nsc, ctx=ctx,
                arena=arena, ak=ak, fk=fk, Y=Y, sid=sid)


def subset(d, idx):
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def fwd_fill(X):
    """Forward-fill each row's last non-blank symbol id (UNK before any). Matches the
    encode() semantics; used to recompute the fill channel after blank-augmentation so
    training and test see the same imputation behavior."""
    B, L = X.shape
    isreal = (X != PAD) & (X != BLK)
    pos = np.where(isreal, np.arange(L)[None, :], -1)
    np.maximum.accumulate(pos, axis=1, out=pos)
    gathered = np.take_along_axis(X, np.clip(pos, 0, None), axis=1)
    return np.where(pos >= 0, gathered, UNK).astype(np.int64)


# ----------------------------------------------------------------------------
# model + CRF
# ----------------------------------------------------------------------------
def build_model_classes():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Tagger(nn.Module):
        def __init__(self, n_sym, n_arena, n_ak, n_fk, rnn_type="lstm"):
            super().__init__()
            e_sym, e_cat, hid = CFG["e_sym"], CFG["e_cat"], CFG["hidden"]
            self.E = nn.Embedding(n_sym, e_sym, padding_idx=0)  # shared actor/foe
            self.arena = nn.Embedding(n_arena, 8)
            self.ak = nn.Embedding(n_ak, e_cat)
            self.fk = nn.Embedding(n_fk, e_cat)
            self.match = nn.Sequential(nn.Linear(e_cat * 2, 32), nn.GELU(), nn.Linear(32, 16))
            in_dim = e_sym * 5 + 2 + 8 + e_cat * 2 + 16
            self.inp = nn.Linear(in_dim, hid); self.lnorm = nn.LayerNorm(hid)
            self.c3 = nn.Conv1d(hid, hid, 3, padding=1); self.c5 = nn.Conv1d(hid, hid, 5, padding=2)
            self.drop = nn.Dropout(CFG["dropout"])
            R = nn.LSTM if rnn_type == "lstm" else nn.GRU
            self.rnn = R(hid, hid, num_layers=CFG["layers"], batch_first=True,
                         bidirectional=True, dropout=CFG["dropout"] if CFG["layers"] > 1 else 0.0)
            self.emit = nn.Sequential(nn.Linear(hid * 2, 256), nn.GELU(), nn.LayerNorm(256),
                                      nn.Dropout(0.2), nn.Linear(256, 8))
            self.bhead = nn.Sequential(nn.Linear(hid * 2, 128), nn.GELU(), nn.Linear(128, 1))
            self.trans = nn.Parameter(torch.zeros(8, 8))
            self.start = nn.Parameter(torch.zeros(8))
            self.end = nn.Parameter(torch.zeros(8))

        def feats(self, A, Fo, Af, Ff, arena, ak, fk):
            B, Lf = A.shape
            a = self.E(A); f = self.E(Fo)
            e = torch.cat([
                a, f, a * f, self.E(Af), self.E(Ff),
                (A == BLK).float().unsqueeze(-1), (Fo == BLK).float().unsqueeze(-1),
                self.arena(arena)[:, None, :].expand(B, Lf, -1),
                self.ak(ak)[:, None, :].expand(B, Lf, -1),
                self.fk(fk)[:, None, :].expand(B, Lf, -1),
                self.match(torch.cat([self.ak(ak), self.fk(fk)], -1))[:, None, :].expand(B, Lf, -1),
            ], dim=-1)
            h = F.gelu(self.inp(e)); h = self.lnorm(h)
            hc = h.transpose(1, 2); hc = F.relu(self.c3(hc)) + F.relu(self.c5(hc)); h = h + hc.transpose(1, 2)
            h = self.drop(h)
            out, _ = self.rnn(h); out = self.drop(out)
            return out

        def heads(self, h):
            return self.emit(h), self.bhead(h).squeeze(-1)

    def crf_nll(em, tags, mask, trans, start, end):
        B, S, K = em.shape
        idx = torch.arange(B, device=em.device)
        score = start[tags[:, 0]] + em[idx, 0, tags[:, 0]]
        for t in range(1, S):
            e_t = em[:, t, :].gather(1, tags[:, t:t + 1]).squeeze(1)
            tr = trans[tags[:, t - 1], tags[:, t]]
            score = score + (e_t + tr) * mask[:, t].float()
        last = mask.sum(1) - 1
        score = score + end[tags[idx, last]]
        alpha = start.unsqueeze(0) + em[:, 0, :]
        for t in range(1, S):
            a = alpha.unsqueeze(2) + trans.unsqueeze(0) + em[:, t, :].unsqueeze(1)
            new = torch.logsumexp(a, dim=1)
            m = mask[:, t].unsqueeze(1)
            alpha = torch.where(m, new, alpha)
        alpha = alpha + end.unsqueeze(0)
        return (torch.logsumexp(alpha, dim=1) - score).mean()

    class EMA:
        def __init__(self, model, decay):
            self.decay = decay
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

        def update(self, model):
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    self.shadow[k] = v.detach().clone()

        def copy_to(self, model):
            model.load_state_dict(self.shadow, strict=True)

    return torch, nn, F, Tagger, crf_nll, EMA


# ----------------------------------------------------------------------------
# boundary-modulated batched Viterbi (numpy) + decode-knob tuning
# ----------------------------------------------------------------------------
_OFF8 = (~np.eye(8, dtype=bool)).astype(np.float32)


def viterbi_mod(P, B, Ls, T, start, end, g, alpha, lam0, gamma, b0):
    """P (R,S,8) log-probs; B (R,S) boundary probs; per-position off-diagonal
    switch cost = -lam0 + gamma*(B[:,t]-b0) (gamma=0 -> static -lam0 penalty)."""
    R, S, K = P.shape
    E = P + g[None, None, :]
    Tt = (alpha * T).astype(np.float32)
    dp = np.empty((R, S, K), np.float32); ptr = np.empty((R, S, K), np.int8)
    dp[:, 0, :] = start[None, :] + E[:, 0, :]
    static = Tt - lam0 * _OFF8 if gamma == 0.0 else None
    for t in range(1, S):
        if gamma == 0.0:
            a = dp[:, t - 1, :, None] + static[None, :, :]
        else:
            adj = (-lam0 + gamma * (B[:, t] - b0)).astype(np.float32)  # (R,)
            trans_t = Tt[None, :, :] + _OFF8[None, :, :] * adj[:, None, None]
            a = dp[:, t - 1, :, None] + trans_t
        ptr[:, t, :] = a.argmax(1)
        dp[:, t, :] = a.max(1) + E[:, t, :]
    out = np.zeros((R, S), np.int64)
    for r in range(R):
        L = int(Ls[r])
        last = int((dp[r, L - 1, :] + end).argmax()); out[r, L - 1] = last
        for t in range(L - 1, 0, -1):
            last = int(ptr[r, t, last]); out[r, t - 1] = last
    return [out[r, :int(Ls[r])].tolist() for r in range(R)]


def tune_knobs(P, B, Ls, T, start, end, y_true):
    alpha, lam0, gamma, b0, g = 1.0, 0.0, 0.0, 0.1, np.zeros(8)
    def sc(al, la, ga, bz, gg):
        return metric_score(y_true, viterbi_mod(P, B, Ls, T, start, end, gg, al, la, ga, bz))["score"]
    best = sc(alpha, lam0, gamma, b0, g)
    for _ in range(4):
        for al in [0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.6]:
            s = sc(al, lam0, gamma, b0, g)
            if s > best: best, alpha = s, al
        for la in [-1.2, -0.8, -0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6, 1.0, 1.5, 2.0]:
            s = sc(alpha, la, gamma, b0, g)
            if s > best: best, lam0 = s, la
        for ga in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
            s = sc(alpha, lam0, ga, b0, g)
            if s > best: best, gamma = s, ga
        for bz in [0.0, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]:
            s = sc(alpha, lam0, gamma, bz, g)
            if s > best: best, b0 = s, bz
        for c in range(8):
            for v in [-1.0, -0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6, 1.0]:
                gg = g.copy(); gg[c] = v
                s = sc(alpha, lam0, gamma, b0, gg)
                if s > best: best, g = s, gg
    return dict(alpha=alpha, lam0=lam0, gamma=gamma, b0=b0, g=g, best=best)


# ----------------------------------------------------------------------------
# training + inference
# ----------------------------------------------------------------------------
def make_batches(order, bs):
    b = [order[i:i + bs] for i in range(0, len(order), bs)]
    random.shuffle(b)
    return b


def train_seed(torch, Tagger, crf_nll, EMA, F, tr, vs, rnn_type, seed, dev, cw):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    m = Tagger(*vs, rnn_type=rnn_type).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"], betas=(0.9, 0.98))
    order = np.argsort(tr["full_len"]); spe = math.ceil(len(order) / CFG["batch"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["lr"], total_steps=spe * CFG["epochs"], pct_start=0.08)
    ema = EMA(m, CFG["ema_decay"])
    cwt = torch.tensor(cw, dtype=torch.float32, device=dev)
    pw = torch.tensor([CFG["bnd_pos_w"]], device=dev)
    ba = CFG["blank_aug"]

    for ep in range(CFG["epochs"]):
        if time.time() - START_TIME > CFG["time_hard"]:
            log(f"  [budget-hard] stop {rnn_type}#{seed} at ep{ep}")
            break
        m.train(); bb = make_batches(order, CFG["batch"]); tot = 0.0
        for idx in bb:
            Lm = int(tr["full_len"][idx].max()); Sm = int(tr["nsc"][idx].max())
            A_np = tr["A"][idx, :Lm]; Fo_np = tr["Fo"][idx, :Lm]
            if ba > 0 and ep < CFG["epochs"] - 2:
                A_np = A_np.copy(); Fo_np = Fo_np.copy()
                for X in (A_np, Fo_np):
                    X[(X > BLK) & (np.random.random(X.shape) < ba)] = BLK
                Af_np = fwd_fill(A_np); Ff_np = fwd_fill(Fo_np)   # recompute imputation after masking
            else:
                Af_np = tr["Af"][idx, :Lm]; Ff_np = tr["Ff"][idx, :Lm]
            A = torch.as_tensor(A_np, device=dev); Fo = torch.as_tensor(Fo_np, device=dev)
            Af = torch.as_tensor(Af_np, device=dev); Ff = torch.as_tensor(Ff_np, device=dev)
            arena = torch.as_tensor(tr["arena"][idx], device=dev)
            ak = torch.as_tensor(tr["ak"][idx], device=dev)
            fk = torch.as_tensor(tr["fk"][idx], device=dev)
            Y = torch.as_tensor(tr["Y"][idx, :Sm], device=dev)
            h = m.feats(A, Fo, Af, Ff, arena, ak, fk)
            emf, blog = m.heads(h)
            off = int(tr["ctx"][idx][0])       # context_len is a constant 60
            em = emf[:, off:off + Sm, :]
            bl = blog[:, off:off + Sm]
            nsc_t = torch.as_tensor(tr["nsc"][idx], device=dev)
            mask = torch.arange(Sm, device=dev)[None, :] < nsc_t[:, None]
            tags = Y.clone(); tags[tags < 0] = 0
            loss = crf_nll(em, tags, mask, m.trans, m.start, m.end)
            ce = F.cross_entropy(em.reshape(-1, 8), Y.reshape(-1), weight=cwt,
                                 ignore_index=-100, label_smoothing=CFG["label_smooth"])
            bt = torch.zeros_like(bl); bt[:, 1:] = (Y[:, 1:] != Y[:, :-1]).float()
            bmask = mask.clone(); bmask[:, 0] = False; bmask = bmask & (Y != -100)
            bce = F.binary_cross_entropy_with_logits(bl[bmask], bt[bmask], pos_weight=pw)
            loss = loss + CFG["aux_ce_w"] * ce + CFG["bnd_w"] * bce
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), CFG["grad_clip"])
            opt.step(); sch.step(); ema.update(m)
            tot += float(loss.item())
        log(f"  {rnn_type}#{seed} ep{ep} loss={tot/len(bb):.4f}")
    ema.copy_to(m)
    return m


def infer(torch, m, data, dev):
    m.eval(); n = len(data["A"]); order = np.argsort(data["full_len"]); bs = 128
    S = CFG["max_sc"]
    Pout = np.zeros((n, S, 8), np.float32); Bout = np.zeros((n, S), np.float32)
    with torch.no_grad():
        for i in range(0, n, bs):
            idx = order[i:i + bs]; Lm = int(data["full_len"][idx].max())
            g = lambda k: torch.as_tensor(data[k][idx, :Lm], device=dev)
            h = m.feats(g("A"), g("Fo"), g("Af"), g("Ff"),
                        torch.as_tensor(data["arena"][idx], device=dev),
                        torch.as_tensor(data["ak"][idx], device=dev),
                        torch.as_tensor(data["fk"][idx], device=dev))
            emf, blog = m.heads(h)
            emf = torch.log_softmax(emf, -1).float().cpu().numpy()
            bp = torch.sigmoid(blog).float().cpu().numpy()
            for bi, r in enumerate(idx):
                c = int(data["ctx"][r]); s = int(data["nsc"][r])
                Pout[r, :s] = emf[bi, c:c + s]; Bout[r, :s] = bp[bi, c:c + s]
    return Pout, Bout, m.trans.detach().cpu().numpy(), m.start.detach().cpu().numpy(), m.end.detach().cpu().numpy()


# ----------------------------------------------------------------------------
# submission
# ----------------------------------------------------------------------------
def write_submission(path, seq_ids, states_strs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id", "states"])
        for sid, st in zip(seq_ids, states_strs):
            w.writerow([sid, st])


def placeholder_states(n):
    return " ".join(CLASSES[(k // 14) % 8] for k in range(n))


def write_all(paths, seq_ids, states_strs):
    for p in paths:
        try:
            write_submission(p, seq_ids, states_strs)
        except Exception as e:
            log("write skipped", p, e)


# ----------------------------------------------------------------------------
def main():
    public_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset/public"
    sub_out = sys.argv[2] if len(sys.argv) > 2 else "./working/submission.csv"
    # Write ONLY to the platform-supplied submission path (PROMPT.md: do not hardcode
    # paths; touch only public_dir for read and submission_out for write). The early
    # placeholder below guarantees this path exists from the start.
    out_paths = [sub_out]

    log("reading test")
    test_rows = read_csv_dicts(os.path.join(public_dir, "test.csv"))
    test_ids = [int(r["seq_id"]) for r in test_rows]
    write_all(out_paths, test_ids, [placeholder_states(int(r["n_scored"])) for r in test_rows])
    log(f"placeholder written ({len(test_rows)} rows)")

    log("reading train")
    train_rows = read_csv_dicts(os.path.join(public_dir, "train.csv"))

    sym = build_sym()
    arena_v = build_cat(r["arena"] for r in train_rows)
    ak_v = build_cat(r["actor_kind"] for r in train_rows)
    fk_v = build_cat(r["foe_kind"] for r in train_rows)
    vs = (len(sym), len(arena_v), len(ak_v), len(fk_v))
    log("vocab sizes", vs)

    full = encode(train_rows, sym, arena_v, ak_v, fk_v, True)
    te = encode(test_rows, sym, arena_v, ak_v, fk_v, False)

    n = len(train_rows); rng = np.random.default_rng(12345)
    perm = rng.permutation(n); nhold = int(n * CFG["holdout_frac"])
    hold_idx = np.sort(perm[:nhold]); tr_idx = np.sort(perm[nhold:])
    tr = subset(full, tr_idx); ho = subset(full, hold_idx)
    log(f"train {len(tr_idx)}  holdout {len(hold_idx)}  test {len(te['A'])}")

    cnt = np.bincount(tr["Y"][tr["Y"] >= 0], minlength=8).astype(np.float64)
    cw = (np.median(cnt) / cnt) ** CFG["class_weight_pow"]; cw = cw / cw.mean()
    log("class weights", np.round(cw, 3))

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available()
                       else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                             else "cpu"))
    log("device", dev)
    torch, nn, F, Tagger, crf_nll, EMA = build_model_classes()

    ho_true = [ho["Y"][r, :int(ho["nsc"][r])].tolist() for r in range(len(ho["A"]))]
    S = CFG["max_sc"]
    accHoP = np.zeros((len(ho["A"]), S, 8)); accHoB = np.zeros((len(ho["A"]), S))
    accTeP = np.zeros((len(te["A"]), S, 8)); accTeB = np.zeros((len(te["A"]), S))
    accT = np.zeros((8, 8)); accS = np.zeros(8); accE = np.zeros(8); nm = 0
    last_kn = None

    for rnn_type, seed in CFG["ensemble"]:
        if time.time() - START_TIME > CFG["time_soft"]:
            log("[budget-soft] stop launching new seeds")
            break
        log(f"=== train {rnn_type}#{seed} ===")
        m = train_seed(torch, Tagger, crf_nll, EMA, F, tr, vs, rnn_type, seed, dev, cw)
        hP, hB, T, st, en = infer(torch, m, ho, dev)
        tP, tB, _, _, _ = infer(torch, m, te, dev)
        accHoP += hP; accHoB += hB; accTeP += tP; accTeB += tB
        accT += T; accS += st; accE += en; nm += 1
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        Ph = accHoP / nm; Bh = accHoB / nm; Pt = accTeP / nm; Bt = accTeB / nm
        T_ = accT / nm; S_ = accS / nm; E_ = accE / nm
        # tune decode knobs on the holdout ensemble; if past the hard budget, reuse the
        # previous member's knobs so the final decode+write always lands inside budget.
        if time.time() - START_TIME < CFG["time_hard"] or last_kn is None:
            kn = tune_knobs(Ph, Bh, ho["nsc"], T_, S_, E_, ho_true)
            last_kn = kn
        else:
            kn = last_kn
            log("  [budget] reuse previous decode knobs (skip re-tune)")
        hpred = viterbi_mod(Ph, Bh, ho["nsc"], T_, S_, E_, kn["g"], kn["alpha"], kn["lam0"], kn["gamma"], kn["b0"])
        scr = metric_score(ho_true, hpred)
        log(f"  holdout after {nm} model(s): score={scr['score']:.3f} mcc={scr['mcc']:.4f} bf1={scr['boundary_f1']:.4f} "
            f"| alpha={kn['alpha']} lam0={kn['lam0']} gamma={kn['gamma']} b0={kn['b0']} g={np.round(kn['g'],2).tolist()}")
        te_pred = viterbi_mod(Pt, Bt, te["nsc"], T_, S_, E_, kn["g"], kn["alpha"], kn["lam0"], kn["gamma"], kn["b0"])
        states = [" ".join(CLASSES[i] for i in p) for p in te_pred]
        write_all(out_paths, [int(x) for x in te["sid"]], states)
        log(f"  submission written ({nm} model ensemble)")

    if nm == 0:
        log("WARNING: no model trained; placeholder stands")
        return
    log("done")


if __name__ == "__main__":
    main()
