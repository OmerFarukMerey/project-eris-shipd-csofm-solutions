"""Route and Read: Joint Tool Selection and Readiness.

One end-to-end CPU script:
  1. parse the request text, the supplied-argument JSON and the three candidate schemas;
  2. compute per-candidate schema-validation counts (missing required keys, enum / type /
     range violations, top level and nested) as input features for a learned readiness head;
  3. fine-tune three multilingual pretrained backbones that score the request against each
     candidate's description text: two bi-encoders (sentence encoders; listwise softmax over
     the row's three candidates plus in-batch negatives) and one cross-encoder (a pretrained
     reranker reading request and candidate together; listwise softmax over the three);
     each backbone carries its own readiness head and outputs the joint 6-way distribution
     p(label = k + 3 * j) = p_route(k) * p_ready(j);
  4. early stopping: every epoch is scored on a stratified 20% train holdout, and each
     backbone keeps the test predictions of its best holdout epoch; the blend weights of the
     three backbones are searched on the same holdout against joint accuracy.
Run: python3 solution.py <public_dir> <submission_out>
"""
import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import StratifiedKFold

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cpu")

SEED = 0
# (backbone, kind, candidate text view, max candidate tokens; for "cross" the max request+candidate pair tokens)
#   kind "bi":    sentence encoder, request and candidate embedded separately, cosine score
#   kind "cross": pretrained reranker, request and candidate read jointly, relevance logit
#   view "root+params": root description followed by the schema's parameter descriptions
#   view "root":        root description only (cheaper; the larger backbones use it)
BACKBONES = [
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "bi", "root+params", 128),
    ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "bi", "root", 64),
    ("BAAI/bge-reranker-base", "cross", "root", 96),
]
MAX_EPOCHS = 2          # fixed plan; the epoch whose predictions are kept is chosen on the holdout
BATCH_ROWS = 16
ENC_LR = 2e-5
HEAD_LR = 1e-2
WARMUP_DIV = 10         # linear warm-up over the first tenth of the schedule, then linear decay
TEMP = 0.05             # cosine-similarity temperature of the bi-encoder
MAX_LEN_Q = 64
HOLDOUT_FOLDS = 5       # holdout = first fold of a stratified 5-fold split (20% of train)
BLEND_STEPS = 10        # blend weights live on the simplex grid with step 1/10

PYTYPES = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
           "array": (list,), "object": (dict,)}
NUMERIC = ("integer", "number")
N_VALIDATION_FEATS = 7


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- parsing
def split_dialogue(text):
    """Dialogue = free-text request followed by one JSON object of supplied arguments.

    Chinese requests are rendered as a list of short fragments joined by the enumeration
    comma; the commas are removed so the tokenizer sees contiguous text."""
    j = text.find("{")
    return text[:j].strip().replace("、", ""), json.loads(text[j:])


def walk(schema, val, depth, acc):
    """Recursive JSON-schema check of a supplied value; accumulates violation counts.

    acc = [missing_top, missing_nested, n_required, type_bad, enum_bad, range_bad, n_checked]
    """
    t = schema.get("type")
    isnum = isinstance(val, (int, float)) and not isinstance(val, bool)
    type_ok = isinstance(val, PYTYPES.get(t, (object,))) and not (t in NUMERIC and isinstance(val, bool))
    x = float({True: val, False: 0}[isnum])
    acc[3] += float(not type_ok)
    acc[4] += float("enum" in schema and val not in schema.get("enum", []))
    acc[5] += float(isnum and "minimum" in schema and x < schema.get("minimum", 0))
    acc[5] += float(isnum and "maximum" in schema and x > schema.get("maximum", 0))
    acc[6] += 1.0
    d = {True: val, False: {}}[isinstance(val, dict)]
    req = schema.get("required", []) * isinstance(val, dict)
    acc[min(depth, 1)] += sum(float(r not in d) for r in req)
    acc[2] += len(req)
    props = schema.get("properties", {})
    for k in [k for k in d if k in props]:
        walk(props[k], d[k], depth + 1, acc)
    items = [v for v in {True: val, False: []}[isinstance(val, list)] if "items" in schema]
    for v in items:
        walk(schema["items"], v, depth + 1, acc)
    return acc


def param_descriptions(schema, out):
    """Descriptions of every (nested) parameter of a schema, in document order."""
    for v in schema.get("properties", {}).values():
        out.extend([v.get("description")] * isinstance(v.get("description"), str))
        param_descriptions(v, out)
    items = schema.get("items")
    for it in [items] * isinstance(items, dict):
        out.extend([it.get("description")] * isinstance(it.get("description"), str))
        param_descriptions(it, out)
    return out


def parse_frame(df):
    """Row-local parsing only: nothing here is fitted or aggregated across rows."""
    reqs, roots, full, feats = [], [], [], []
    for dialogue, specs_json in zip(df["dialogue"].astype(str), df["tool_specifications"].astype(str)):
        request, args = split_dialogue(dialogue)
        specs = json.loads(specs_json)
        reqs.append(request)
        root = [str(s.get("description", "")) for s in specs]
        params = [param_descriptions(s.get("parameters", {}), []) for s in specs]
        # a parameter description that all three candidates of the row share cannot discriminate between them
        shared = set(params[0]) & set(params[1]) & set(params[2])
        roots.append(root)
        full.append([r + " | " + " ; ".join(p for p in ps if p not in shared) for r, ps in zip(root, params)])
        cand = np.stack([walk(s.get("parameters", {}), args, 0, np.zeros(N_VALIDATION_FEATS)) for s in specs])
        # readiness is a row-level property: summarise the three candidates' validation counts
        feats.append(np.log1p(np.concatenate([cand.mean(0), cand.max(0), cand.min(0)])))
    return {"req": reqs, "root": roots, "root+params": full, "feats": np.asarray(feats, dtype=np.float32)}


# ----------------------------------------------------------------------------- model
class JointRouter(torch.nn.Module):
    def __init__(self, name, loader, n_feats):
        super().__init__()
        self.encoder = loader.from_pretrained(name)
        self.ready_head = torch.nn.Sequential(torch.nn.Linear(n_feats, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))

    def embed(self, batch):
        out = self.encoder(**batch).last_hidden_state
        m = batch["attention_mask"].unsqueeze(-1).float()
        return F.normalize((out * m).sum(1) / m.sum(1), dim=-1)


def encode_batch(tok, model, data, view, max_len_t, idx):
    q = tok([data["req"][i] for i in idx], padding=True, truncation=True, max_length=MAX_LEN_Q, return_tensors="pt")
    t = tok([data[view][i][k] for i in idx for k in range(3)], padding=True, truncation=True,
            max_length=max_len_t, return_tensors="pt")
    return model.embed(q.to(DEVICE)), model.embed(t.to(DEVICE)).view(len(idx), 3, -1)


def bi_scores(tok, model, data, view, max_len_t, idx):
    qe, te = encode_batch(tok, model, data, view, max_len_t, idx)
    return (qe.unsqueeze(1) * te).sum(-1) / TEMP


def bi_train_loss(tok, model, data, view, max_len_t, idx, route_y):
    """Listwise over the row's 3 candidates + the other rows' candidates as in-batch negatives."""
    qe, te = encode_batch(tok, model, data, view, max_len_t, idx)
    logits = qe @ te.reshape(-1, te.shape[-1]).T / TEMP
    return F.cross_entropy(logits, route_y[idx] + 3 * torch.arange(len(idx)))


def cross_scores(tok, model, data, view, max_len_t, idx):
    pairs = tok([data["req"][i] for i in idx for _ in range(3)],
                [data[view][i][k] for i in idx for k in range(3)],
                padding=True, truncation="only_second", max_length=max_len_t, return_tensors="pt")
    return model.encoder(**pairs.to(DEVICE)).logits[:, 0].view(len(idx), 3)


def cross_train_loss(tok, model, data, view, max_len_t, idx, route_y):
    """Listwise softmax over the row's 3 candidates."""
    return F.cross_entropy(cross_scores(tok, model, data, view, max_len_t, idx), route_y[idx])


# kind -> (weights loader, training loss, inference scores)
KINDS = {
    "bi": (AutoModel, bi_train_loss, bi_scores),
    "cross": (AutoModelForSequenceClassification, cross_train_loss, cross_scores),
}


def joint_log_probs(route_logits, ready_logits):
    """log p(label = k + 3*j) = log p_route(k) + log p_ready(j); j=0 ready, j=1 not ready."""
    lr = F.log_softmax(route_logits, -1)
    ly = F.log_softmax(ready_logits, -1)
    return torch.cat([lr + ly[:, :1], lr + ly[:, 1:]], 1)


def predict(tok, model, kind, data, view, max_len_t, idx):
    """Per-row inference: each row's scores depend only on that row's request, candidates and features."""
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), 64):
            b = idx[s:s + 64]
            route = KINDS[kind][2](tok, model, data, view, max_len_t, b)
            ready = model.ready_head(torch.from_numpy(data["feats"][b]).to(DEVICE))
            out.append(joint_log_probs(route, ready).cpu().numpy())
    return np.concatenate(out)


def train_backbone(name, kind, view, max_len_t, data, y6, tr_idx, eval_sets):
    """Fine-tune one backbone on training rows tr_idx for MAX_EPOCHS epochs.

    eval_sets: list of (data, idx) predicted after every epoch.
    Returns, per epoch, the joint log-probs for every eval set."""
    seed_everything(SEED)
    tok = AutoTokenizer.from_pretrained(name)
    loader, train_loss, _ = KINDS[kind]
    model = JointRouter(name, loader, data["feats"].shape[1]).to(DEVICE)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": ENC_LR, "weight_decay": 0.01},
        {"params": model.ready_head.parameters(), "lr": HEAD_LR, "weight_decay": 0.0},
    ])
    steps_per_epoch = (len(tr_idx) + BATCH_ROWS - 1) // BATCH_ROWS
    total = MAX_EPOCHS * steps_per_epoch
    warm = total // WARMUP_DIV
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / (total - warm + 1)))
    route_y = torch.from_numpy(y6 % 3).long()
    ready_y = torch.from_numpy(y6 // 3).long()
    rng = np.random.RandomState(SEED)
    history = []
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for s in range(0, len(perm), BATCH_ROWS):
            b = perm[s:s + BATCH_ROWS]
            ready = model.ready_head(torch.from_numpy(data["feats"][b]).to(DEVICE))
            loss = train_loss(tok, model, data, view, max_len_t, b, route_y) + F.cross_entropy(ready, ready_y[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        history.append([predict(tok, model, kind, d, view, max_len_t, idx) for d, idx in eval_sets])
        print(f"  [{name.split('/')[-1]}] epoch {ep + 1}/{MAX_EPOCHS} train loss {tot / len(tr_idx):.4f}", flush=True)
    return history


def report(tag, lp, y6, lang):
    pred = lp.argmax(1)
    acc = pred == y6
    print(f"  {tag}: joint acc {acc.mean():.4f} | route acc {(pred % 3 == y6 % 3).mean():.4f} | "
          f"ready acc {(pred // 3 == y6 // 3).mean():.4f} | en {acc[lang == 'en'].mean():.4f} "
          f"zh {acc[lang == 'zh'].mean():.4f}", flush=True)
    return acc.mean()


def main():
    public_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    # schema-valid placeholder, overwritten at the end
    pd.DataFrame({"id": test["id"], "prediction": 0}).to_csv(out_path, index=False)
    print(f"train {train.shape} test {test.shape}", flush=True)

    tr_data = parse_frame(train)
    te_data = parse_frame(test)
    y6 = train["target"].astype(int).to_numpy()
    lang = train["language"].astype(str).to_numpy()
    te_idx = np.arange(len(test))

    strat = train["target"].astype(str) + "_" + train["language"].astype(str)
    fit_idx, hold_idx = next(StratifiedKFold(HOLDOUT_FOLDS, shuffle=True, random_state=SEED)
                             .split(np.zeros(len(train)), strat))
    print(f"plan: {len(BACKBONES)} backbones x {MAX_EPOCHS} epochs on {len(fit_idx)} rows, "
          f"early stopping on a {len(hold_idx)}-row holdout", flush=True)

    hold_lp, test_lp = [], []
    for name, kind, view, max_len_t in BACKBONES:
        hist = train_backbone(name, kind, view, max_len_t, tr_data, y6, fit_idx,
                              [(tr_data, hold_idx), (te_data, te_idx)])
        accs = [report(f"{name.split('/')[-1]} epoch {e + 1} holdout", h[0], y6[hold_idx], lang[hold_idx])
                for e, h in enumerate(hist)]
        e_star = int(np.argmax(accs))
        print(f"  {name.split('/')[-1]}: keeping epoch {e_star + 1}", flush=True)
        hold_lp.append(hist[e_star][0])
        test_lp.append(hist[e_star][1])

    # blend weights of the three backbones' joint log-probs: simplex grid searched on the holdout
    grid = np.array([(i, j, BLEND_STEPS - i - j) for i in range(BLEND_STEPS + 1)
                     for j in range(BLEND_STEPS + 1 - i)], dtype=np.float64) / BLEND_STEPS
    hold_stack = np.stack(hold_lp)
    blend_acc = np.array([(np.tensordot(w, hold_stack, 1).argmax(1) == y6[hold_idx]).mean() for w in grid])
    w_star = grid[int(np.argmax(blend_acc))]
    order = np.argsort(-blend_acc, kind="stable")[:5]
    print("  blend search (top 5 of %d): %s" % (len(grid), [(grid[i].round(1).tolist(), round(float(blend_acc[i]), 4))
                                                          for i in order]), flush=True)
    print(f"  equal-weight blend holdout acc {(hold_stack.mean(0).argmax(1) == y6[hold_idx]).mean():.4f}", flush=True)
    report(f"blend w={w_star.round(1).tolist()} holdout", np.tensordot(w_star, hold_stack, 1), y6[hold_idx], lang[hold_idx])

    pred = np.tensordot(w_star, np.stack(test_lp), 1).argmax(1).astype(int)
    sub = pd.DataFrame({"id": test["id"].to_numpy(), "prediction": pred})
    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(sub)}", flush=True)


if __name__ == "__main__":
    main()
