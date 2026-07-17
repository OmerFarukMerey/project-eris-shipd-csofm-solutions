"""Train-only program synthesizer for regulatory citation repair.

The amendment note's repair family is learned by a word/bigram logistic
classifier.  Four non-trivial repair families use conditional-logit candidate
rankers trained on hierarchy, citation-state, node-tag, size, and positional
features.  Scope promotion uses the row's unique scope node.

All estimators and feature vocabularies are fit before test.csv is read.  Test
rows are then transformed and predicted independently; no cross-row test
statistic or adaptation is used.
"""
import sys
import json
import re
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

SEED = 42


SEM = []
LEN_VALUES = []
SBORD = {"short": 0, "medium": 1, "long": 2, "very_long": 3}



def parse_rows(df, op_predictions=None):
    rows = []
    has_target = "target_program" in df.columns
    if not has_target and op_predictions is None:
        raise ValueError("test rows require predicted operation families")
    for i, (_, rec) in enumerate(df.iterrows()):
        note = rec["amendment_note"]
        mcit = re.search(r"(C\d+)", note)
        manch = re.search(r"[Aa]nchor(?:\snode|=)?\s*(N\d+)", note)
        nodes = json.loads(rec["node_cards"])
        cits = json.loads(rec["citation_cards"])
        tp = json.loads(rec["target_program"]) if has_target else None
        op = tp["ops"][0]["op"] if tp is not None else str(op_predictions[i])
        row = {
            "id": rec["id"],
            "nodes": nodes,
            "cits": cits,
            "op": op,
            "cit": mcit.group(1) if mcit else None,
            "anchor": manch.group(1) if manch else None,
            "nodemap": {n["node"]: n for n in nodes},
            "citmap": {c["citation"]: c for c in cits},
            "sections": sorted(
                [n for n in nodes if n["kind"] == "section"],
                key=lambda n: n["ordinal"],
            ),
            "scope": next((n for n in nodes if n["kind"] == "scope"), None),
        }
        if tp is not None:
            row["tp"] = tp
        rows.append(row)
    return rows


def fit_feature_schema(rows):
    """Fit node-tag feature vocabularies from training rows only."""
    global SEM, LEN_VALUES
    tags = {tag for r in rows for n in r["sections"] for tag in n["heading_tags"]}
    SEM = sorted(tag for tag in tags if not tag.startswith("len"))
    LEN_VALUES = sorted(
        {int(tag[3:]) for tag in tags if tag.startswith("len") and tag[3:].isdigit()}
    )


def _ctx(r):
    om = {n["node"]: n["ordinal"] for n in r["nodes"]}
    c = r["citmap"][r["cit"]]
    cur_ords = [om[t] for t in c["targets"] if t in om]
    other_tgt_count = collections.Counter()
    other_src = set()
    for c2 in r["cits"]:
        if c2["citation"] == r["cit"]:
            continue
        other_src.add(c2["source"])
        for t in c2["targets"]:
            other_tgt_count[t] += 1
    scope_tags = set(r["scope"]["heading_tags"]) if r["scope"] else set()
    return om, c, cur_ords, other_tgt_count, other_src, scope_tags


def local_block(r, node, ctx):
    _, _, _, _, _, scope_tags = ctx
    n = r["nodemap"][node]
    tags = set(n["heading_tags"])
    f = [1.0 if s in tags else 0.0 for s in SEM]
    lenv = {
        int(x[3:]) for x in tags if x.startswith("len") and x[3:].isdigit()
    }
    f.append(1.0 if lenv else 0.0)
    f += [1.0 if k in lenv else 0.0 for k in LEN_VALUES]
    f.append(len(tags) / 3.0)
    f += [
        1.0 if n["size_bucket"] == size else 0.0
        for size in ["short", "medium", "long", "very_long"]
    ]
    f.append(1.0 if tags - scope_tags else 0.0)
    return f


def intrinsic_block(r, node):
    """Non-linear node traits learned by the node-choice ranker."""
    n = r["nodemap"][node]
    tags = set(n["heading_tags"])
    semantic_count = sum(tag in SEM for tag in tags)
    f = [1.0 if len(tags) == k else 0.0 for k in range(1, 5)]
    f += [1.0 if semantic_count == k else 0.0 for k in range(4)]
    f += [
        1.0 if tag in tags and n["size_bucket"] == size else 0.0
        for tag in SEM
        for size in ["short", "medium", "long", "very_long"]
    ]
    return f


def pos_block(r, node, ctx):
    om, c, cur_ords, other_tgt_count, other_src, _ = ctx
    n = r["nodemap"][node]
    nsec = max(1, len(r["sections"]))
    o = om[node]
    ao = om.get(r["anchor"], om[c["source"]])
    so = om[c["source"]]
    sda = o - ao
    f = []
    f += [1.0 if sda == k else 0.0 for k in range(-10, 13)]
    f.append(1.0 if sda <= -11 else 0.0)
    f.append(1.0 if sda >= 13 else 0.0)
    f.append(abs(sda) / nsec)
    f.append((o - 1) / max(1, nsec - 1))
    f.append(abs(o - so) / nsec)
    f.append(1.0 if node == c["source"] else 0.0)
    f.append(1.0 if node == r["anchor"] else 0.0)
    if cur_ords:
        dc = min(abs(o - x) for x in cur_ords)
        f.append(dc / nsec)
        f += [1.0 if dc == k else 0.0 for k in range(4)]
        f.append(1.0 if node in c["targets"] else 0.0)
        f.append(1.0 if min(cur_ords) <= o <= max(cur_ords) else 0.0)
    else:
        f += [0.0] * 7
    f.append(other_tgt_count.get(node, 0) / 2.0)
    f.append(1.0 if node in other_src else 0.0)
    return f


def drop_extra(r, tgts, i, om):
    ao = om.get(r["anchor"], 0)
    sizes = [SBORD[r["nodemap"][t]["size_bucket"]] for t in tgts]
    dists = [abs(om[t] - ao) for t in tgts]
    ords = [om[t] for t in tgts]
    return [
        i / max(1, len(tgts) - 1),
        1.0 if sizes[i] == max(sizes) else 0.0,
        1.0 if sizes[i] == min(sizes) else 0.0,
        1.0 if dists[i] == max(dists) else 0.0,
        1.0 if dists[i] == min(dists) else 0.0,
        1.0 if ords[i] == max(ords) else 0.0,
        1.0 if ords[i] == min(ords) else 0.0,
    ]


NODE_FAMS = ["SET_UNIT", "ADD_TARGET", "DROP_TARGET"]


def node_candidates(r):
    if r["op"] == "DROP_TARGET":
        return list(r["citmap"][r["cit"]]["targets"])
    return [n["node"] for n in r["sections"]]


def node_family_feats(r):
    ctx = _ctx(r)
    om = ctx[0]
    fam = r["op"]
    fi = NODE_FAMS.index(fam)
    cands = node_candidates(r)
    npos = len(pos_block(r, cands[0], ctx))
    out = []
    for i, nd in enumerate(cands):
        lb = local_block(r, nd, ctx) + intrinsic_block(r, nd)
        pb = pos_block(r, nd, ctx)
        de = drop_extra(r, cands, i, om) if fam == "DROP_TARGET" else [0.0] * 7
        blocks = [[0.0] * npos, [0.0] * npos, [0.0] * npos]
        blocks[fi] = pb
        out.append(lb + blocks[0] + blocks[1] + blocks[2] + de)
    return out


def range_candidates(r):
    secs = r["sections"]
    cands = []
    for i in range(len(secs)):
        for j in range(i, len(secs)):
            cands.append((secs[i]["node"], secs[j]["node"]))
    return cands


def range_feats(r):
    ctx = _ctx(r)
    om, c, _, _, _, _ = ctx
    nsec = max(1, len(r["sections"]))
    ao = om.get(r["anchor"], om[c["source"]])
    so2 = om[c["source"]]
    # Pool the two endpoints' salience: a SET_RANGE repair replaces the whole
    # target list, so the endpoints' current-target relation features (pos_block
    # indices 30-36) are dropped as noise; direction is carried by the pair
    # anchor-offset features below.
    node_block = {}
    for n in r["sections"]:
        pb = pos_block(r, n["node"], ctx)
        pb = [v for j, v in enumerate(pb) if not 30 <= j <= 36]
        node_block[n["node"]] = local_block(r, n["node"], ctx) + pb
    out = []
    for s, e in range_candidates(r):
        so_, eo_ = om[s], om[e]
        w = eo_ - so_
        pf = [1.0 if w == k else 0.0 for k in range(12)]
        pf.append(1.0 if w >= 12 else 0.0)
        pf.append(w / nsec)
        pf.append(1.0 if so_ <= ao <= eo_ else 0.0)
        d_sa, d_ea = so_ - ao, eo_ - ao
        pf += [1.0 if d_sa == k else 0.0 for k in range(-7, 8)]
        pf += [1.0 if d_ea == k else 0.0 for k in range(-7, 8)]
        pf.append(d_sa / nsec)
        pf.append(d_ea / nsec)
        pf.append(1.0 if so_ <= so2 <= eo_ else 0.0)
        endpoints = [a + b for a, b in zip(node_block[s], node_block[e])]
        out.append(endpoints + pf)
    return out


def fit_clogit(Xs, ys, l2, iters):
    """Fit grouped softmax likelihood with deterministic L-BFGS."""
    lengths = np.array([len(x) for x in Xs], dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    flat = np.concatenate(Xs, axis=0).astype(np.float64, copy=False)
    labels = starts[:-1] + np.asarray(ys, dtype=np.int64)
    ngroups = len(Xs)

    def objective(w):
        scores = flat @ w
        maxima = np.maximum.reduceat(scores, starts[:-1])
        exp_scores = np.exp(scores - np.repeat(maxima, lengths))
        totals = np.add.reduceat(exp_scores, starts[:-1])
        probs = exp_scores / np.repeat(totals, lengths)
        loss = (
            np.log(totals) + maxima - scores[labels]
        ).mean() + 0.5 * l2 * (w @ w)
        probs[labels] -= 1.0
        gradient = flat.T @ probs / ngroups + l2 * w
        return loss, gradient

    with np.errstate(all="ignore"):
        result = minimize(
            objective,
            np.zeros(flat.shape[1]),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-20.0, 20.0)] * flat.shape[1],
            options={
                "maxiter": iters,
                "ftol": 1e-9,
                "gtol": 1e-6,
                "maxcor": 10,
            },
        )
    if not np.isfinite(result.x).all():
        raise RuntimeError(f"conditional-logit optimization failed: {result.message}")
    return result.x


def op_key(op):
    if op["op"] == "SET_RANGE":
        return (op["start"], op["end"])
    return op.get("target") or op.get("scope")


def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    submission_out.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)

    # Fit every vocabulary, classifier, ranker, and statistic on train only.
    train = pd.read_csv(public_dir / "train.csv")
    op_labels = train["target_program"].map(
        lambda value: json.loads(value)["ops"][0]["op"]
    )
    op_model = make_pipeline(
        CountVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-z][a-z-]+\b",
        ),
        LogisticRegression(C=10.0, max_iter=500, random_state=SEED),
    )
    op_model.fit(train["amendment_note"], op_labels)

    tr = parse_rows(train)
    fit_feature_schema(tr)

    node_tr = [r for r in tr if r["op"] in NODE_FAMS]
    Xn, Yn = [], []
    for r in node_tr:
        cands = node_candidates(r)
        key = op_key(r["tp"]["ops"][0])
        Xn.append(np.asarray(node_family_feats(r), dtype=np.float32))
        Yn.append(cands.index(key))
    w_node = fit_clogit(Xn, Yn, l2=3e-4, iters=200)

    range_tr = [r for r in tr if r["op"] == "SET_RANGE"]
    Xr, Yr = [], []
    for r in range_tr:
        cands = range_candidates(r)
        target = r["tp"]["ops"][0]
        key = (target["start"], target["end"])
        Xr.append(np.asarray(range_feats(r), dtype=np.float32))
        Yr.append(cands.index(key))
    w_range = fit_clogit(Xr, Yr, l2=1e-2, iters=200)

    # Test is first touched here: transform and predict only.
    test = pd.read_csv(public_dir / "test.csv")
    op_predictions = op_model.predict(test["amendment_note"])
    te = parse_rows(test, op_predictions=op_predictions)

    predictions = []
    for r in te:
        if r["cit"] not in r["citmap"] or r["anchor"] not in r["nodemap"]:
            raise ValueError(f"unparseable amendment note for row {r['id']}")

        if r["op"] == "SET_SCOPE":
            op = {
                "op": "SET_SCOPE",
                "citation": r["cit"],
                "scope": r["scope"]["node"],
            }
        elif r["op"] in NODE_FAMS:
            cands = node_candidates(r)
            X = np.asarray(node_family_feats(r), dtype=np.float32)
            with np.errstate(all="ignore"):
                scores = X @ w_node
            if not np.isfinite(scores).all():
                raise RuntimeError("node ranker produced non-finite scores")
            choice = int(np.argmax(scores))
            op = {"op": r["op"], "citation": r["cit"], "target": cands[choice]}
        elif r["op"] == "SET_RANGE":
            cands = range_candidates(r)
            X = np.asarray(range_feats(r), dtype=np.float32)
            with np.errstate(all="ignore"):
                scores = (X @ w_range) / 1.3
            if not np.isfinite(scores).all():
                raise RuntimeError("range ranker produced non-finite scores")
            scores -= scores.max()
            probs = np.exp(scores)
            probs /= probs.sum()
            p_start = collections.Counter()
            p_end = collections.Counter()
            for (start, end), probability in zip(cands, probs):
                p_start[start] += probability
                p_end[end] += probability
            # The leaderboard also weights the lowest quartile.  Relative to
            # pure expected-row-score decoding, slightly more endpoint-marginal
            # weight protects uncertain rows from missing both endpoints.
            expected_scores = [
                0.60 * probability
                + (0.40 / 3.0) * (p_start[start] + p_end[end])
                for (start, end), probability in zip(cands, probs)
            ]
            start, end = cands[int(np.argmax(expected_scores))]
            op = {
                "op": "SET_RANGE",
                "citation": r["cit"],
                "start": start,
                "end": end,
            }
        else:
            raise ValueError(f"unknown predicted operation family {r['op']}")

        predictions.append(json.dumps({"ops": [op]}, separators=(",", ":")))

    submission = pd.DataFrame(
        {"id": [r["id"] for r in te], "predicted_program": predictions}
    )
    submission.to_csv(submission_out, index=False)
    print(f"wrote {len(submission)} rows to {submission_out}")


if __name__ == "__main__":
    main()
