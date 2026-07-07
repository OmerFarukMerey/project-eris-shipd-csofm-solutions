"""
GUI widget grounding and interaction prediction.

For every screenshot + instruction pair in dataset/public/test.csv, predicts
the target widget box, action type, element role, interaction blocker, the
canonical repair-step sequence, and the resulting interaction status. Writes
working/submission.csv.

Facts mined from the public train split (verified against the 1900 labeled
rows, not assumed) drive the whole pipeline:
  - repair_sequence and interaction_status are exact deterministic functions
    of (blocker_type, action_type) in every one of the 1900 train rows -- no
    model is fit for those two columns, only a lookup applied after
    blocker_type/action_type are predicted.
  - action_type (~98%) and element_role (~99%) are recoverable directly from
    user_instruction text: three fixed wrapper phrases are 100% deterministic
    and cover ~70% of rows; the remaining "raw" instructions are handled by a
    small TF-IDF + logistic-regression text classifier trained on the public
    train split (it picks up recurring page/label vocabulary such as
    "SELECT ALL" -> select or a recurring blog title -> type that a fixed
    keyword regex cannot).
  - ui_context_note and prior_action_trace show no measurable association
    with any target column (checked via chi-square/Cramer's V on the train
    split) and are treated as decoys, not features.
  - target_box == [0,0,0,0] if and only if blocker_type == 'wrong_page',
    exactly, in both directions across all 1900 train rows.
  - blocker_type and target_box are the two genuinely vision-dependent
    targets. They are solved with OCR-based text grounding (shelling out to
    the Tesseract binary already installed on this machine -- no pip OCR
    dependency needed) plus classical pixel features (a blue "Dismiss"
    button detector, and local saturation/edge/brightness statistics),
    feeding a single sklearn RandomForestClassifier trained only on the
    public train split. A separate threshold on that model's P(wrong_page)
    -- tuned on out-of-fold predictions to directly maximize box score --
    decides when target_box is zeroed out, since target_box and blocker_type
    are graded as independent columns and do not have to agree.

No external labels, sample_id/row-order lookups, or matching against a
source dataset are used anywhere in this pipeline -- every prediction is
computed from that row's own screenshot and instruction text.
"""
import os
import re
import json
import tempfile
import subprocess
import unicodedata
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "public", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "dataset", "public", "test.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "public")
OUT_DIR = os.path.join(BASE_DIR, "working")
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")
OCR_CACHE_PATH = os.path.join(OUT_DIR, "ocr_cache.json")

os.makedirs(OUT_DIR, exist_ok=True)

BLOCKER_CLASSES = ["none", "ambiguous", "covered_modal", "disabled", "wrong_page"]
ACTION_CLASSES = ["click", "type", "select"]
ROLE_CLASSES = ["text_field", "tab", "button", "link", "widget"]
STATUS_CLASSES = ["ready", "blocked", "needs_navigation", "ambiguous"]

PREFIX_MAP = {
    "none": "",
    "ambiguous": "inspect_duplicates",
    "covered_modal": "dismiss_modal",
    "disabled": "enable_control",
    "wrong_page": "navigate_back",
}
STATUS_MAP = {
    "none": "ready",
    "ambiguous": "ambiguous",
    "covered_modal": "blocked",
    "disabled": "blocked",
    "wrong_page": "needs_navigation",
}

try:
    subprocess.run(["tesseract", "--version"], capture_output=True, timeout=5, check=True)
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


# ==========================================================================
# Instruction parsing: action_type / element_role / target label
# ==========================================================================
WRAP_TYPE = "type into the control described by this request:"
WRAP_CLICK = "click the widget described by this request:"
WRAP_SELECT = "select the option or control described by this request:"

TAB_OVERRIDE_RE = re.compile(r"\btableau|\btableur")
ROLE_KEYWORDS = [
    ("widget", re.compile(r"\bcheckbox|\btext\s*area\b|\blist\b|\bdropdown\b|\bslider\b|\bswitch\b")),
    ("text_field", re.compile(r"\btext\s*field\b")),
    ("tab", re.compile(r"\btab\b")),
    ("link", re.compile(r"\blink\b")),
    ("button", re.compile(r"\bbutton|radio\s*button\b")),
]
ROLE_HINT_RE = re.compile(r"\((link|button|checkbox|radio button|text field|list|tab|widget)\)\s*$", re.I)

VERB_STRIP_RE = re.compile(
    r"^(click|type into|type|paste in|paste|select|choose)\s+(on\s+|left\s+|right\s+)?", re.I
)
ROLE_WORD_STRIP_RE = re.compile(
    r"^(text\s*field|text\s*area|checkbox|radio\s*button|button|tab|link|list|dropdown|slider|switch|widget)"
    r"\s*[:]?\s*(corresponding to|for)?\s*",
    re.I,
)


def has_wrapper(instr):
    sl = instr.strip().lower()
    return sl.startswith(WRAP_TYPE) or sl.startswith(WRAP_CLICK) or sl.startswith(WRAP_SELECT)


def regex_action_type(instr):
    """Deterministic wrapper-phrase action_type. Returns None for raw (unwrapped) instructions."""
    sl = instr.strip().lower()
    if sl.startswith(WRAP_TYPE):
        return "type"
    if sl.startswith(WRAP_CLICK):
        return "click"
    if sl.startswith(WRAP_SELECT):
        return "select"
    return None


def parse_element_role(instr):
    sl = instr.lower()
    if TAB_OVERRIDE_RE.search(sl):
        return "tab"
    for role, pat in ROLE_KEYWORDS:
        if pat.search(sl):
            return role
    return "widget"


def extract_label(instr):
    """Strip wrapper/verb/role-word prefixes and whitelisted trailing role hints
    to isolate the widget's referring text, used to ground it in the screenshot."""
    s = instr.strip()
    sl = s.lower()
    for wrap in (WRAP_TYPE, WRAP_CLICK, WRAP_SELECT):
        if sl.startswith(wrap):
            s = s[len(wrap):].strip()
            break
    m = re.search(r"\bthe following text\b", s, re.I)
    if m:
        s = s[: m.start()].strip()
    s = VERB_STRIP_RE.sub("", s, count=1)
    s = ROLE_WORD_STRIP_RE.sub("", s, count=1)
    s = ROLE_HINT_RE.sub("", s).strip()
    s = s.strip(" :-")
    return s if s else instr.strip()


def fit_action_type_model(train_df):
    """TF-IDF + logistic regression fallback for instructions with no deterministic wrapper."""
    raw_mask = ~train_df["user_instruction"].apply(has_wrapper)
    X = train_df.loc[raw_mask, "user_instruction"].values
    y = train_df.loc[raw_mask, "action_type"].values
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    Xt = vec.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=5)
    clf.fit(Xt, y)
    return vec, clf


def predict_action_types(instructions, vec, clf):
    out = np.empty(len(instructions), dtype=object)
    raw_idx, raw_texts = [], []
    for i, instr in enumerate(instructions):
        wrapped = regex_action_type(instr)
        if wrapped is not None:
            out[i] = wrapped
        else:
            raw_idx.append(i)
            raw_texts.append(instr)
    if raw_texts:
        preds = clf.predict(vec.transform(raw_texts))
        for i, p in zip(raw_idx, preds):
            out[i] = p
    return out


# ==========================================================================
# OCR layer (shells out to the Tesseract CLI binary, no pip dependency)
# ==========================================================================
def preprocess_for_ocr(path):
    im = Image.open(path).convert("L")
    im = ImageOps.autocontrast(im, cutoff=1)
    im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    return im


def ocr_image(path):
    if not OCR_AVAILABLE:
        return []
    tmp_path = None
    try:
        im = preprocess_for_ocr(path)
        W, H = im.size
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        im.save(tmp_path)
        result = subprocess.run(
            ["tesseract", tmp_path, "stdout", "tsv"],
            capture_output=True, text=True, timeout=30,
        )
        words = []
        for line in result.stdout.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 12 or parts[0] != "5":
                continue
            text = parts[11].strip()
            if not text:
                continue
            words.append({
                "block": int(parts[2]), "par": int(parts[3]), "line": int(parts[4]),
                "word": int(parts[5]),
                "x": float(parts[6]) / W, "y": float(parts[7]) / H,
                "w": float(parts[8]) / W, "h": float(parts[9]) / H,
                "conf": float(parts[10]), "text": text,
            })
        return words
    except Exception:
        return []
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_ocr_cache(df):
    cache = {}
    if os.path.exists(OCR_CACHE_PATH):
        try:
            with open(OCR_CACHE_PATH) as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    missing = [(sid, os.path.join(IMAGES_DIR, ip))
               for sid, ip in zip(df["sample_id"], df["image_path"]) if sid not in cache]
    if missing and OCR_AVAILABLE:
        print(f"  running OCR on {len(missing)} images...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(lambda t: (t[0], ocr_image(t[1])), missing))
        for sid, words in results:
            cache[sid] = words
        with open(OCR_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    elif missing:
        for sid, _ in missing:
            cache[sid] = []
    return cache


# ==========================================================================
# Fuzzy phrase matching / grounding
# ==========================================================================
def normalize_text(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def group_lines(words):
    lines = defaultdict(list)
    for w in words:
        lines[(w["block"], w["par"], w["line"])].append(w)
    for key in lines:
        lines[key].sort(key=lambda w: w["word"])
    return lines


def candidate_phrases(words, max_win=6):
    lines = group_lines(words)
    sorted_keys = sorted(lines.keys())
    candidates = []
    for key in sorted_keys:
        ws = lines[key]
        n = len(ws)
        for i in range(n):
            for j in range(i + 1, min(n, i + max_win) + 1):
                window = ws[i:j]
                _add_candidate(candidates, window)
    by_block_par = defaultdict(list)
    for key in sorted_keys:
        by_block_par[(key[0], key[1])].append(key)
    for keys in by_block_par.values():
        keys = sorted(keys)
        for span in (2, 3):
            for i in range(len(keys) - span + 1):
                ws_all = [w for k in keys[i:i + span] for w in lines[k]]
                if ws_all:
                    _add_candidate(candidates, ws_all)
    return candidates


def _add_candidate(candidates, window):
    text = " ".join(w["text"] for w in window)
    x0 = min(w["x"] for w in window)
    y0 = min(w["y"] for w in window)
    x1 = max(w["x"] + w["w"] for w in window)
    y1 = max(w["y"] + w["h"] for w in window)
    conf = float(np.mean([w["conf"] for w in window]))
    candidates.append({"text": text, "bbox": (x0, y0, x1 - x0, y1 - y0), "conf": conf, "top": y0})


def score_candidates(label, candidates):
    norm_label = normalize_text(label)
    scored = []
    if not norm_label:
        return scored
    for c in candidates:
        norm_c = normalize_text(c["text"])
        if not norm_c:
            continue
        score = SequenceMatcher(None, norm_label, norm_c).ratio()
        scored.append((score, c))
    scored.sort(key=lambda t: -t[0])
    return scored


_MATCH_CACHE = {}


def get_match_data(sid, ocr_cache, labels_by_id):
    """candidate_phrases()+score_candidates() depend only on the image's OCR
    words and the row's parsed label, both fixed per sample_id -- memoize so
    the (expensive) fuzzy-matching pass runs once per image no matter how
    many times a row is touched across CV folds."""
    cached = _MATCH_CACHE.get(sid)
    if cached is not None:
        return cached
    words = ocr_cache.get(sid, [])
    label = labels_by_id[sid]
    cands = candidate_phrases(words) if words else []
    scored = score_candidates(label, cands) if cands else []
    result = (cands, scored)
    _MATCH_CACHE[sid] = result
    return result


_PAGE_TOKENS_CACHE = {}


def get_page_tokens(sid, ocr_cache):
    """Bag-of-words vocabulary of everything OCR read on the page, memoized
    per sample_id. A phrase-level fuzzy match can miss a real label that OCR
    split across non-adjacent lines; word overlap catches that case and
    helps separate a genuinely absent label (wrong_page) from one that is
    merely fragmented on the page."""
    cached = _PAGE_TOKENS_CACHE.get(sid)
    if cached is not None:
        return cached
    tokens = set()
    for w in ocr_cache.get(sid, []):
        tokens.update(normalize_text(w["text"]).split())
    _PAGE_TOKENS_CACHE[sid] = tokens
    return tokens


def word_overlap_ratio(label, page_tokens):
    label_tokens = set(normalize_text(label).split())
    if not label_tokens or not page_tokens:
        return 0.0
    hits = sum(1 for t in label_tokens if t in page_tokens)
    return hits / len(label_tokens)


def duplicate_cluster_count(scored, score_floor=0.72, dist_thresh=0.06):
    centers = []
    for s, c in scored:
        if s < score_floor:
            break
        x, y, w, h = c["bbox"]
        cx, cy = x + w / 2, y + h / 2
        if all(abs(ex - cx) >= dist_thresh or abs(ey - cy) >= dist_thresh for ex, ey in centers):
            centers.append((cx, cy))
    return len(centers)


# ==========================================================================
# Box grounding: role-specific offset/scale correction, fit from train
# ==========================================================================
def fit_role_priors(train_df):
    """Unconditional per (true) element_role box prior -- position + size --
    from every non-wrong_page train row. Used as the fallback box when no
    OCR match is available or usable."""
    priors = {}
    for role in ROLE_CLASSES:
        sub = train_df[(train_df["element_role"] == role) & (train_df["blocker_type"] != "wrong_page")]
        if len(sub):
            arr = np.array([json.loads(b) for b in sub["target_box"]])
            cx = arr[:, 0] + arr[:, 2] / 2
            cy = arr[:, 1] + arr[:, 3] / 2
            priors[role] = {
                "cx": float(np.median(cx)), "cy": float(np.median(cy)),
                "w": float(np.median(arr[:, 2])), "h": float(np.median(arr[:, 3])),
            }
        else:
            priors[role] = {"cx": 0.5, "cy": 0.5, "w": 0.08, "h": 0.035}
    return priors


def fit_box_offset_params(train_df, ocr_cache, labels_by_id):
    """Per-role offset (matched-phrase center -> true box center) and
    scale (matched-phrase size -> true box size), measured only on rows
    where OCR found a confident match. Also fits the tab top-chrome-band
    special case since tab targets there sit at a near-constant row."""
    per_role = defaultdict(list)
    tab_band_rows = []
    for _, row in train_df.iterrows():
        if row["blocker_type"] == "wrong_page":
            continue
        _, scored = get_match_data(row["sample_id"], ocr_cache, labels_by_id)
        if not scored or scored[0][0] < 0.6:
            continue
        role = row["element_role"]
        score, c = scored[0]
        tx, ty, tw, th = json.loads(row["target_box"])
        tcx, tcy = tx + tw / 2, ty + th / 2
        ox, oy, ow, oh = c["bbox"]
        ocx, ocy = ox + ow / 2, oy + oh / 2
        per_role[role].append({
            "dcx": tcx - ocx, "dcy": tcy - oy,
            "sw": tw / max(ow, 1e-4), "sh": th / max(oh, 1e-4),
        })
        if role == "tab" and oy < 0.05:
            tab_band_rows.append((ty, th))

    params = {}
    for role in ROLE_CLASSES:
        items = per_role.get(role, [])
        if len(items) >= 5:
            params[role] = {
                "dcx": float(np.median([it["dcx"] for it in items])),
                "dcy": float(np.median([it["dcy"] for it in items])),
                "sw": float(np.clip(np.median([it["sw"] for it in items]), 0.2, 5.0)),
                "sh": float(np.clip(np.median([it["sh"] for it in items]), 0.2, 5.0)),
            }
        else:
            params[role] = {"dcx": 0.0, "dcy": 0.0, "sw": 1.0, "sh": 1.0}

    if len(tab_band_rows) >= 5:
        ys = [t[0] for t in tab_band_rows]
        hs = [t[1] for t in tab_band_rows]
        tab_band = {"y": float(np.median(ys)), "h": float(np.median(hs)), "band_thresh": 0.05}
    else:
        tab_band = None
    return params, tab_band


def ground_box(role, scored, offset_params, priors, tab_band, match_thresh=0.45):
    if scored and scored[0][0] >= match_thresh:
        score, c = scored[0]
        ox, oy, ow, oh = c["bbox"]
        ocx = ox + ow / 2
        pr = priors[role]
        if role == "tab" and tab_band is not None and oy < tab_band["band_thresh"]:
            h = tab_band["h"]
            y = tab_band["y"]
            w = float(np.clip(ow * 1.4, 0.05, 0.22))
            x = ocx - w / 2
        else:
            p = offset_params[role]
            cx = ocx + p["dcx"]
            cy = oy + p["dcy"]
            w = float(np.clip(ow * p["sw"], 0.3 * pr["w"], 3.0 * pr["w"]))
            h = float(np.clip(oh * p["sh"], 0.3 * pr["h"], 3.0 * pr["h"]))
            x = cx - w / 2
            y = cy - h / 2
    else:
        pr = priors[role]
        w, h = pr["w"], pr["h"]
        x, y = pr["cx"] - w / 2, pr["cy"] - h / 2

    x = float(np.clip(x, 0.0, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    w = float(np.clip(w, 0.0, 1.0 - x))
    h = float(np.clip(h, 0.0, 1.0 - y))
    return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]


# ==========================================================================
# Pixel features: blue "Dismiss" button detector + local texture stats
# ==========================================================================
def blue_button_score(img_arr):
    r = img_arr[:, :, 0].astype(np.int16)
    g = img_arr[:, :, 1].astype(np.int16)
    b = img_arr[:, :, 2].astype(np.int16)
    mask = (b > 120) & (b - r > 25) & (b - g > 15) & (r < 160)
    if mask.sum() < 20:
        return 0.0, 0
    lbl, n = ndimage.label(mask)
    H, W = mask.shape
    best_fill, qualifying = 0.0, 0
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(xs) < 20:
            continue
        h_box, w_box = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        fill = len(xs) / (h_box * w_box)
        aspect = w_box / max(h_box, 1)
        if fill > 0.55 and 1.2 < aspect < 10 and (w_box / W) < 0.25 and (h_box / H) < 0.08:
            qualifying += 1
            best_fill = max(best_fill, fill)
    return best_fill, qualifying


_PIXEL_CACHE = {}


def get_pixel_globals(sid, img_arr):
    """blue_button_score() and the whole-image texture stats depend only on
    the image itself, not on the predicted box -- memoize per sample_id."""
    cached = _PIXEL_CACHE.get(sid)
    if cached is not None:
        return cached
    fill, qual = blue_button_score(img_arr)
    global_tex = texture_features(img_arr, [0.0, 0.0, 1.0, 1.0], pad=0.0)
    result = (fill, qual, global_tex)
    _PIXEL_CACHE[sid] = result
    return result


def texture_features(img_arr, box_norm, pad=0.15):
    H, W = img_arr.shape[:2]
    x, y, w, h = box_norm
    cx, cy = x + w / 2, y + h / 2
    pw, ph = max(w * (1 + 2 * pad), 0.02), max(h * (1 + 2 * pad), 0.02)
    x0, x1 = max(0, int((cx - pw / 2) * W)), min(W, int((cx + pw / 2) * W))
    y0, y1 = max(0, int((cy - ph / 2) * H)), min(H, int((cy + ph / 2) * H))
    if x1 <= x0 or y1 <= y0:
        return dict(sat_mean=0.0, edge_mean=0.0, bright_mean=0.0, red_frac=0.0)
    crop = img_arr[y0:y1, x0:x1].astype(np.float32)
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    maxc, minc = crop.max(axis=-1), crop.min(axis=-1)
    sat = np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1e-4), 0)
    bright = crop.mean(axis=-1)
    gx = np.abs(np.diff(bright, axis=1)) if bright.shape[1] > 1 else np.zeros((1, 1))
    gy = np.abs(np.diff(bright, axis=0)) if bright.shape[0] > 1 else np.zeros((1, 1))
    edge = 0.5 * gx.mean() + 0.5 * gy.mean()
    red_frac = float(((r > 150) & (r - g > 40) & (r - b > 40)).mean())
    return dict(sat_mean=float(sat.mean()), edge_mean=float(edge),
                bright_mean=float(bright.mean()), red_frac=red_frac)


# ==========================================================================
# Feature assembly for blocker_type models
# ==========================================================================
def build_row_features(row, ocr_cache, labels_by_id, offset_params, priors, tab_band, img_cache):
    sid = row["sample_id"]
    role = row["_pred_role"]
    cands, scored = get_match_data(sid, ocr_cache, labels_by_id)

    best_score = scored[0][0] if scored else 0.0
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    all_scores = [s for s, _ in scored[:20]]
    rel_score = best_score - (float(np.median(all_scores)) if all_scores else 0.0)
    n_high = sum(1 for s in all_scores if s >= 0.72)
    dup_count = duplicate_cluster_count(scored)
    best_conf = scored[0][1]["conf"] if scored else 0.0
    word_overlap = word_overlap_ratio(labels_by_id[sid], get_page_tokens(sid, ocr_cache))

    box_pred = ground_box(role, scored, offset_params, priors, tab_band)

    img_path = os.path.join(IMAGES_DIR, row["image_path"])
    img_arr = img_cache.get(sid)
    if img_arr is None:
        try:
            img_arr = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            img_arr = np.zeros((10, 10, 3), dtype=np.uint8)
        img_cache[sid] = img_arr

    fill, qual, global_tex = get_pixel_globals(sid, img_arr)
    local_tex = texture_features(img_arr, box_pred, pad=0.15)

    feat = {
        "best_score": best_score, "second_score": second_score, "rel_score": rel_score,
        "n_high": n_high, "dup_count": dup_count, "best_conf": best_conf,
        "n_candidates": len(cands), "word_overlap": word_overlap,
        "modal_fill": fill, "modal_qual": qual,
        "local_sat": local_tex["sat_mean"], "local_edge": local_tex["edge_mean"],
        "local_bright": local_tex["bright_mean"], "local_red": local_tex["red_frac"],
        "global_sat": global_tex["sat_mean"], "global_edge": global_tex["edge_mean"],
        "edge_ratio": local_tex["edge_mean"] / max(global_tex["edge_mean"], 1e-3),
    }
    return feat, box_pred


def build_feature_matrix(df, ocr_cache, labels_by_id, offset_params, priors, tab_band):
    img_cache = {}
    feats, boxes = [], []
    for _, row in df.iterrows():
        f, b = build_row_features(row, ocr_cache, labels_by_id, offset_params, priors, tab_band, img_cache)
        feats.append(f)
        boxes.append(b)
    X = pd.DataFrame(feats)
    for role in ROLE_CLASSES:
        X[f"role_{role}"] = (df["_pred_role"].values == role).astype(int)
    for act in ACTION_CLASSES:
        X[f"act_{act}"] = (df["_pred_action"].values == act).astype(int)
    return X, boxes


# ==========================================================================
# Deterministic post-processing
# ==========================================================================
def repair_sequence_from(blocker, action):
    prefix = PREFIX_MAP[blocker]
    target = f"{action}_target"
    return f"{prefix}>{target}" if prefix else target


def interaction_status_from(blocker):
    return STATUS_MAP[blocker]


# ==========================================================================
# Local scoring: exact competition formula
# ==========================================================================
def iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def box_score(true_box, pred_box):
    pred_box = [float(np.clip(v, 0.0, 1.0)) for v in pred_box]
    is_zero_true = all(v == 0 for v in true_box)
    if is_zero_true:
        is_zero_pred = all(v == 0 for v in pred_box)
        return 1.0 if is_zero_pred else box_score_normal([0, 0, 0, 0], pred_box)
    return box_score_normal(true_box, pred_box)


def box_score_normal(true_box, pred_box):
    i = iou(true_box, pred_box)
    tcx, tcy = true_box[0] + true_box[2] / 2, true_box[1] + true_box[3] / 2
    pcx, pcy = pred_box[0] + pred_box[2] / 2, pred_box[1] + pred_box[3] / 2
    center = max(0.0, 1 - abs(tcx - pcx) - abs(tcy - pcy))
    return 0.75 * i + 0.25 * center


def token_levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]


def repair_sequence_score(true_seq, pred_seq):
    a, b = true_seq.split(">"), pred_seq.split(">")
    dist = token_levenshtein(a, b)
    denom = max(len(a), len(b), 1)
    sim = 1 - dist / denom
    exact = 1.0 if true_seq == pred_seq else 0.0
    return 0.30 * sim + 0.70 * exact


def inv_freq_weights(y_true, classes):
    counts = pd.Series(y_true).value_counts()
    weights = {c: 1.0 / np.sqrt(counts.get(c, 0) + 1) for c in classes}
    median_w = float(np.median(list(weights.values())))
    cap = 5 * median_w
    weights = {c: min(w, cap) for c, w in weights.items()}
    total = sum(weights.values())
    return {c: w / total for c, w in weights.items()}


def weighted_macro_f1(y_true, y_pred, classes):
    weights = inv_freq_weights(y_true, classes)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    score = 0.0
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        score += weights[c] * f1
    return score


# ==========================================================================
# blocker_type: single joint 5-way classifier
# ==========================================================================
# A rigid wrong_page-vs-rest first stage was tried and measured to fail: OCR
# match quality (best_score/second_score) is suppressed both when the target
# is genuinely absent (wrong_page) and when it is merely occluded by the
# modal panel (covered_modal), so the two overlap heavily on those features
# alone (means 0.454 vs 0.487 on the public train split) and an isolated
# binary stage over-fires wrong_page on covered_modal/disabled/none rows.
# A single joint classifier lets features that DO separate those cases
# (the blue-button detector for covered_modal, edge/texture stats for
# disabled) correct for a weak match score instead of being locked out by
# an earlier hard gate.
FEATURE_COLS = [
    "best_score", "second_score", "rel_score", "n_high", "dup_count", "best_conf",
    "n_candidates", "word_overlap", "modal_fill", "modal_qual", "local_sat", "local_edge",
    "local_bright", "local_red", "global_sat", "global_edge", "edge_ratio",
] + [f"role_{r}" for r in ROLE_CLASSES] + [f"act_{a}" for a in ACTION_CLASSES]


def train_blocker_model(X, y):
    sw = y.map(inv_freq_weights(y, BLOCKER_CLASSES)).values
    clf = RandomForestClassifier(
        n_estimators=600, max_depth=10, min_samples_leaf=4, random_state=SEED
    )
    clf.fit(X[FEATURE_COLS].values, y.values, sample_weight=sw)
    return clf


def predict_blocker_model(clf, X):
    return clf.predict(X[FEATURE_COLS].values)


def wrong_page_proba(clf, X):
    proba = clf.predict_proba(X[FEATURE_COLS].values)
    idx = list(clf.classes_).index("wrong_page")
    return proba[:, idx]


def tune_zero_box_threshold(wp_proba, true_boxes, grounded_boxes, thresholds=None):
    """The blocker_type LABEL (argmax over 5 classes) and the decision to
    zero out target_box are scored as two independent columns, so they do
    not have to agree: pick whichever P(wrong_page) cutoff maximizes the
    OOF box score directly, instead of always zeroing exactly when the
    argmax label happens to say wrong_page."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.85, 0.05)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        scores = []
        for p, tb, gb in zip(wp_proba, true_boxes, grounded_boxes):
            pred = [0.0, 0.0, 0.0, 0.0] if p >= t else gb
            scores.append(box_score(tb, pred))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_t = mean_score, t
    return best_t, best_score


# ==========================================================================
# Main
# ==========================================================================
def main():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"  train={len(train_df)} test={len(test_df)} OCR_AVAILABLE={OCR_AVAILABLE}")

    train_df["_pred_role"] = train_df["user_instruction"].apply(parse_element_role)
    test_df["_pred_role"] = test_df["user_instruction"].apply(parse_element_role)
    labels_by_id = {}
    for df_ in (train_df, test_df):
        for sid, instr in zip(df_["sample_id"], df_["user_instruction"]):
            labels_by_id[sid] = extract_label(instr)

    print("Building OCR cache (this is cached to working/ocr_cache.json after the first run)...")
    all_df = pd.concat([train_df[["sample_id", "image_path"]], test_df[["sample_id", "image_path"]]],
                        ignore_index=True)
    ocr_cache = build_ocr_cache(all_df)

    print("Fitting box grounding priors from train...")
    priors = fit_role_priors(train_df)
    offset_params, tab_band = fit_box_offset_params(train_df, ocr_cache, labels_by_id)

    # ---------------- 5-fold CV over train for local validation ----------
    print("Running 5-fold CV for local validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_action = np.empty(len(train_df), dtype=object)
    oof_role = train_df["_pred_role"].values.copy()
    oof_blocker = np.empty(len(train_df), dtype=object)
    oof_grounded_box = [None] * len(train_df)
    oof_wp_proba = np.zeros(len(train_df))

    y_blocker_full = train_df["blocker_type"]
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y_blocker_full)):
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        vec, clf = fit_action_type_model(tr_df)
        va_action = predict_action_types(va_df["user_instruction"].values, vec, clf)
        for j, i in enumerate(va_idx):
            oof_action[i] = va_action[j]
        va_df["_pred_action"] = va_action
        tr_df["_pred_action"] = predict_action_types(tr_df["user_instruction"].values, vec, clf)

        fold_priors = fit_role_priors(tr_df)
        fold_offset, fold_tab_band = fit_box_offset_params(tr_df, ocr_cache, labels_by_id)

        Xtr, _ = build_feature_matrix(tr_df, ocr_cache, labels_by_id, fold_offset, fold_priors, fold_tab_band)
        Xva, boxes_va = build_feature_matrix(va_df, ocr_cache, labels_by_id, fold_offset, fold_priors, fold_tab_band)

        blocker_clf = train_blocker_model(Xtr, tr_df["blocker_type"])
        va_blocker = predict_blocker_model(blocker_clf, Xva)
        va_wp_proba = wrong_page_proba(blocker_clf, Xva)
        for j, i in enumerate(va_idx):
            oof_blocker[i] = va_blocker[j]
            oof_grounded_box[i] = boxes_va[j]
            oof_wp_proba[i] = va_wp_proba[j]
        print(f"  fold {fold + 1}/5 done")

    # ---------------- tune the box-zeroing threshold on OOF data -------
    # blocker_type (label) and target_box are scored as independent columns,
    # so the box doesn't have to be zeroed only when the argmax label says
    # wrong_page -- pick whichever P(wrong_page) cutoff maximizes OOF box
    # score directly (see tune_zero_box_threshold's docstring).
    true_boxes = [json.loads(b) for b in train_df["target_box"]]
    zero_thresh, tuned_box_component = tune_zero_box_threshold(
        oof_wp_proba, true_boxes, oof_grounded_box
    )
    print(f"  tuned zero-box threshold on P(wrong_page): {zero_thresh:.2f} "
          f"(OOF box score at this threshold: {tuned_box_component:.4f})")
    oof_box = [
        [0.0, 0.0, 0.0, 0.0] if p >= zero_thresh else gb
        for p, gb in zip(oof_wp_proba, oof_grounded_box)
    ]

    # ---------------- score OOF predictions --------------------------
    box_scores = [box_score(t, p) for t, p in zip(true_boxes, oof_box)]
    box_component = float(np.mean(box_scores))

    true_repair = [repair_sequence_from(b, a) for b, a in zip(train_df["blocker_type"], train_df["action_type"])]
    pred_repair = [repair_sequence_from(b, a) for b, a in zip(oof_blocker, oof_action)]
    repair_component = float(np.mean([repair_sequence_score(t, p) for t, p in zip(true_repair, pred_repair)]))

    action_component = weighted_macro_f1(train_df["action_type"].values, oof_action, ACTION_CLASSES)
    role_component = weighted_macro_f1(train_df["element_role"].values, oof_role, ROLE_CLASSES)
    blocker_component = weighted_macro_f1(train_df["blocker_type"].values, oof_blocker, BLOCKER_CLASSES)

    true_status = train_df["blocker_type"].map(interaction_status_from).values
    pred_status = pd.Series(oof_blocker).map(interaction_status_from).values
    status_component = weighted_macro_f1(true_status, pred_status, STATUS_CLASSES)

    overall = (0.32 * box_component + 0.20 * repair_component + 0.12 * action_component
               + 0.12 * role_component + 0.12 * blocker_component + 0.12 * status_component)

    print("\n===== Local 5-fold CV score (out-of-fold) =====")
    print(f"  target_box     (w=0.32): {box_component:.4f}")
    print(f"  repair_sequence(w=0.20): {repair_component:.4f}")
    print(f"  action_type    (w=0.12): {action_component:.4f}")
    print(f"  element_role   (w=0.12): {role_component:.4f}")
    print(f"  blocker_type   (w=0.12): {blocker_component:.4f}")
    print(f"  interaction_status(w=0.12): {status_component:.4f}")
    print(f"  OVERALL WEIGHTED SCORE       : {overall:.4f}")
    print(f"  (sample_submission.csv baseline: 0.116865)")
    print("blocker_type OOF confusion (rows=true, cols=pred):")
    print(pd.crosstab(train_df["blocker_type"], pd.Series(oof_blocker), rownames=["true"], colnames=["pred"]))

    # ---------------- refit on full train, predict test ----------------
    print("\nRefitting on full train and predicting test...")
    vec, clf = fit_action_type_model(train_df)
    train_df["_pred_action"] = predict_action_types(train_df["user_instruction"].values, vec, clf)
    test_df["_pred_action"] = predict_action_types(test_df["user_instruction"].values, vec, clf)

    Xtr_full, _ = build_feature_matrix(train_df, ocr_cache, labels_by_id, offset_params, priors, tab_band)
    Xte, boxes_te = build_feature_matrix(test_df, ocr_cache, labels_by_id, offset_params, priors, tab_band)

    blocker_clf = train_blocker_model(Xtr_full, train_df["blocker_type"])
    test_blocker = predict_blocker_model(blocker_clf, Xte)
    test_wp_proba = wrong_page_proba(blocker_clf, Xte)

    test_action = test_df["_pred_action"].values
    test_role = test_df["_pred_role"].values
    test_box = [
        [0.0, 0.0, 0.0, 0.0] if p >= zero_thresh else b
        for p, b in zip(test_wp_proba, boxes_te)
    ]
    test_repair = [repair_sequence_from(b, a) for b, a in zip(test_blocker, test_action)]
    test_status = [interaction_status_from(b) for b in test_blocker]

    print("Predicted blocker_type distribution (test):")
    print(pd.Series(test_blocker).value_counts())

    out = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "target_box": [json.dumps([round(v, 4) for v in b]) for b in test_box],
        "action_type": test_action,
        "element_role": test_role,
        "blocker_type": test_blocker,
        "repair_sequence": test_repair,
        "interaction_status": test_status,
    })
    out.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
