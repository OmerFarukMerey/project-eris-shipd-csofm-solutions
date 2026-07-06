# Code Behavior Fingerprint Recovery

## The problem

For each row, a candidate solution (`candidate_code`) was written against a `problem_statement` and run
through a hidden 10-test suite. The task is to predict, for every test row, the 10-bit `pass_mask` of which
of the 10 hidden tests passed — `fail_count` (the number of zeros) and `score_bucket` (`"s{#ones}"`) are
deterministic functions of that mask, so only the mask needs predicting.

### What drives the score (measured on the 1200 train rows, against the exact metric)

- The score is dominated by the *number* of passing tests, not which positions fail (true count + placement
  ≈ 0.81; a ±1 count error ≈ 0.66; the placement method itself only moves it ≈ 0.01).
- `fail_count` and `score_bucket` are deterministic functions of the mask, so the model only ever predicts
  the mask.

## The solution

`solution.py` is a single, self-contained script. Reads `dataset/public/train.csv` and
`dataset/public/test.csv`; writes `working/submission.csv` (columns: `id, answer_json`). Needs Python
stdlib + `numpy` + `pandas` + `scikit-learn`.

1. **Pass count via text k-NN.** An ensemble of three TF-IDF views of `problem_statement + candidate_code`
   (char_wb 3-6, char 4-7, word 1-2); their cosine similarities are averaged. For each test row, take its
   k=3 most similar train rows and use the **median** pass count of those neighbours. This adapts per row
   and lowers the count only where the neighbours are genuinely low. (The ensemble is lower-variance than
   any single vectorizer; neighbour-level placement and similarity-weighted counts were tried and both
   hurt, so the count stays a plain neighbour median.)

2. **Broken override via static analysis** (code-based, so it transfers regardless of the answer
   distribution): syntax errors, sandbox-fatal external dependencies (live network / `input()` / plotting
   GUI), and undefined-name (`NameError`) analysis force an all-fail mask. On a truly-all-fail row, all-zeros
   scores ≈1.0 vs ≈0.0 for all-ones — this is the single most reliable lever (~13% of solutions fail every
   test).

3. **Placement.** Given the count k, fail the `(10-k)` most fail-prone positions using a
   `P(fail | count=k)` table learned from train plus blueprint priors (positions whose test mentions an
   exception, or has an unusual assert count, are weighted as more fail-prone — this earns some
   failed-position F1).

4. **Non-broken count floor.** When the k-NN predicts a very low count for code that is *not* statically
   broken, that prediction is unreliable (on train such rows pass ~5-6 tests on average), so the count is
   floored so a near-all-fail mask is never emitted for working code.

5. **Sample-execution verdict.** Each problem contains a sample input/output. It's extracted, the candidate
   is run on the sample input in a sandboxed subprocess (timeout + fixed `PYTHONHASHSEED`; env-dependent /
   random / time code is skipped for determinism), and compared to the sample output:
   - `MATCH` (reproduced the sample) → lean the count up (≥8)
   - `NOMATCH` (ran but wrong output) → lean the count down (≤6)
   - `CRASH` (raised on the sample) → lean the count down (≤6)

   This is the only static/runtime signal that observes *real* behaviour. Best-effort and fully graceful: if
   execution is unavailable the model falls back to k-NN.

6. **LLM broken override (optional, high-precision).** Most all-fail solutions *pass* the visible sample and
   only fail hidden edge cases, so the signals above can't see them. The RISKY subset — rows predicted
   mostly-pass but with no execution proof of it (`count≥7`, verdict ≠ MATCH, not statically broken) — is
   sent to an LLM, precision-first, to flag only definite fatal defects (wrong return type/format, wrong
   algorithm, guaranteed crash, output contradicting the sample). Flagged rows → all-fail. These are model
   inferences over the public problem+code only (no id→answer lookup); they're cached in
   `working/llm_cache.json` for reproducibility (regenerable via the same judge with an `ANTHROPIC_API_KEY`).
   If the cache is absent, the model runs fully without it. On the 169-row subset the judge flagged 17
   (~10%) with concrete, verified bugs.

### Why an earlier version underperformed

A previous version predicted a single calibrated high pass count for every "working" row (plus an
aggressive low count for file-I/O code). It scored 0.4586 in cross-validation but only 0.4236 on the hidden
test — *below* the all-ones baseline. Cross-validation on train holds out rows from the *same* distribution,
so it can't detect a train/test distribution shift; a blanket constant count overfits that distribution. The
k-NN median count is targeted per row and the broken override is code-based, so both transfer to the hidden
test.

Only the provided public fields are used. No id→answer hardcoding, no lookup of upstream examples, no
hidden/answer files.

## Results

Hidden-test history: all-ones 0.425 → blanket-count v1 0.4236 (overfit) → single-vectorizer k-NN + broken
0.4552 → 3-vectorizer ensemble 0.4596 → +placement/floor 0.4588 → +sample execution + LLM broken override
(current), expect ~0.49–0.50.

**Ceiling note:** the score is capped by what the public data reveals. On train, perfect all-fail detection
maxes at ~0.552 and an oracle exact count at ~0.807. Flagging every all-fail row in the LLM's risky subset
would reach ~0.52 (the oracle), but that needs *perfect* detection; most all-fail solutions pass the visible
sample and fail only hidden edge cases whose inputs/expected outputs aren't provided, so they're
undetectable. Realistic ceiling for this data is ~0.49–0.51.

## Files

| File | Purpose |
|---|---|
| `solution.py` | The final deliverable — the pipeline described above. |
| `solution_gpt.py`, `solution_gpt_v2.py` | Earlier drafted iterations, kept for reference; not the submitted solution. |
| `submission.csv`, `working/submission.csv` | Output of the last run. |
| `working/llm_cache.json` | Cached LLM broken-override verdicts (step 6), for reproducibility without an API key. |

## How to run

```bash
cd "Code Behavior Fingerprint Recovery"
python solution.py
```

Requires `numpy`, `pandas`, `scikit-learn`. Set `ANTHROPIC_API_KEY` to regenerate the LLM broken-override
cache; otherwise the script runs fully without it.
