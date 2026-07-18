Deadzone: Hidden-State Transduction from Paired Symbol Sequences
================================================================

APPROACH (one sentence)
-----------------------
Treat the task as bidirectional sequence tagging with latent-state tracking: a
from-scratch neural encoder reads BOTH opaque symbol strings jointly, infers the
hidden automaton state, and a linear-chain CRF decodes a coherent 8-class label
string; the Viterbi decoder is then calibrated on a train-holdout to directly
maximize the exact competition metric.

WHY THIS SHAPE (evidence from the training split)
-------------------------------------------------
* The symbol->class map is genuinely one-to-many. H(state)=2.68 bits, and
  conditioning on the current actor symbol only lowers it to 2.32 bits; a
  per-position argmax lookup on the actor symbol reaches just ~37% train
  accuracy (the published "per-symbol lookup" baseline scores 19.31). Local
  features cap out around the GBT baseline (43.34): H(state | actor,foe,kind) is
  still ~2.06 bits.
* Sequential state tracking is the whole game. H(state | prev_state)=0.35 bits
  and argmax(actor_symbol, TRUE prev_state) reaches 96.3% oracle accuracy;
  self-transitions are ~95% and states persist in long runs (mean length ~19,
  only ~5% of runs are length 1, boundary rate ~4.7%). A model that carries and
  infers the hidden state is required to beat the local baselines.
* The class prior is strongly archetype-dependent: the share of `atk` ranges
  from 8% to 82% across actor_kind values, so actor_kind / foe_kind embeddings
  are a large, fully-learned lever.
* ~28% of outputs (hit/evd/rsp) are foe-forced, so both strings must be read
  jointly; 22% of input positions are blanked to [X].

MODEL ARCHITECTURE / ALGORITHM
------------------------------
A single genuinely-trained PyTorch sequence tagger (trained from scratch on
every run; no pretrained weights -- the alphabet is salted, so none would help):

  Input per position (whole sequence = 60-symbol prefix + n_scored):
    - ONE shared symbol embedding table (dim 64) applied to: the actor symbol,
      the foe symbol, and each stream's forward-filled last-non-blank symbol
      (imputes the 22% [X] holes). A shared table lets foe symbols regularize
      actor symbols.
    - the actor(x)foe Hadamard interaction (targets foe-forced hit/evd/rsp).
    - explicit blank flags for actor and foe.
    - record-level categoricals broadcast to every position: arena (dim 8),
      actor_kind (16), foe_kind (16), and a learned matchup vector
      MLP([actor_kind, foe_kind]) -> 16.
  Encoder: Linear -> LayerNorm -> a local Conv1d(k=3)+Conv1d(k=5) residual
    front-end -> a 2-layer bidirectional LSTM/GRU (hidden 256/direction). It
    reads the entire sequence so the 60-symbol prefix warms the hidden state
    before the scored region begins.
  Head: an MLP emission head (-> 8 logits) plus a linear-chain CRF (learned 8x8
    transition matrix + start/end potentials). The CRF's diagonal learns the
    ~95% self-transition; Viterbi decoding emits coherent runs and places the
    rare boundaries, which is what earns BoundaryF1.

  Training loss = CRF negative log-likelihood + 0.5 * class-weighted
    cross-entropy (weights = (median_freq/freq)^0.5, so rare classes rsp/shd/grb
    get gradient the CRF-NLL alone suppresses) with 0.02 label smoothing on the
    aux CE. AdamW + OneCycleLR, gradient clipping, dropout 0.3, EMA of weights
    (decay 0.999, used for inference), and blank-augmentation (an extra ~10% of
    non-blank symbols randomly masked to [X] during training, off for the last 2
    epochs) for robustness to the lossy inputs.

  Ensemble: several seeds (mixed BiLSTM + BiGRU) averaged in log-probability and
    transition space, then decoded once.

  DECODE CALIBRATION (the metric-facing step): the Viterbi decoder has three
    knobs -- a transition scale `alpha`, an extra switch penalty `lam0`, and
    per-class additive gains `g[8]`. They are tuned by coordinate ascent on the
    held-out split to DIRECTLY maximize the exact pooled score
    (0.70*MCC8 + 0.30*BoundaryF1). Because MCC8 is chance-corrected, nudging the
    rare classes up recovers diagonal mass and lifts the score. These knobs are
    small decode-time transforms fit on train data only; the predictions
    themselves come from the neural CRF, never a lookup/frequency table.

FEATURE ENGINEERING
-------------------
All features are learned embeddings / flags fed to the neural model (no
hand-coded symbol->class rules): shared symbol embedding on actor/foe/fills,
actor(x)foe Hadamard, blank flags, arena/actor_kind/foe_kind embeddings, and a
matchup MLP. The forward-fill and blank flags are the only light preprocessing
and they exist to feed the model, not to replace it.

VALIDATION STRATEGY
-------------------
Fixed, seeded record-level 90/10 split of TRAIN (never split within a record;
test categories are a strict subset of train, so a plain split is leak-free).
All ensemble seeds train on the 90%. The 10% holdout is the single leak-free
bank used for (a) reporting, and (b) tuning the decode knobs. Metric usage
follows PROBLEM.md exactly: all held-out scored positions are POOLED into one
8x8 confusion for MCC8 and one boundary tally for BoundaryF1 (MCC is
non-additive, so it is never averaged per-record). The exact metric is
reimplemented in-script and cross-checked (constant prediction -> 0, perfect ->
100, uniform-random -> 2.71, matching the published references).

Reference scores (published grader): per-symbol lookup 19.31, GBT-over-span
43.34, bidirectional RNN 58.95, perfect 100.

Held-out composite score (this solution), measured on the local 90/10 holdout
with the exact shipped code (20 epochs, full boundary-modulated tuned decode),
by ensemble size (LSTM-only local validation; the shipped run also adds BiGRU
members on the CUDA target):
  1 seed  -> 58.19  (MCC8 0.652, BoundaryF1 0.419)
  2 seeds -> 59.66  (MCC8 0.670, BoundaryF1 0.426)
  3 seeds -> 60.50  (MCC8 0.679, BoundaryF1 0.432)
  4 seeds -> 60.12  (MCC8 0.676, BoundaryF1 0.427)
The score plateaus around 60.1-60.5 (the small 3->4 dip is ordinary ensemble /
holdout-tuning noise on the fixed 882-record holdout). Even a single model beats
the GBT baseline (43.34) by a wide margin; the 2-seed ensemble already clears the
bidirectional-RNN reference (58.95), and the full ensemble does so with ~1.5-point
margin. Each solution.py run logs its own holdout score. The emitted submission
was validated: 3174 rows, every test seq_id exactly once, every row exactly
n_scored tokens, all tokens in the 8-class set.

The stacking of levers (holdout): single model argmax ~53 -> ensemble argmax
55.3 -> + CRF-Viterbi 57.8 -> + metric-tuned knobs (alpha/lam0/gains) 59.3 ->
+ boundary-head modulation 59.7. The shipped configuration trains a 5-seed mixed
BiLSTM + BiGRU ensemble; each solution.py run logs its own holdout score. (GRU
members are used only on the CUDA target; local MPS validation is LSTM-only due
to a device kernel quirk -- the code path is identical.)

LEAKAGE STATEMENT
-----------------
Every transform and statistic is fit on TRAIN only and test is used for
inference (transform + predict) only:
  * The symbol vocabulary is fixed (c000..c253 + [X] + UNK + PAD); the arena /
    actor_kind / foe_kind encoders are built from TRAIN rows only, each with an
    explicit UNK slot for unseen categories.
  * Class weights, the 90/10 holdout split, and all decode knobs (alpha, lam0,
    per-class gains) are computed from TRAIN rows only (the holdout is a subset
    of train). No knob, threshold, or statistic is derived from test.
  * There is no train+test concatenation anywhere, no test-derived statistics,
    no pseudo-labeling, and no test-time adaptation. The `states` column is read
    only in the train code path. Test rows are only ever passed through the
    encoder and the (train-fitted) decoder.
  * Fixed seeds (python/numpy/torch). Device auto-selects cuda, else mps, else
    cpu. The script only reads from the given public_dir and writes the two
    submission paths.

RUNTIME / ROBUSTNESS
--------------------
Runtime contract: `python3 solution.py <public_dir> <submission_out>`. The
submission is written ONLY to argv[2] (its parent dir is created); the script
reads only from public_dir and writes only that one output file -- no hardcoded
paths. A schema-valid NON-constant placeholder submission is written to argv[2]
immediately after reading test.csv and overwritten after each ensemble member
finishes, so a valid, complete submission always exists at that path. A
wall-clock guard stops launching new seeds at ~3000s and stops training at
~3300s (and skips re-tuning the decode knobs if past that, reusing the previous
member's knobs), then always proceeds to decode + write, keeping the run inside
the 1.5h budget even if a seed is cut short.

WHAT WORKED / WHAT DID NOT
--------------------------
Worked: (1) carrying hidden state with a prefix-warmed BiLSTM + CRF Viterbi is
the decisive jump over local per-position classification; (2) the decode-knob
calibration against the exact metric gives a large, cheap gain, especially by
recovering rare-class MCC mass; (3) actor_kind/foe_kind embeddings and the
shared-alphabet actor(x)foe fusion add real signal (archetype prior + foe-forced
states); (4) EMA + seed ensembling reduce variance and sharpen the averaged
posteriors so Viterbi can run at a lower switch penalty without spurious
boundaries.
Did not help / avoided: plain per-position argmax jitters within runs and wrecks
BoundaryF1; a constant/majority string scores exactly 0 (chance-corrected MCC);
heavy rare-class loss reweighting decalibrates the posterior, so the rare-class
push is moved to the decode-time gains instead.
