================================================================================
Middle Polish Archive Variant Lattice  —  solution notes
================================================================================

Run:  python3 solution.py <public_dir> <submission_out>
Files: solution.py (the only .py file, fully self-contained), readme.txt

working/submission.csv in this directory was produced by this pipeline on the
laptop-scale configuration described in section 2 (identical code path, only the
device / model width / epoch count reduced so a full 900-row run fits on a laptop
GPU).  It is provided as a checkable artefact; the graded submission is whatever
solution.py generates when it runs on the platform's A10G at full size.


--------------------------------------------------------------------------------
0. WHAT PROBLEM.md ASKS FOR  (written down before any code, per the brief)
--------------------------------------------------------------------------------

Submission schema (exact)
  CSV with exactly two columns, in this order:  case_id,variant_lattice
  One row per test case_id, each exactly once.
  variant_lattice is a JSON array with one entry per query_tokens position.
  Each position is a list of 1..3 candidate objects; each object has EXACTLY the
  keys "text" and "prob".
    - text : unique within its position, non-empty, trimmed, <= 160 characters,
             no control or surrogate characters.
    - prob : finite, in [0.01, 1.0], listed in NON-INCREASING order, summing to
             1.0 within 1e-6.
  extra_candidates = sum over positions of (n_candidates - 1) must not exceed the
  row's public expansion_budget.  Outer list length must equal len(query_tokens).

Evaluation metric (exact)
  Per position, with submitted (v_j, p_j) and true observed token t:
      p_true             = sum of p_j over candidates with v_j == t
      probability_credit = max(0, 2*p_true - sum_j p_j^2)
      char_similarity(v,t) = max(0, 1 - levenshtein(v,t)/max(len(v),len(t),1))
      expected_char      = sum_j p_j * char_similarity(v_j, t)
  Position groups: changed (query != observed), unchanged (query == observed),
  novel-variant (changed and the exact pair is not shown in any public training
  target or public calibration example).
  Row score =
      0.20 * changed_probability   (mean probability_credit on changed)
    + 0.10 * changed_top1
    + 0.10 * changed_char
    + 0.25 * novel_probability
    + 0.10 * novel_top1
    + 0.15 * full_changed_coverage (1 iff EVERY changed target is somewhere in
                                    its candidate list, else 0)
    + 0.10 * unchanged_probability
  Row scores are averaged within each profile_id; the final score is the
  unweighted mean of the profile means, times 100.

Guidebook domain
  NLP / seq-to-seq — guidebook section 5.1.  Not a Fine-tuning challenge and no
  model-type restriction is declared, so a from-scratch neural model is the
  natural fit and is what is used here.  Nothing pretrained is downloaded or
  loaded, so the stricter 5.5 rules are satisfied as well; the solution needs no
  internet access at all.

Challenge-specific rules honoured
  No external APIs; no external historical spelling pairs, corpora or
  dictionaries; no attempt to re-identify source passages; no hard-coding by
  case_id / profile_id / row order / filename; no exploitation of serialization,
  preparation or grader behaviour.  Boilerplate that contradicts the guidebook
  (e.g. "rule-based is fine") is ignored per guidebook 4.4 — this solution is a
  trained model end to end.


--------------------------------------------------------------------------------
1. APPROACH
--------------------------------------------------------------------------------

The task is reverse historical normalisation with few-shot adaptation to an
unseen source profile.  Data facts measured up front (train.csv only):

  * 38.4% of tokens change between editorial and source spelling.
  * Within one profile the query -> observed mapping is 99.4% deterministic, but
    across profiles only 91.5% — so the mapping is largely shared but has a real
    profile-specific component (e.g. the a->á rate ranges from 0.00 to 0.89
    across profiles, and the row's own calibration examples estimate it with
    r = 0.87).
  * A row's three calibration examples cover only ~29% of its query tokens
    verbatim; the other ~71% must be produced by generalising the character-level
    behaviour.  ~35% of changed positions are "novel" (pair unseen publicly).
  * Sequences are long (mean 25 tokens) while expansion_budget is small (2..8,
    a deterministic function of length), so candidate slots are scarce.

Model: a character-level NEURAL TRANSDUCER, trained from scratch in-script.

  (a) Piece induction.  Every training pair (query_token, observed_token) is
      aligned with a monotonic Levenshtein alignment (align_pieces).  Each source
      character is assigned the output substring it produced; a word-start marker
      slot carries any prefix insertion.  Concatenating the pieces reproduces the
      observed token exactly, so the transduction is lossless.  The label of a
      character is a special COPY symbol when the piece equals the character, and
      the literal piece otherwise.  The piece inventory is INDUCED from the
      training alignments (about 200-300 labels covering 99.6% of tokens); it is
      not a hand-written table of spelling rules.

  (b) Encoder.  The whole sentence is a single character stream (word-start
      marker + characters, per token).  Inputs per position: character embedding,
      position-in-word embedding, absolute position embedding, and the mean
      embedding of that token's analysis_codes.  Five pre-LN transformer blocks
      (d=320, 8 heads), each with self-attention over the sentence and
      cross-attention into a calibration memory.

  (c) Calibration memory (this is what makes it few-shot).  The row's three
      calibration examples are aligned the same way and encoded as a character
      stream whose embeddings include the GOLD piece label at each position
      (legitimately available: observed_tokens of the calibration examples are
      given inputs).  Two transformer blocks encode it; the query characters
      cross-attend into it.  The network therefore learns *when* to imitate this
      profile's own habits, rather than being told to copy.

  (d) Style summary.  A compact per-row vector summarising the same calibration
      evidence: a sqrt-histogram over the emitted pieces, a per-character "was it
      changed" rate, and a per-character "was it seen at all" mask, plus the log
      size of the evidence.  This is added to every query and memory position.  It
      is a SUPPORT feature: it is consumed by the network and never emits an
      answer.

  (e) Head.  Autoregressive within a token: the label at character i is predicted
      from the contextual state h_i and the embedding of the label emitted at
      i-1 (teacher-forced during training, with 15% of previous labels replaced
      by <unk> to limit exposure bias).  Beam search (width 8) over the token's
      characters yields up to 4 distinct source spellings with joint
      probabilities; the probability of the unchanged spelling is additionally
      computed exactly by forced decoding of the identity path.

  (f) Extra training signal.  Each profile's calibration sentences are also used
      as training targets with a leave-one-out memory (the other two examples),
      which adds ~2,200 supervised sentences on top of the 3,000-odd rows.

  (g) Up to three models (different seeds) are trained and ensembled by averaging
      log-probabilities at every beam step.  An exponential moving average of the
      weights is maintained for each model; raw vs EMA weights are chosen on the
      holdout.

Decoder: decision-theoretic lattice construction (make_lattice).

  Because probability_credit = max(0, 2*p_true - sum_j p_j^2) clips at zero, a
  second candidate can only earn credit at all when its probability exceeds
  ~0.293; below that it costs the top candidate almost nothing (2*eps^2) and
  earns nothing.  It follows that (i) probabilities should be as sharp as the
  format allows, and (ii) extra candidates pay for themselves only through the
  0.15 full_changed_coverage term.  This is derived analytically and then
  CONFIRMED by the in-script search on the holdout, which picks mode='sharp'
  (0.98/0.01/0.01) over renormalised probabilities and pmin=0.01 over larger
  values.

  Given that, the coverage probability of position i with k candidates is
  P(observed == query) + sum of the probabilities of the non-query candidates,
  and the probability that the whole row is fully covered is the product over
  positions.  Allocating the row's expansion_budget is therefore an exact
  knapsack DP maximising the sum of log coverage probabilities subject to
  sum(extras) <= budget, with per-position options 0/1/2 extras.

  Every free constant of the decoder — mode, pmin, temp, min_gain, max_c — is
  found by in-script coordinate ascent on the profile-disjoint train holdout,
  scored with an exact re-implementation of the PROBLEM.md metric (row_score).
  Nothing is pasted in from an offline experiment.


--------------------------------------------------------------------------------
2. VALIDATION STRATEGY AND SCORE
--------------------------------------------------------------------------------

Split: GROUP-AWARE (profile-disjoint).  12% of train profile_ids are held out;
no profile appears on both sides, which mirrors the real test condition (train
and test profiles are disjoint).  The holdout is used for (a) the raw-vs-EMA
choice, (b) the decoder-knob search, and (c) the reported CV number.  It is never
used to fit model weights or vocabularies.

HEADLINE VALIDATION NUMBER
  Running this exact solution.py pipeline end to end (only the device, model size
  and epoch count reduced so it fits a laptop: 1 seed, d=256 / 4 layers / 1 memory
  layer, 15 epochs) on the full public train.csv gives, on its own 12%
  profile-disjoint holdout (326 rows, 87 profiles):

      HOLDOUT CV SCORE 62.148
        changed_top1 0.708 | novel_top1 0.489
        unchanged_top1 0.933 | full_changed_coverage 0.279
        chosen knobs: mode=sharp, pmin=0.01, temp=1.25, min_gain=0.002, max_c=3
        (identity baseline on the same holdout: 16.9)

  The submitted configuration is strictly larger (d=320 / 5 layers / 2 memory
  layers, up to 24 epochs, up to 4 seeds ensembled), so this is a floor, not a
  ceiling.  The script prints its own HOLDOUT CV SCORE on every run.

Development ladder, measured on a wider 20% profile-disjoint holdout of train
(692 rows, 146 profiles) with the exact PROBLEM.md metric:

    identity (submit the query token everywhere)                17.0
    global majority lookup over train pairs                     33.2
    row calibration lookup, backing off to the global lookup     34.5
    neural transducer, 1 seed, d=256/4 layers/1 memory layer     57.2

  Component breakdown of the transducer run:
    changed_probability 0.685 | changed_top1 0.685 | changed_char 0.935
    novel_probability   0.444 | novel_top1   0.444
    full_changed_coverage 0.246 | unchanged_probability 0.942
    budget actually spent: 5.07 extra candidates per row

  Decoder-knob sensitivity measured on the same holdout (this is what the
  in-script search reproduces):
    max_c = 1 / 2 / 3        ->  54.9 / 56.9 / 56.9   (spending the budget is
                                                       worth ~2 points)
    mode  = sharp / renorm   ->  56.9 / 56.5
    pmin  = 0.01/0.05/0.1/0.2->  56.9 / 56.8 / 56.7 / 56.0
    min_gain = 0 is best (always spend the budget when it helps coverage)

Both holdouts agree on the ordering of every design decision; the absolute
numbers differ because the 12% holdout leaves more data for fitting (3,174 vs
2,808 rows) and is a smaller, easier sample of profiles.

The submitted run also keeps an exponential moving average of each model's
weights and chooses raw-vs-EMA on the holdout (raw won at 62.1 vs 41.4 in the
validation run, so EMA is carried purely as insurance against a late-training
collapse), and it refuses to overwrite the identity placeholder at all if the
trained ensemble fails to beat the identity baseline on the holdout.

Note on the local novel-variant flag: PROBLEM.md defines novel positions
relative to all public training targets and all public calibration examples.
The local flag is computed from the TRAIN split's targets and calibration
examples plus the holdout row's own calibration examples, deliberately excluding
anything derived from test rows (see the leakage statement).  It is therefore a
slightly different set from the grader's, which affects only the reported number
and the knob search, never a prediction.


--------------------------------------------------------------------------------
3. LEAKAGE STATEMENT
--------------------------------------------------------------------------------

Every transform, vocabulary and statistic in this solution is fitted on training
data only.  Specifically:

  * The character vocabulary, the induced piece inventory, the analysis-code
    vocabulary and the (character -> label) transition index are built from the
    TRAIN SPLIT only (the 88% fit portion), never from test.csv and never from
    the holdout.  Unseen characters/codes/pieces at inference map to <unk>.
  * Model weights are fitted on the train split only.
  * The decoder knobs are searched on the profile-disjoint train holdout only.
  * The public-pair set used for the local novel-variant flag is built from the
    train split and from each holdout row's own calibration examples.  Nothing
    is counted over test rows, not even for validation bookkeeping.
  * There is no pd.concat / merge / append of train and test anywhere.
  * Test data is touched only by transform(test) and predict(test).  Every
    operation applied to a test-derived variable is per-row: features come from
    that row's own query_tokens, analysis_codes and calibration_examples; the
    lattice for a row is built from that row's model output and that row's
    expansion_budget alone.  Batching sorts rows by length purely for padding
    efficiency; every padded position is masked out (key_padding_mask on both the
    query and the memory, a count mask on the analysis codes, and an explicit
    length mask in the beam), so batch composition carries no information between
    rows.  Verified by re-running inference over all 900 test rows at batch size 1
    and at batch size 24: the top-1 candidate is identical at every one of the
    23,860 positions and all 900 lattices match; 4 positions differ only in the
    ORDER of their rank-2/rank-3 candidates, at the 1e-6 level, which is float
    reduction-order noise from different padded tensor widths, not information
    flow.
  * No statistic is aggregated across test rows: no counts, no argmax
    distribution, no normalisation to a target class balance, no pseudo-labelling
    or self-training, no test-time adaptation.
  * test.csv's calibration examples are used ONLY as inputs to the row they
    belong to.  They are deliberately NOT added to the training set, even though
    they contain (query, observed) pairs, because that would be a cross-row use
    of test data.

--------------------------------------------------------------------------------
4. HARDCODING STATEMENT
--------------------------------------------------------------------------------

No discovered generation pattern is hard-coded.  Concretely:

  * There is no dict, lookup table, if-chain, template, n-gram table or regex
    anywhere in solution.py that maps a query token, phrase or character to an
    output spelling.  Grepping the source for orthography literals returns
    nothing: the only string constants are dict keys, log format strings and the
    two decoder mode names.
  * align_pieces is a generic monotonic Levenshtein alignment.  It encodes no
    Polish orthography; it is the standard way to turn string pairs into
    per-character labels, and it is verified lossless (concatenating its pieces
    reproduces the observed token for all 86,354 training token pairs).
  * The piece inventory and the transition index are LEARNED from the training
    alignments (a label set, like a tagset or a BPE vocabulary), not asserted.
    Which label to emit where is decided entirely by the trained network.
  * style_vec is a frequency summary of the row's own calibration evidence.  It
    is a feature vector consumed by nn.Linear inside the model; it cannot emit a
    token.  Per the brief, statistical features are allowed as SUPPORT into a
    trained model, and that is exactly their role here.
  * Every numeric constant that influences an output — mode, pmin, temp,
    min_gain, max_c — is searched IN-SCRIPT by coordinate ascent on a train
    holdout against the exact PROBLEM.md metric.  Nothing was tuned offline and
    pasted in.  The remaining constants are ordinary training hyper-parameters
    (learning rate, dropout, batch size, model width/depth, beam width) and time
    limits, none of which decides an output family.
  * Nothing keys on case_id, profile_id, row order or filename.

STRIP-THE-ML TEST — with all trained models removed, the pipeline produces the
identity lattice: one candidate per position whose text is the query token
verbatim, with probability 1.0.  Measured with this script's own metric
implementation, that scores 16.9/100 on the holdout (it is the null "copy the
input" prediction: it earns unchanged_probability and partial changed_char, and
scores exactly zero on changed_top1, novel_probability, novel_top1 and
full_changed_coverage).  The trained transducer is what turns 16.9 into 62.1; the
alignment, the induced label set and the calibration features produce nothing on
their own.  The script computes this same identity score at run time and logs it
next to the model score.


--------------------------------------------------------------------------------
5. ROBUSTNESS / RUNTIME
--------------------------------------------------------------------------------

  * Paths come from sys.argv only; the parent directory of the submission path
    is created before writing.  The script reads only public_dir and writes only
    submission_out.
  * A schema-valid placeholder submission is written immediately after test.csv
    is read, before any training, and is overwritten at the end.  All heavy work
    runs inside a try/except that leaves the last valid file in place on failure.
  * Per-row lattice construction is wrapped so that a single bad row falls back
    to its placeholder instead of killing the run.  There are no assertions on
    dataset shape or size.
  * write_sub is the single serialisation choke point: it coerces candidates into
    the {"text":..,"prob":..} object form and refuses to emit a file with a row
    count different from len(test).
  * validate_lattice is a final safety net applied to every submitted row: it
    sanitises text (strip, drop control/surrogate characters, cap at 160 chars),
    de-duplicates, re-enforces 1..3 candidates, re-enforces the expansion budget,
    and repairs probabilities to be in [0.01, 1.0], non-increasing and summing to
    1.0.  It was fuzz-tested over 4,000 randomised adversarial rows (empty and
    whitespace-only candidates, duplicates after sanitisation, control
    characters, 200-character strings, budget 0 and 8) with zero constraint
    violations.
  * Wall-clock guard: both the number of epochs AND the number of seeds are
    planned after the first epoch from its measured cost on whatever machine this
    actually runs on, so the cosine schedule always completes rather than being
    cut off mid-anneal, and an extra seed is only started if at least ~12 epochs
    fit in its share.  New training stops being launched after 2350 s and any
    running loop breaks at 2850 s, leaving ample room for inference and writing
    the submission inside the 1 h target.  A single model plus inference always
    fits.  Reference timing at reduced scale on a laptop GPU: 891 s to train,
    25 s for both holdout passes plus the knob search, 17 s to predict and write
    all 900 test rows.
  * Seeds are fixed (torch / numpy / random).

--------------------------------------------------------------------------------
6. WHAT WORKED / WHAT DID NOT
--------------------------------------------------------------------------------

Worked
  * Framing the task as a monotonic character transducer with an induced piece
    inventory.  It is lossless, it composes unseen spellings (which is what the
    0.35-weight novel-variant terms reward), and it gives exact per-token string
    probabilities via beam search.
  * Cross-attention into the aligned calibration examples.  Top-1 accuracy on
    changed positions is 0.85 when the token appears verbatim in the row's
    calibration, 0.71 when the token was seen in training, and 0.52 when it was
    seen in neither — the memory is clearly being exploited.
  * Working out the decoder analytically before searching.  The clipping in
    probability_credit means sharp probabilities dominate; the search confirmed
    it (sharp 56.9 vs renormalised 56.5) and the knapsack on coverage is worth
    about 2 points over never spending the budget.
  * Using each profile's calibration sentences as extra leave-one-out training
    targets (+~2,200 sentences).

Did not work / rejected
  * Lookup-style baselines plateau at ~34.5 because novel-variant positions are
    35% of changed positions and are unreachable by construction — they score 0
    on both novel terms, which are 35% of the metric.
  * Splitting probability across candidates (the intuitive "calibrated
    uncertainty" reading of the metric) is actively harmful: the max(0, .)
    clipping means a hedged second candidate usually earns nothing while
    diluting the first.
  * Feeding a global train-set token -> spelling lexicon as an explicit feature
    was rejected: the model already memorises those regularities in its weights,
    and an explicit lexicon feature is exactly the lookup-table shape the
    hardcoding ban targets.
  * Adding test.csv's calibration pairs to the training set was rejected as a
    cross-row use of test data, despite being tempting (they are gold pairs).
  * Sharper profile-style features (the per-(character -> label) transition rate
    measured on the row's calibration, ~440 extra dimensions) were implemented and
    A/B-tested against the baseline feature set at identical config and epoch
    count.  They HURT badly and were removed: the run tracked the baseline until
    mid-training (55.6 vs 55.8 at epoch 9) and then collapsed as the learning rate
    annealed (48.7 vs 57.2 at epoch 14), with changed_top1 falling 0.656 -> 0.536
    while unchanged accuracy rose — the model had learned to lean on the style
    vector and under-predict changes.  The submitted solution uses the coarser
    style summary that validated at 57.2.  This is also why the script keeps an
    EMA copy of every model, picks raw-vs-EMA on the holdout, and refuses to ship
    a model that does not beat the identity baseline there.

Main residual error mode
  á/a confusion accounts for about 55% of remaining top-1 errors (739 cases of
  predicting á where the source has a, 395 the other way round): whether a given
  'a' is written 'á' depends jointly on the word and on how strongly the profile
  marks it.  Oracle recall inside the model's top-4 candidates is 0.96 overall
  (0.89 on changed positions) against 0.685 top-1, so the headroom is in ranking,
  not in candidate generation.  The obvious attack on it — handing the network a
  sharper per-transition profile-rate feature — was tried and made things worse
  (see above); the honest conclusion is that more capacity and more seeds, not
  more hand-built style statistics, is what is left.
