Multilingual Document Reading-Order Graph Repair
================================================================================
Run:  python3 solution.py <public_dir> <submission_out>
      (locally: python3 solution.py ./dataset/public ./working/submission.csv)

--------------------------------------------------------------------------------
0. WHAT PROBLEM.md SPECIFIES  (copied out before any code was written)
--------------------------------------------------------------------------------

Submission schema (exact)
  * CSV with exactly two columns, in this order:  id,repair_sequence
  * exactly one row per identifier in test.csv (no missing / extra / duplicate ids)
  * repair_sequence is a non-empty ';'-separated list of operations; every operation
    has exactly five '|'-separated fields:  OPERATION|RELATION|SOURCE|TARGET|END
      - OPERATION in {DEL_EDGE, ADD_EDGE}
      - RELATION  in {CONTAINS, NEXT_BLOCK, NEXT_LINE}
      - SOURCE/TARGET must be node ids of that page with the right kinds
        (CONTAINS: BLOCK->LINE, NEXT_LINE: LINE->LINE, NEXT_BLOCK: BLOCK->BLOCK)
      - the fifth field is always the literal token END
  * canonical order: ALL DEL_EDGE ops first, then ALL ADD_EDGE ops; within each
    group sorted lexicographically by (relation, source, target); no duplicates
  * no surrounding whitespace, no empty operations
  * everything is case sensitive
  Any violation of the above zeroes the WHOLE submission, so the script emits
  the repair as the symmetric difference between the observed edge set and the
  predicted intended edge set, sorted exactly as required (see `ops_string`).

Evaluation metric (exact, implemented verbatim in `page_score`)
    page_score = 0.73 * operation_f1 + 0.02 * edge_f1 + 0.25 * exact_graph
  operation_f1 over (operation, relation, source, target) tuples; edge_f1 over the
  final (relation, source, target) edges after applying the ops; exact_graph is 1
  iff the reconstructed edge set equals the intended one.  Final score = arithmetic
  mean over pages.  An inapplicable edit or a structurally invalid resulting graph
  scores 0.0 for that page.  There is no tail / worst-group term - the mean is over
  pages, explicitly "rather than an average over layout groups".
  => the metric is dominated by getting the page EXACTLY right (0.73+0.02+0.25=1.0
     when exact, and operation_f1 is 0 whenever a different single corruption is
     predicted, because the op sets are then disjoint).  All decode tuning below
     optimises this exact expression on a train holdout.

Guidebook domain
  NLP / seq-to-seq structured prediction (guidebook 5.1) - PROBLEM.md itself calls
  it "a sequence-to-sequence structured-prediction problem" and the decisive
  evidence is multilingual OCR text.  Layout geometry and the page image are extra
  evidence, not the primary modality.  PROBLEM.md declares no model-type
  restriction and does not categorise this as a Fine-tuning challenge, so a
  fine-tuned general-purpose multilingual backbone (guidebook 4.1) is used, and
  section 4.3's grey-area warnings about regex / n-gram shortcuts are respected:
  no regex, no n-gram table and no retrieval produces any part of the answer.

Challenge-specific rules honoured
  * "do not use ... reverse-image search, OCR-text lookup, perceptual matching, or
    record linkage against public document archives" - nothing external is queried;
    the only download is the general-purpose backbone weight file.
  * "no external copies of the source pages or their original layout annotations",
    "no manual annotation of individual test pages" - none.
  * "assumptions that row order, file names, opaque page identifiers, node-number
    suffixes, or operation counts encode the target" are forbidden - the model
    never sees the page id, the file name, the row index or a raw node identifier.
    Node ids are used only as opaque keys to build per-page integer indices, and
    those indices are assigned by the OBSERVED graph traversal order, never by the
    numeric suffix.  No feature is derived from the id string.

--------------------------------------------------------------------------------
1. APPROACH
--------------------------------------------------------------------------------

Observation from the data (used to define the hypothesis space, not the answer):
PROBLEM.md states the observed graph is produced from the intended graph by one
controlled local corruption, of one of four kinds.  Reconstructing the intended
graph for all 3359 train rows confirms exactly four families:

    A  swap of two adjacent lines inside one block          (884 rows)
    B  swap of two adjacent blocks in page order            (784 rows)
    C  exchange of two lines between two blocks, keeping    (835 rows)
       each line's position index inside its new block
    D  C and B applied together                             (856 rows)

Every one of the 3359 training targets is reachable by inverting exactly one such
corruption (verified: 3359/3359 coverage), so the set of *candidate intended
graphs* for a page is enumerable:  |A| = L-B, |B| = B-1, |C| = cross-block line
pairs, |D| = |C|x|B|  (median ~300, up to ~6800 candidates per page).

That candidate set is the structured-prediction hypothesis space - the analogue of
a CRF's label set.  It contains no information about which candidate is right; a
trained model has to choose.  The important properties of the page turn out to be:
  * the OCR text of consecutive lines is genuinely continuous prose, so the
    decisive evidence is linguistic ("... give our clients a mix of all" is
    followed by "the programs which has been proven ...");
  * bounding boxes are quantised to a 1/16 grid and perturbed hard.  Over the
    43917 consecutive line pairs of the intended graphs, y is strictly increasing
    for only 49.8%, equal for 13.1% and DECREASING for 37.1%.  Geometry therefore
    cannot order lines locally at all; it is only a weak global prior.

--------------------------------------------------------------------------------
2. MODEL
--------------------------------------------------------------------------------

Text encoder (fine-tuned in-script, every run, from the raw provided data)
  distilbert-base-multilingual-cased (WordPiece, covers en / ja / zh_hans).  All
  line texts of a page are packed into ONE sequence in observed reading order,
  "[CLS] line0 [SEP] line1 [SEP] ...", with a per-line token budget so the whole
  page fits in 352 tokens; when a line is longer than its budget its head and tail
  are kept (the junction between lines is what matters).  The budget is allocated by
  water-filling (`alloc_budget`): short lines take less than their fair share and the
  surplus flows to the long ones, which keeps 93.0% of the real tokens instead of the
  88.5% a uniform per-line cap keeps, at identical compute.  This makes the encoder a
  true cross-encoder over the page: it can see directly that the text flow breaks
  between two particular lines.  Per line we pool four vectors: mean over its
  tokens, its [SEP], its first-3 tokens ("head") and its last-3 tokens ("tail").
  The 92M-parameter WordPiece embedding table is frozen; the 6 transformer layers
  (42.5M parameters) are fine-tuned.
  Fallback: if the backbone cannot be fetched (no internet in the grading
  environment), a character-level transformer is built from a vocabulary fit on
  the TRAIN texts only and trained from scratch with the identical interface.  The
  pipeline is otherwise unchanged, so the run can never die on a download failure.

Page transformer
  3-layer / 8-head pre-norm transformer, width 256, over L LINE tokens + B BLOCK
  tokens + a CLS token.  Line tokens = layout features + text projections; block
  tokens = block layout features.

Heads
  (a) three locally-normalised models of the INTENDED graph, trained on the
      intended graphs reconstructed from train.csv:
        p(next line | line)   log-softmax over the page's lines + END   [L, L+1]
        p(next block | block) log-softmax over the page's blocks + END  [B, B+1]
        p(block | line)       log-softmax over the page's blocks        [L, B]
      Each pairwise logit is a sum of: a content bilinear term, a *junction*
      bilinear term <tail_i , head_j> (the actual text continuity signal), a small
      pairwise MLP, and learned bias tables indexed by observed relative position
      and same-block-ness.  Those bias tables are nn.Embedding parameters trained
      from data - they let the model absorb the "the observed order is usually
      right" prior cheaply so its capacity goes into the text signal (adding them
      moved the next-line NLL from 2.54 to 1.07 in one epoch).
  (b) three pairwise discriminative heads scoring the corruption participants
      (which NEXT_LINE junction was swapped / which block pair / which line pair),
      plus an auxiliary per-line "is this line misplaced" head.

Energy and exact decoding
  A candidate intended graph G differs from the observed graph in a handful of
  edges, and the three relation types are affected independently, so the change in
  graph log-likelihood is computed exactly as a gather over the three log-softmax
  matrices (unit-tested against a brute-force recomputation: max error 1.6e-14).
      E(G) = mix[f,0] * dLogLik(G)  +  mix[f,1] * pairHeadScore(G)  +  bias[f](page)
  with one mix pair per corruption family f, all learned nn.Parameters.  bias[f] is
  NOT a global scalar: which family a page suffered is a page-level question, so it is
  produced by a small head reading the page CLS vector together with the candidate
  counts (the families have very different candidate-set sizes, which systematically
  biases their log-sum-exp).  Its last layer is initialised to zero, so training starts
  exactly where a global per-family scalar would, and it is tanh-bounded to +-3.  Because E(D(p,k)) = E_C(p) + E_B(k) +
  bias, the partition function over the whole candidate set factorises:
      logZ = logsumexp( LSE(u)+b0, LSE(v)+b1, LSE(w)+b2, LSE(w)+LSE(v)+b3 )
  so the model is trained with an exact globally-normalised cross-entropy over all
  candidates (not sampled negatives), and inference is a plain argmax over the
  same set.  The predicted repair is the symmetric difference between the observed
  edge set and the argmax candidate's edge set, which is applicable and
  structurally valid by construction.

Losses (all on provided labels)
  exact global candidate cross-entropy  +  NLL of the intended graph under the
  three dense heads  +  0.3 * (within-family cross-entropies + misplaced-line BCE).

Training
  AdamW, encoder lr 3e-5 / head lr 4e-4, weight decay 0.01, grad-norm clip 1.0.
  The LR follows a warmup + cosine schedule driven by WALL CLOCK, annealing to
  zero exactly at min(3000 s guard, 18 epochs worth of time measured after the
  first epoch), so the full compute budget is used on any machine and the schedule
  still completes if the epoch cap binds first.  Non-finite loss / gradient steps
  are skipped rather than crashing, and if training raises (e.g. the GPU cannot
  hold the batch) it is retried with batch size 4 and then 2.  On CUDA the training
  step runs under bf16 autocast (batch 12); inference always runs in fp32.  bf16 was
  chosen over fp16 because it keeps fp32's dynamic range, so no GradScaler is needed
  and the log-softmax / logsumexp energy arithmetic cannot overflow - measured on CPU
  bf16, the loss moves by 4e-4 relative and the energies by ~0.01.  The best epoch on
  the holdout is the one that is kept.

Robustness
  A schema-valid submission is written before any heavy work (0.3 s into the run)
  and overwritten at the end.  Per-page failures fall back to that placeholder
  instead of raising; dataset-shape surprises are logged, never asserted.  Every
  candidate the decoder can emit is applicable and structurally valid by
  construction, so no page can be zeroed for validity.

--------------------------------------------------------------------------------
3. FEATURE ENGINEERING
--------------------------------------------------------------------------------
Per LINE (49 values): normalised bbox and derived width/height/area/aspect, angle,
text length and placeholder-char ratio, page aspect, position in the observed
global reading order (raw + sinusoidal), position inside its observed block,
first/last/singleton flags, block size, block index, and the line-vs-block bbox
offsets and containment fraction, plus a language one-hot.
Per BLOCK (39 values): the same geometry plus member-line count, member-line
centre mean/std and extent, total member text length, block index, language.
All of these are per-page deterministic functions of that page's own record.
Block sizes / block membership come from the observed graph - note that block
SIZES are invariant under all four corruptions, so they are uncorrupted evidence.

--------------------------------------------------------------------------------
4. VALIDATION STRATEGY AND SCORE
--------------------------------------------------------------------------------
Holdout: 10% of the training rows, stratified by language, drawn with a fixed
seed, held out from all gradient updates.  It is used for (i) best-epoch
selection and (ii) the decode-offset search.  Every reported number is the exact
PROBLEM.md page score, computed by the same code path as the metric section above.

Because the four families are scored jointly, the family-level calibration is the
noisiest part of the decode.  Four scalar offsets (one per family) are therefore
searched IN-SCRIPT by coordinate ascent on the holdout against the real metric -
they are not constants pasted in from an offline experiment, and if the search
does not improve on the un-offset decode the offsets are reset to zero.  The same
one-round search is applied when scoring each epoch, so epoch selection is not
distorted by calibration noise.

MEASURED RESULT (full local run: python3 solution.py ./dataset/public ./working/submission.csv)

  fit 3023 pages / holdout 336 pages, Apple-M-series MPS, fp32
    epoch 0   holdout 0.3032   exact 0.1220
    epoch 1   holdout 0.3649   exact 0.1935
    epoch 2   holdout 0.4053   exact 0.2738
    epoch 3   holdout 0.4162   exact 0.2857     <- best epoch, kept
    epoch 4   holdout 0.3971   exact 0.2738
    time guard stopped training before epoch 5

  FINAL HOLDOUT SCORE = 0.4209   (exact_graph rate 0.2917)
    before the in-script offset search: 0.4040 (exact 0.2768)
    searched offsets: [-1.80, -1.50, 1.00, 0.30]
    total wall clock 2567 s; 841/841 test rows predicted by the model (the placeholder
    was not used for any row)

  Three full runs were made while tuning; all numbers below are the same holdout, the
  same seed and the same metric, so they are directly comparable:

    run   configuration                                        holdout   exact
    ---   --------------------------------------------------  -------   -----
     1    global per-family scalar bias                         0.4195   0.2887
     2    + water-fill, per-family mix, 4-line window head A    0.3992   0.2470
     3    run 2 minus the window head, + page-conditional bias   0.4209   0.2917   <- shipped

  Run 3 is what ships.  It is also ahead of run 1 at every matched epoch (0.3649 vs
  0.3488 at epoch 1, 0.4053 vs 0.3985 at epoch 2), which is better evidence than the
  final figure alone: the 0.0014 gap between runs 1 and 3 is far inside the standard
  error of a 336-page holdout (0.4/sqrt(336) = 0.022).

  Per-family holdout scores and, separately, the within-family top-1 accuracy (how
  often the right parameter is chosen when the family is GIVEN):

                        A        B        C        D
    page score        0.126    0.609    0.633    0.428
    within-family     0.143    0.734    0.618    0.296

  Reading these two rows together is what drove the last change.  In run 2, family B
  had 0.719 within-family accuracy but only 0.396 page score - it was finding the right
  block pair and then losing the page to the wrong FAMILY.  The page-conditional family
  bias was added for exactly that, and B's page score moved 0.396 -> 0.609 at unchanged
  within-family accuracy.  The remaining weakness is family A (0.143 against a ~0.08
  chance rate) and family D, which must get the line pair and the block pair right at
  once (0.618 x 0.734 = 0.454 if independent, but only 0.296 observed).

  Caveat: only 4-5 epochs fit inside the 3000 s guard on this machine.  The schedule is
  wall-clock driven, so a faster GPU trains for more epochs in the same budget - though
  runs 2 and 3 both peak at epoch 2-3 and fall back afterwards, so extra epochs are
  worth much less than the first run's four rising points suggested.  Best-epoch
  selection on the holdout keeps the peak either way.

  Provenance of ./working/submission.csv: it is the output of run 3, produced by the
  current script.  Independently re-checked against every PROBLEM.md format rule plus
  the applicability and structural-validity conditions: all 841 rows pass, and all 841
  produce a structurally valid final graph, so no page can be zeroed for validity.

Baselines for reference (measured on all 3359 train rows):
  * strip-the-ML, all model scores set to 0      : score 0.0891, exact 0.0330
  * strip-the-ML, model scores replaced by noise : score 0.1430, exact 0.0101
Note on the grading split: PROBLEM.md says train/test are grouped by language and
coarse layout family, and that no group crosses the split.  The public package
does not expose the layout-group id, so the holdout is language-stratified only;
it may therefore be slightly optimistic relative to the private split.  Nothing in
the model keys on a page-level identity (there is no page-id, filename or row-index
feature), and only 497 of 13782 distinct test line texts also occur in train
(3.6%, and those are overwhelmingly single-token strings such as the ideographic
full stop), so text memorisation is not a route either.

--------------------------------------------------------------------------------
5. LEAKAGE STATEMENT
--------------------------------------------------------------------------------
Every transform, statistic, vocabulary and parameter in this script is fit on the
TRAIN split only; test rows are used for inference only.  Specifically:

  * There is no scaler, imputer, encoder, TF-IDF/count vectoriser, PCA/SVD/NMF,
    clustering step or frequency table anywhere in the pipeline.  Line and block
    features are deterministic functions of a single page's own record, so there
    is nothing to fit.
  * The tokenizer is either the pretrained WordPiece vocabulary (no data fitting
    at all) or, in the fallback path, a character vocabulary built from the TRAIN
    line texts only (`CharTokenizer(... for k in range(len(tr_pages)) ...)`).
  * There is no pd.concat / merge / append of train and test anywhere.
  * The model contains no BatchNorm; only LayerNorm, which normalises within a
    single sample and uses no batch statistics.  Dropout is disabled at eval time.
  * Data-flow taint trace: te_cases -> te_pages -> te_cands / te_toks ->
    loaders["test"] -> collect_scores -> ts[i] = [u, v, w, ...] -> decode_from_scores(...) ->
    ops_string -> seqs[i] -> write_sub.  Every step after the raw case is
    per-page: decode_from_scores reads only page i's own score vectors and the
    offsets, and ops_string reads only page i's own observed edges.  No argmax
    count, bincount, mean, std, quantile, sort or normalisation is ever taken
    across test rows.
  * The decode offsets are searched on the TRAIN holdout, never on test, and never
    against any property of the test predictions.
  * The only test-derived aggregate anywhere is the final completeness assertion
    (row count / duplicate-id check on the file we just wrote).  It prints a flag
    and cannot change any prediction.
  * No pseudo-labelling, no self-training, no test-time adaptation, no calibration
    of outputs to a test distribution.

--------------------------------------------------------------------------------
6. HARDCODING STATEMENT
--------------------------------------------------------------------------------
No discovered generation pattern is hardcoded, and every constant that influences
an output is either a learned parameter or searched in-script on a train holdout.

Enumeration of everything that could look like a rule:
  * `build_candidates` / `cand_struct` enumerate the four corruption families that
    PROBLEM.md itself describes ("swap adjacent lines within a paragraph; swap
    adjacent paragraph regions; exchange visually close lines between two regions;
    combine a cross-region line exchange with a paragraph-order swap").  This is
    the hypothesis space of the structured predictor, exactly like the label set of
    a CRF or the lattice of a beam search.  It ranks nothing and decides nothing:
    all four families and all their parameters are scored by the trained model and
    resolved by one argmax over a globally-normalised energy.
  * `ops_string` is pure graph algebra (symmetric difference + the canonical sort
    the submission format demands).  It formats the model's answer; it does not
    choose it.
  * The four decode offsets are found by in-script coordinate ascent on a train
    holdout against the exact PROBLEM.md metric (`search_offsets`).  Nothing was
    tuned offline and pasted in.
  * `mix` (energy weights) and `bias` (per-family) are nn.Parameters learned by
    backpropagation.  The relative-position and same-block tables are
    nn.Embeddings, i.e. learned from data, indexed by an input feature - the same
    device as a transformer's relative-position bias.
  * The remaining literals are ordinary architecture / optimiser hyperparameters
    (width 256, 3 layers, lr, weight decay, 352-token budget, loss weights, the
    tanh score bound, the relative-position clamp).  None of them maps an input to
    an output class or asserts anything about how the data was generated.
  * There is NO phrase->label dictionary, NO if-chain over text or node ids, NO
    template instantiation, NO frequency/n-gram table, NO regex-driven mapping and
    NO index arithmetic that encodes the generator (in particular nothing reads a
    node's numeric suffix, and node indices are assigned by traversing the observed
    graph).
  * The only literal candidate emitted without the model is the crash-safety
    placeholder written before training starts (a single adjacent-block swap).  It
    exists so a schema-valid file always exists, and it is overwritten by the
    model's predictions for every row the model successfully scores.

Strip-the-ML test (guidebook 4.3), actually measured, not asserted:
  With all trained models removed the pipeline produces "an arbitrary candidate
  from the enumeration", scoring 0.0891 (exact 3.3%) with zeroed scores and 0.1430
  (exact 1.0%) with random scores on the training rows - i.e. essentially the
  chance level of picking one of several hundred candidates.  The trained model is
  what produces the answer; the enumeration, the symmetric difference and the
  canonical sort only express it.

--------------------------------------------------------------------------------
7. WHAT WORKED / WHAT DID NOT
--------------------------------------------------------------------------------
Worked
  * Reading the whole page's text as ONE encoder sequence instead of encoding each
    line independently.  A bi-encoder over frozen mBERT line embeddings plateaued
    around 0.28 and could not learn the within-paragraph swap family at all
    (family A stuck at ~0.05); the page cross-encoder learns it.
  * Modelling the intended graph densely (p(next line|line), p(next block|block),
    p(block|line)) instead of only classifying "which corruption".  This turns one
    label per page into O(L^2) supervised decisions per page and was the single
    largest improvement.
  * The exact factorised partition function.  Because D = C x B factorises, the
    global normalisation over all candidates is closed-form, so the training
    objective is literally the decoding objective.
  * Learned relative-position / same-block bias tables: next-line NLL after one
    epoch dropped from 2.54 to 1.07 and every family improved.
  * Exposing per-line head/tail token vectors to the junction scorers - the
    "does line j continue line i" signal lives in the last tokens of i and the
    first tokens of j, and a single pooled line vector blurs it.
  * Making the family bias page-conditional instead of one global scalar per family.
    The diagnostic above showed the loss was in family ROUTING, not in finding the
    parameter inside a family; feeding the page CLS vector and the candidate counts
    into a small zero-initialised head moved family B from 0.396 to 0.609 page score
    at unchanged within-family accuracy.
  * Calibration-aware epoch selection.  Un-offset holdout scores swing a lot
    between families (e.g. B 0.44 <-> 0.74 across consecutive epochs) purely
    because the four families compete in one argmax; scoring each epoch after a
    quick offset search removes that noise from model selection.

Did not work / rejected
  * A four-way "corruption type" softmax head on the CLS token: it never beat
    chance (~0.25) and made the decode worse than letting the type fall out of the
    joint energy argmax.  Removed.
  * Sorting lines by bounding box.  The boxes are quantised to 1/16 of the page
    and perturbed by several line heights: across the intended graphs only 49.8% of
    consecutive line pairs have strictly increasing y, 13.1% are tied by the
    quantisation and 37.1% actually decrease.  A local geometric sort is worse than
    a coin flip, so geometry is kept only as a weak global prior.
  * Using the page image - investigated properly and rejected on measurements, not
    on a hunch.  Three things were established from train data alone, by matching
    detected text rows to the intended reading order on pages where the row count
    equals the line count:
      (i)  the intended reading order IS exactly the visual top-to-bottom row order
           (1.000 agreement on all 173 such pages) - so the image really does contain
           the answer;
      (ii) the bbox perturbation is enormous: observed y-centre minus true y has
           std 0.197 with median |error| 0.136, and only 24% of lines land within one
           1/16 grid cell.  That is why geometry cannot order lines locally;
      (iii) but the image cannot be cashed in cheaply.  A horizontal ink projection
           finds rows exactly for single-column pages yet over-segments multi-column
           ja/zh pages ~4x (only 173/900 pages give a clean row count), and the cue
           that would let a model match a line to a row - row ink width vs line
           character count - correlates only 0.27 on average within a page (above 0.8
           on just 12% of pages).  Exploiting the image therefore needs a real text
           detector, which does not fit the budget with any confidence.  Left on the
           table, but with the price tag now measured rather than guessed.
  * Data augmentation by applying additional corruptions to the reconstructed
    intended graphs.  It would have multiplied the supervision for family A
    considerably, but guidebook 4.2.6 forbids training on self-generated data and
    the conservative reading (PROMPT.md: "if genuinely unsure ... implement the
    more conservative version") is that synthesising new (input, label) pairs is
    exactly that.  Not used.
  * Two half-trained models ensembled: replaced by one model trained on the whole
    budget.
  * A 4-line-window head for family A (prev, x, y, next instead of just x, y).
    Well motivated - the decision genuinely depends on the neighbouring lines - but
    it moved family A from 0.033 to 0.039 on a 1500-page A/B and the full-data run
    with it scored 0.3992 against 0.4195 without.  Reverted.
  * Freezing the bottom two encoder layers plus dropout 0.15, tried because both A/B
    arms peaked at epoch 2 of 5 on 1500 pages.  It was worse at matched epochs
    (0.254 / 0.321 at epochs 0-1 vs 0.274 / 0.361), so the overfitting is better
    handled by best-epoch selection than by shrinking the trainable set.

Known limitations
  * Family A (adjacent line swap inside a paragraph) remains the weakest: it needs
    a fine-grained judgement of which of two orderings of the same two lines reads
    correctly, and there are only ~884 directly-labelled examples of it.
  * The holdout is language-stratified but not layout-group-aware, because the
    public package does not expose the layout-family id.
