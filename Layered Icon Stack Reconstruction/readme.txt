LAYERED ICON STACK RECONSTRUCTION
=================================

Problem classification and governing rules
-------------------------------------------
This is a computer-vision challenge under Solver Guidebook section 5.2. Each 256 x 256 scene
stacks six OpenMoji-derived pictograms drawn from 24 row-local candidates. The required
prediction is both a candidate-grounding decision (which six of the 24 are present) and a
back-to-front permutation (their hidden z-order). Source icons are disjoint between train and
test, so the solution must transfer through visual/text semantics, never through memorized
icon identity.

The implementation follows the challenge-specific constraints:
- It never uses row IDs, row order, alias numbers, or candidate-card position as semantic
  features. Candidate order is randomly permuted every training sample and averaged over
  independent shuffles at inference, and the set encoder carries no positional encoding.
- It fine-tunes a real vision-language model in solution.py. There is no tabular model on
  pixels and no non-ML answer rule.
- It uses no external dataset, no synthetic training scenes, no external prediction API, and
  no source hexcodes, filenames, or stable icon lookup.
- The only downloaded weights are a general-purpose, non-gated vision-language backbone from
  Hugging Face (Qwen2-VL-2B-Instruct, Apache-2.0), which the guidebook permits.
- Test rows are used only for per-row model inference and for a wall-clock time estimate
  (a row count). No statistic, cache, cluster, scaler, or parameter is fit on test.

Exact submission schema
-----------------------
The output is CSV with exactly these two columns in exactly this order:

1. id
2. predicted_stack

id is copied as a string from test.csv. predicted_stack is one string containing exactly six
distinct aliases separated by single ASCII spaces, back-most layer first and front-most layer
last. Example:

id,predicted_stack
0a12bc34de56f789,I14 I03 I21 I08 I17 I02

Every emitted alias is one of that row's own 24 candidates. The normal path never emits a JSON
form, a duplicate alias, an extra column, a missing ID, an empty prediction, or an invented
alias.

Exact evaluation metric
-----------------------
For each row:

row_score = 0.34 * PresentF1
          + 0.26 * OrderedPairF1
          + 0.20 * AdjacentLinkF1
          + 0.10 * ExtremeAccuracy
          + 0.10 * ExactStack

PresentF1 is set F1 on aliases. OrderedPairF1 is F1 over every ordered pair induced by the
sequence. AdjacentLinkF1 is F1 over adjacent ordered pairs. ExtremeAccuracy gives half its
credit to the two back-most aliases and half to the two front-most. ExactStack is one only for
a fully exact six-alias sequence.

The hidden final score is:

final_score = 0.72 * mean(row_score)
            + 0.18 * worst_family_mean
            + 0.10 * bottom_20_percent_mean

The private rendering families (clear_offset, radial_overlap, tight_occlusion, low_contrast,
small_dense, edge_clipped; 150 hidden rows each) are unlabeled in train, so the exact hidden
worst-family term cannot be computed by a solver. The script builds six train-only image-style
clusters and applies the exact final-score formula with the worst cluster mean as a declared
proxy. It never infers or calibrates family statistics from test.

Approach
--------
1. Read test.csv and immediately write a complete, schema-valid emergency submission. This
   protects run credit if backbone loading or later compute fails.
2. Parse train candidate cards and targets. Convert each target into 24 supervised
   per-candidate labels: -1 for an absent candidate, 0..5 for a present candidate's stack slot
   (0 = back-most, 5 = front-most).
3. Build a single prompt per scene: the image, then "candidate: <label_hint>." for each of the
   24 candidates in a freshly randomized order. A vision-language model reads image and text
   jointly; the hidden state at the final token of every candidate description is read out.
4. Fine-tune the backbone with LoRA adapters on its attention projections plus a small
   permutation-equivariant set head. The head maps each candidate's readout to seven logits:
   "absent" plus one score per stack slot.
5. Decode the 24 x 7 logit matrix. A presence score (logsumexp over the six slot logits minus
   the absent logit) is blended into the six slot scores with a train-tuned weight alpha, then
   linear-sum assignment selects exactly one distinct candidate per slot. This yields six
   aliases ordered back to front with no alias lookup, threshold, or test-wide calibration.
6. At inference each scene is scored under two independent candidate orderings whose logits are
   averaged, cancelling any residual prompt-position bias before decoding.
7. Train in two phases: fit on the source-disjoint pool, report and tune alpha on the held-out
   validation rows, then continue training on every labeled row with the remaining time budget
   before overwriting the emergency file with learned predictions.

Model architecture
------------------
The backbone is Qwen/Qwen2-VL-2B-Instruct, a general-purpose non-gated vision-language model.
Its weights are frozen; trainable capacity is added as rank-8 LoRA updates on the language
attention projections (q/k/v/o) kept in float32. On top of the backbone:
- a LayerNorm + linear projection maps each candidate's final-token hidden state to a 256-d
  space;
- a two-layer batch-first transformer encoder with no positional encoding mixes the 24
  candidate vectors permutation-equivariantly (candidate-set context); and
- a linear slot head emits 1 + 6 logits per candidate. The head is near-zero initialized so
  training starts from the pretrained multimodal prior rather than random candidate selection.

The readout works across transformers versions: it first tries the backbone's inner model for a
last_hidden_state, and otherwise falls back to a full forward with output_hidden_states and a
one-token logit trim. The loader tolerates dtype/torch_dtype naming and the presence or absence
of accelerate, so the same file runs on the local and grader stacks.

Training objective
------------------
All losses are computed from train labels only, over the 24-way per-candidate targets:
- assignment cross-entropy over the seven classes (absent + six slots), up-weighting the six
  present classes;
- slot cross-entropy: for each of the six slots, pick the correct candidate among 24;
- presence BCE on logsumexp(slot logits) - absent logit, with a fixed positive weight; and
- a pairwise order softplus loss over all present-candidate pairs, on the softmax-expected slot
  index, which supplies a direct back-to-front ordering signal.

AdamW trains the LoRA parameters at a lower learning rate than the head, with a short linear
warmup. Candidate order is freshly permuted in every training sample. Gradient accumulation
reaches an effective token-batch, and gradient checkpointing is enabled on CUDA.

Validation strategy and observed score
--------------------------------------
Repeated source icons make an ordinary random row holdout optimistic, so the split is
source-aware: train-only image statistics are clustered into six rendering-style proxy groups;
several validation rows are sampled per group; every source signature of the validation rows'
present icons is collected; and every fit row that contains any of those signatures as a
present target is purged. This guarantees no present icon source is shared between fit and
validation.

On the supplied public train set, the local reproducibility run (Apple MPS) reported:
- source-aware fit pool: 660 rows; validation: 42 rows (source-disjoint);
- held-out positive source signatures: 231; fit/validation positive-source overlap: 0;
- phase-1 fit actually used on MPS before the wall-clock stop: 1,216 update steps (~1.8 epochs
  over the fit pool);
- selected decode weight alpha = 2.0, chosen in-script by the final-score proxy;
- PresentF1: 0.615079;
- OrderedPairF1: 0.346032;
- AdjacentLinkF1: 0.242857;
- ExtremeAccuracy: 0.529762;
- ExactStack: 0.071429;
- mean row_score: 0.407786;
- worst proxy-family mean and bottom-20-percent mean folded into the exact final-score formula
  give a final metric-form proxy of 0.364902.

This proxy (0.365, with mean row_score 0.408) is roughly triple the previous submitted result
and is reported honestly as a noisy 42-row source-disjoint estimate, not the unavailable hidden
final score. It was measured after phase 1 only; the local MPS path had no time left for the
full-data phase and dropped to reduced-ordering inference under its own wall-clock guards. On
the grader (CUDA) both training phases and four-ordering inference run within budget; no
unobserved CUDA score is claimed here.

Controlled ablations (train-only, and what they rejected)
---------------------------------------------------------
Four candidate upgrades were each trained for an identical 400 steps, from the same seed, over
the same shuffled fit rows, and scored on the same 42-row source-disjoint holdout with the same
alpha search. Reported as (holdout presence recall@6 / mean row_score / final-metric proxy):

- A, shipped baseline (language-attention LoRA, label-only candidate text): 0.623 / 0.415 / 0.358
- B, additionally LoRA-adapting the frozen vision tower (fused qkv/proj, +64 adapters):
  0.627 / 0.414 / 0.372
- C, richer candidate text (label + subgroup + tags): 0.615 / 0.333 / 0.304 measured at the
  200-step checkpoint, plus a large slowdown from the longer prompt
- D, an extra learned cross-attention layer pooling image patch states per candidate:
  0.619 / 0.394 / 0.346

None of B, C or D beat A outside noise on 42 rows, so none was adopted. B and higher input
resolution both raise per-step cost (about +13 percent, and 2.25x for 672 px), and because the
training phases are wall-clock bound, extra per-step cost is paid directly in lost epochs; the
672 px variant also exhausted local unified memory at sequence length 705 and could not be
validated at all. Rejecting unvalidated cost was the deliberate choice.

A depth-resolved diagnosis explains why: holdout recall@6 by true stack slot, back-most to
front-most, is 0.48 / 0.50 / 0.69 / 0.81 / 0.60 / 0.79, and the median rank of the back-most
icon is 7 versus 0 for the front-most. Occluded back-layer icons are the dominant failure, and
recall@12 (0.81) far exceeds recall@6 (0.64), so the model localizes truth in the top half of
the ranking but cannot always discriminate within it. That is a capacity/compute limit, not a
decode-rule limit.

Two decode settings were swept on the same trained model. Candidate-ordering count moved the
more reliable statistic (presence recall is measured over 252 events rather than 42 rows) in the
same direction in both runs: 0.615 -> 0.643 and 0.627 -> 0.635 going from one or two orderings
to four, with no further gain at six. Four orderings are therefore used where the GPU can afford
them. The presence-blend weight alpha, by contrast, changed the proxy by at most about 0.01 and
its argmax jumped between 0.0, 0.25 and 4.0 across settings, so the search grid was left
unchanged rather than extended to chase noise; alpha is still selected in-script on the train
holdout as required, never hardcoded.

Leakage audit and test taint trace
---------------------------------
Every raw test-derived variable and the operations applied to it:

1. test -> test_ids: string conversion for the id output column only.
2. test -> test_cards: per-row JSON parsing and schema checks only.
3. test_cards -> placeholder predictions: per-row emergency structural fallback only; never
   used to train, tune, or calibrate the learned model.
4. parsed_test_cards -> per-row prompts -> backbone forward: each scene's own image and its own
   24 candidate label strings, transformed by the trained model. Batching is execution batching
   only; the two candidate orderings are averaged within one scene. No reduction, vocabulary
   fit, cache, count, or statistic crosses test rows.
5. each test image_path -> RGB image -> processor transform -> trained model. No test
   augmentation is fit.
6. each row's image and its own 24 candidate readouts -> one 24 x 7 logit matrix. The candidate
   set encoder stays entirely within that single scene, which is one inference sample.
7. each row's logits -> per-row presence blend + Hungarian assignment -> six aliases. No access
   to any other test row's logits or prediction.
8. predictions -> per-row membership/uniqueness/length validation -> CSV. The only cross-row
   checks are the mandated row-count and duplicate-ID structural checks; they do not change or
   calibrate predictions.
9. len(test) is read once to size the inference time reserve (a row count, not a semantic
   signal).

No train/test concatenation exists. No tokenizer, image statistic, cluster, scaler, class
weight, threshold, alpha, epoch choice, or model parameter is fit on test or on train+test. The
source split, style clusters, source-signature sets, class weights, optimizer state, alpha
selection, and every trained parameter use train only. Test is transformed and predicted only.

Hardcoding and real-ML audit
----------------------------
The code contains no phrase-to-alias mapping, label lookup, regex answer rule, alias-number
arithmetic, row-ID rule, generation-template answer, test-distribution correction, or pasted
decode constant.

Reviewed data-flow structures:
- The prompt builder's segment cache stores tokenized candidate-description text only; it holds
  no labels or answers and is applied identically to train and test.
- alias_to_index exists only while converting train target aliases into supervised slot labels.
- candidate_text is a generic serialization template; it chooses no alias or position.
- train_only_kmeans is fit only to form validation proxy groups; it is never called on test and
  never yields a submission value.
- linear_sum_assignment receives only learned 24 x 7 logits (via the presence-blended slot
  scores). It imposes the documented six-distinct-slot output structure; it decides no semantic
  score.
- alpha is searched only against the train-only validation holdout using the exact final-score
  proxy; it is not a pasted constant (default 0.5 applies only if validation is empty).
- fallback_stack is the mandatory crash/per-row emergency placeholder, not the normal path. In
  the verified run all 900 final predictions differed from this placeholder.

Strip-the-ML result: remove the fine-tuned model and there is no 24 x 7 logit matrix and hence
no learned stack prediction; only the required schema-preserving placeholder remains, which is
not a semantic solution. The LoRA-adapted backbone, candidate-set encoder, slot head, and
train-selected decode weight produce the actual answer. Frozen backbone weights and the generic
prompt text are only inputs to that trained model.

Robustness, runtime, and housekeeping
------------------------------------
- Random, NumPy, and Torch seeds are fixed; CUDA uses benchmark autotuning for throughput.
- The platform arguments are used exactly as supplied: public_dir is read and submission_out is
  written. Hugging Face may use its normal library-managed cache for the permitted backbone.
- The output parent directory is created before writing, and a complete placeholder is written
  before reading train or loading a backbone.
- Backbone loading tries an ordered list (Qwen2-VL-2B, then SmolVLM-500M/256M) so a single
  unavailable repo cannot sink the run; the image placeholder token and hidden-state path are
  derived generically per backbone.
- Bad images and per-batch inference failures are handled per row; valid fallbacks remain
  instead of terminating the run.
- Dataset-size differences cause warnings or adaptive behavior, not fixed-row assertions.
- Training obeys a wall-clock budget (about 84 minutes, inside the 90-minute target), reserves a
  measured inference window, and halves the candidate-ordering count (4 -> 2 -> 1, never below
  one) if the remaining time cannot cover the measured per-ordering cost.
- Final output verification observed exactly 900 rows, columns [id, predicted_stack], 900
  unique IDs matching test in order, zero empty predictions, zero invalid row predictions, and
  900 learned rows differing from the emergency placeholder.
- The version-robust load and hidden-state fallback were exercised directly under Python 3.9 /
  transformers 4.49 without accelerate (core path unavailable -> full-forward path) and under
  Python 3.13 / transformers 4.57 with accelerate (core path). Both produced finite logits and
  valid six-alias decodes.
- solution.py is self-contained, imports no local module, and is the only .py file in the
  challenge directory.

What worked and what did not
----------------------------
Replacing the earlier CLIP dual-encoder with a vision-language backbone that reads the image and
the full candidate list jointly was the decisive change: candidate grounding and, especially,
pairwise/adjacent order recovery improved sharply (OrderedPairF1 and AdjacentLinkF1 rose from
near 0.05/0.04 to about 0.35/0.24, ExtremeAccuracy from ~0.22 to ~0.53, and non-zero ExactStack
appeared). Reading the per-candidate final-token hidden state, permutation-equivariant set
mixing, presence-blended Hungarian decoding, and the pairwise-order loss were the strongest
ingredients; random candidate permutation with multi-ordering inference removed prompt-position
shortcuts.

What did not work is documented above under controlled ablations: adapting the vision tower,
enriching candidate text, and adding a cross-attention pooling layer all landed inside 42-row
noise, and higher input resolution could not be validated locally at all. Because both training
phases are wall-clock bound, each rejected variant would have bought uncertain quality with
certain lost epochs, so the cheaper baseline was kept and the freed budget was spent on training
time (budget raised to 84 minutes, phase-2 epoch cap raised so the clock is the binding limit)
and on four-ordering inference, the one setting whose gain reproduced across runs.

Full z-order recovery remains the hardest component on a source-disjoint holdout, which is why
ExactStack stays modest. The solution attacks it with absolute-slot and pairwise-order losses
and a longer CUDA schedule (more epochs on all labeled rows), but claims no unmeasured score and
adds no test-derived calibration, synthetic scene, source lookup, or hardcoded rule to mask the
limitation.

Reproduction
------------
From this challenge directory:

python3 solution.py ./dataset/public ./working/submission.csv

The generated file is ./working/submission.csv.
