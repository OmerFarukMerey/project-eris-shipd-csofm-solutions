Cross-Locale Repair-Window Propagation
======================================

Approach
--------
The solution trains an ensemble of two small transformer models over the 16
normalized windows. Both models are conditioned on the source, anchor draft,
anchor repair, anchor repair mask, target draft, locale pair, and MQM family.
The first model uses alignment features and pairwise 16x16 sketch-similarity
maps. The second also learns 32-dimensional embeddings for the 512 privacy
buckets. All weights are initialized randomly and trained inside solution.py;
there are no pretrained weights, external data, cached artifacts, retrieval,
or hardcoded test predictions.

Before final training, the script creates a train-only route holdout containing
one directed route for every anchor locale and every target locale. The holdout
is never used to fit preprocessing. It selects the epoch counts, neural
ensemble weight, and expected-utility decoder temperature. The selected
configuration is then retrained from scratch on all training rows.

Model architecture and loss
---------------------------
Each model embeds:
- 83 per-window numeric features into width 128;
- five 16x16 alignment/similarity matrices, viewed by row and column;
- anchor locale, target locale, and MQM family via learned embeddings;
- normalized window position via a learned embedding.

The bucket model additionally averages learned embeddings of the four bucket
codes in each of the four sketches and projects the concatenated sketch state
to width 128. Three 4-head transformer encoder layers process the 16-window
sequence. A window head predicts 16 repair logits. A second head predicts the
target repair count in classes 1 through 10, with larger counts pooled into the
last class. Training minimizes locale-balanced binary cross-entropy plus a
soft Dice term and a count cross-entropy term. AdamW, gradient clipping, and a
fixed-seed one-cycle learning-rate schedule are used.

Feature engineering
-------------------
All sketch parsing preserves duplicate buckets. Exact bucket equality produces
window-to-window source/target, anchor/target, repaired/target, source/anchor,
and draft/repaired similarity maps. Features include:
- the anchor mask, shifted masks, Gaussian propagation, and edit distance;
- direct and source-composed propagation into target windows;
- best and expected alignment distance to an edited anchor window;
- overlap with removed, added, source, and anchor-draft edit buckets;
- diagonal, neighboring-window, push-alignment, rank, and within-row z-score
  channels;
- target self-similarity and duplicate-bucket indicators.

A label-free bucket reliability transform is fit on training sketches only. It
estimates smoothed lift for same-window source/anchor-to-target bucket matches
and supplies weighted alignment features. Applying these learned weights to a
test row is a transform of that row, not a statistic over test rows.

Decoding
--------
Window probabilities are ensembled using the train-only validation-selected
weight. For each row, the decoder samples plausible repair counts from the
count head and samples conditional masks with Gumbel top-k draws over window
log-odds. It evaluates candidate masks with the exact RWU matching rule,
including 0.35 adjacent-window credit and budget fidelity, then performs local
bit-flip refinement. The random draws are reset to the same fixed seed for
every row, so decoding is deterministic and depends only on that row's model
outputs.

Validation
----------
The final local end-to-end run used a 4,318-row route-disjoint holdout selected
from train.csv: one held-out directed route per anchor locale and per target
locale. Every code weight, normalization statistic, category vocabulary, and
model parameter for this validation was fit on the remaining 17,270 rows.
Locale-balanced holdout RWU was 0.456770. The script selected 7 epochs for each
transformer, bucket-model ensemble weight 0.7, and decoder temperature 1.2.
It then retrained both models on all 21,588 training rows.

A separate development target-only ablation used only position, target-draft
self-similarity/duplicate signals, and target locale in five directed-route
group folds. Its locale-balanced top-k RWU was 0.391057. Copying the anchor
mask scored 0.365481, and predicting all windows scored 0.299761. The full
anchor-conditioned model therefore provides material signal beyond the target
position prior and does not discard the anchor fields.

What worked and what did not
----------------------------
Direct anchor copying was too weak because related target edits are not exact
copies. Independent 0.5 thresholding also performed poorly because it ignored
repair-count uncertainty and the RWU review-budget term. Alignment propagation,
source-mediated alignment, train-fitted collision reliability, raw bucket
embeddings, count multitask learning, and expected-RWU decoding all improved
validation. A tree-model branch was investigated but removed: on this data it
added insufficient validation value to justify its substantially higher local
runtime. The final neural-only run completed locally in 1,436.4 seconds.

Leakage and compliance audit
----------------------------
- Validation is built from train.csv only and is disjoint by complete directed
  locale route. Validation preprocessing is fit on the fit partition only.
- Final code weights, normalization statistics, category maps, epoch choices,
  ensemble weight, decoder temperature, and both neural models use training
  data only. Epoch/ensemble/temperature selection uses only the train-derived
  route holdout.
- test.csv is loaded only after final preprocessing statistics and both final
  models have been fit. Test usage is limited to build_inputs(test,
  train-fitted code weights), category mapping with a train-fitted vocabulary,
  predict_nn for the two models, per-row decoding, and copying test IDs to the
  output.
- No count, vocabulary, frequency, normalization statistic, threshold,
  embedding, cluster, or model is fit or calibrated on test rows.
- There is no train+test concatenation, merge, pseudo-labeling, test-time
  adaptation, or cross-test-row aggregation.
- Seeds for Python, NumPy, and PyTorch are fixed at 42.
- The script reads only public_dir/train.csv and public_dir/test.csv. It writes
  only submission_out after creating its parent directory.
- solution.py is self-contained and imports no local files.

Output verification
-------------------
The required command

  python3 solution.py ./dataset/public ./working/submission.csv

created 3,695 predictions. The generated file has columns id and
target_repair_windows in that order, exactly one unique row per test ID, and
every mask matches [01]{16}. The observed predicted repair count ranged from 1
to 10 with mean 4.197.
