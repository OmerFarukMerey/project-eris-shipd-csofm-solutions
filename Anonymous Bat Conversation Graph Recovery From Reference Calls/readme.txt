Anonymous Bat Conversation Graph Recovery From Reference Calls
===============================================================

Problem classification and rules
--------------------------------
Guidebook domain: Biology/Chemistry/Other, section 5.6. This is an audio bioacoustics
matching and structured-prediction problem. PROBLEM.md explicitly permits trained models
over bat-call descriptors, metric-learning-style reference comparison, and compact audio
models, and explicitly requires CPU execution. The final solution trains real audio models
from scratch on the provided training data; it does not use a frozen/pretrained model.

Challenge-specific restrictions followed:
- CPU only; no GPU, internet, runtime download, hosted API, external data, upstream corpus,
  private file, or pretrained/fine-tuned weight is used.
- Anonymous node identity is row-local. The model never treats Bat_A/B/C/D as a global bat
  identity. Node names are candidate output symbols only; candidate features come from the
  row's supplied references.
- An audio_path is used only to open its public WAV. Path text, call/reference IDs, source
  names, hidden/original bat IDs, treatment, date, channel, archive order, file size,
  timestamps, gallery_position, and row order are not predictive features.
- Only the native context labels loaded from taxonomy.json can be emitted.
- Graphs are validated as exact aggregations of model-produced call records.

Exact submission schema
-----------------------
The CSV has exactly one row per test episode and exactly these columns in this order:

1. episode_id
2. call_predictions_json
3. graph_json
4. confidence

`episode_id` is the exact test episode ID.

`call_predictions_json` is a JSON-encoded string containing a list with exactly one object
for every gallery call in that episode. Every object has exactly these five fields:

{"call_id":"<exact test call ID>","caller":"<episode node>","addressee":"<different episode node or UNKNOWN>","context":"<taxonomy label>","confidence":<number from 0 through 1>}

`caller` is one node from that row's node_set_json. `addressee` is a different episode node
or the exact string `UNKNOWN`. `context` is one of the 13 strings in taxonomy.json.
Confidence is finite and in [0,1]. JSON is emitted with json.dumps using compact separators;
pandas performs the necessary CSV quoting.

`graph_json` is a JSON-encoded string whose root has exactly the `edges` field:

{"edges":[{"source":"Bat_A","target":"Bat_B","count":2,"contexts":{"GENERAL":1,"SLEEPING":1}}]}

Each edge source and target are distinct episode nodes. `count` is the number of submitted
calls with that directed pair. `contexts` maps only submitted taxonomy contexts to positive
integer counts for that edge. Duplicate edges and zero-count edges are never emitted. Calls
with addressee UNKNOWN do not create an edge.

The row `confidence` is a finite learned/calibrated float in [0,1].

Exact evaluation metric
-----------------------
For each episode:

caller_accuracy    = mean exact caller match
addressee_accuracy = mean exact addressee match, including UNKNOWN
context_accuracy   = mean exact native-context match
call_core = 0.45*caller_accuracy + 0.25*addressee_accuracy + 0.30*context_accuracy

For any count maps P and T:

count_f1 = 2*overlap/(predicted_total + true_total)
overlap  = sum_k min(P[k], T[k])

graph_score = 0.65*directed_edge_count_f1 + 0.35*edge_context_count_f1
consistency = 0.65*edge_count_f1(graph,calls)
            + 0.35*edge_context_f1(graph,calls)
core = 0.80*call_core + 0.15*graph_score + 0.05*consistency

call_calibration = mean max(0, 1 - abs(call_confidence - per_call_correctness))
row_calibration  = max(0, 1 - abs(row_confidence - core))
episode_score = core*(0.94 + 0.04*row_calibration + 0.02*call_calibration)

final = 0.78*mean(episode_score)
      + 0.22*mean(worst_group_mean for each hidden axis)

The hidden axes are episode node-count family, presence of native UNKNOWN addressees, and
dominant source-context family. Hidden group assignments are not available to the script.
For metric-aware train-only selection, solution.py uses the corresponding observable
training episode groups: node count, labeled UNKNOWN presence, and dominant labeled context.
It never estimates a group or class balance from test predictions.

Approach
--------
1. Stateless audio descriptors
   Each mono 100 kHz WAV is DC-centered and transformed with a 2048-sample Hann-window
   spectrum. Two complementary representations are extracted in one WAV read:
   - a 480-dimensional full-Nyquist descriptor with 64 normalized log-power bands,
     distribution quantiles, cepstral summaries, spectral centroid/bandwidth/flatness,
     waveform summaries, and temporal energy modulation;
   - a 1374-dimensional descriptor with denser allocation below 25 kHz, robust summaries
     over all frames and within the highest-energy 10/25/50 percent of that clip, cepstra,
     rolloffs, peak frequencies, and modulation.
   Per-clip rank/quantile operations use only that clip. They are fixed signal transforms,
   not fitted statistics.

2. Learned row-local caller matcher
   For every gallery call and candidate node, symmetric distance features compare the gallery
   descriptor with that node's supplied references. Three ExtraTrees matchers form one
   candidate, and a LightGBM LambdaRank model directly learns the within-call candidate
   ranking. Multiple independently initialized PyTorch metric-network configurations are
   cross-fitted from scratch: each uses a shared MLP encoder, averages the two row-local
   reference embeddings, and learns gallery/reference cosine scores with cross-entropy over
   the episode nodes. Feature normalization for each metric fold is fitted only on that
   fold's training acoustic families. OOF search learns the model/configuration/temperature
   blend. Node symbols are schema-local candidate positions, never persistent bat identities.

3. Learned context and unknown-target models
   Two ExtraTrees classifiers and a class-balanced LightGBM classifier predict native context
   from the dense descriptor. A separate balanced ExtraTrees model learns whether an
   addressee is unknown from gallery acoustics, node count, and cross-fitted context
   probabilities. Model leaf size is selected by grouped OOF AUC. No fixed context or UNKNOWN
   decision rule is used.

4. Learned joint social-call model
   Every valid (caller, addressee) hypothesis is represented. Self-target hypotheses are
   excluded because graph edges must have distinct nodes; UNKNOWN is an explicit candidate.
   Features include gallery acoustics, candidate caller and target reference means/spreads,
   gallery-reference and caller-target contrasts, cross-fitted caller/context probabilities,
   and the trained unknown probability. Balanced ExtraTrees classifiers and LightGBM
   LambdaRank models directly rank complete social-call hypotheses. Grouped OOF search selects
   ranker temperatures, the model blend, and the joint-vs-caller probability blend. The joint
   weight is constrained above zero, so a trained joint model always produces the addressee.

5. Metric-aware context and confidence calibration
   A generic simplex search blends trained context models. Generic coordinate search gives
   every taxonomy class the same candidate log-bias grid and selects values against the real
   episode metric on grouped OOF predictions. Call confidence is produced by a trained
   ExtraTrees regressor over row-local model probabilities/margins. Row confidence is
   produced by a train-selected regressor using only node/reference information already
   present in one test row. All calibration training uses OOF training predictions.

6. Train-learned episode and graph decoder
   Caller identity remains fixed at the strongest call-level posterior. A cross-fitted rate
   regressor predicts each episode's known-addressee fraction from aggregate train-model
   probabilities and supplied node count. A second cross-fitted regressor predicts candidate
   directed-edge counts from row-local acoustic/reference contrasts and posterior summaries.
   In-script OOF search selects the rate shrinkage and edge-head family. Integer edge quotas
   are allocated deterministically, then a Hungarian assignment chooses each call's target
   from its trained social posterior while respecting those quotas. UNKNOWN remains an
   explicit target. The emitted graph is then a lossless count aggregation of the decoded
   known-addressee calls; the validator enforces exact graph/call consistency.

Validation
----------
Strategy: 6-fold GroupKFold over train-only acoustic condition families. Each episode and all
its reference/gallery clips remain wholly in one fold. The family labels come from PCA plus
KMeans on episode reference-acoustic summaries only; no target label enters this grouping.
This deliberately prevents near-duplicate preparation/source conditions from appearing on
both sides of a validation fold. Caller, context, unknown-target, joint, rate, edge-count,
and calibration stacks use OOF predictions. Hyperparameter, ensemble, class-bias,
social/graph decoder, and confidence choices are searched in-script using training OOF
results only. No public-leaderboard or test-derived parameter choice is present.

Observed audio-backed condition-family run before the graph-head extension (fixed seed 1729):
- train-visible final metric proxy, including confidence and worst-family analogues: 0.536809
- mean episode core: 0.550347
- mean episode caller accuracy: 0.470671
- mean episode addressee accuracy: 0.378729
- mean episode context accuracy: 0.747590
- mean episode graph score: 0.504921

An exact train-only OOF ablation of the added decoder, using the current implementation,
raised its matched proxy from 0.532948 to 0.539216 and graph score from 0.493077 to 0.558129;
mean core rose from 0.542729 to 0.547200. This ablation is separate from the run above, so the
two gains are not added. The proxy is not claimed to be the unavailable hidden final score.

Leaderboard diagnosis supplied by the user: the earlier random-episode OOF model scored about
0.41, below a simpler 0.4545 submission despite a 0.552334 local proxy. That inversion is
evidence that episode_id grouping did not isolate prepared/source acoustic families. The
leaderboard values motivated the stricter validation partition; they are not optimization
targets and do not select any weight, threshold, label, or output.

The last audio-backed required command completed in 294.3 seconds and validated 18 episode
rows covering all 249 test call IDs. After the graph extension, its exact helpers were
validated on retained audio-backed train features and a complete structural smoke run again
validated all 249 test call IDs. The local audio files had disappeared before that smoke run,
so its zero-feature fallback metric is deliberately not reported and its output did not
replace the previously scored 0.4545 audio-backed submission.

What worked / what did not
--------------------------
Worked:
- normalized spectral distributions plus active-frame summaries were more stable than raw
  waveform comparison;
- the shared from-scratch metric encoder and LambdaRank caller model made complementary
  errors, while acoustic-family grouping exposed their true out-of-condition error;
- joint ExtraTrees/LambdaRank hypotheses improved caller, addressee, and graph quality;
- grouped OOF stacking prevented in-sample base-model confidence from leaking into the joint
  model;
- the learned episode-rate/edge-count head materially improved graph F1 in its OOF ablation;
- train-searched context and confidence calibration improved the challenge metric without
  using test distributions.

Did not work well enough to retain:
- direct nearest-reference Euclidean/cosine scoring was weaker and would also leave too much
  of the answer to an untrained retrieval rule;
- a compact from-scratch multitask spectrogram CNN fit training pseudo-identities but did not
  improve caller accuracy on held-out episodes;
- reference-augmented pair examples and very high-dimensional relative-rank features reduced
  grouped caller accuracy;
- episode-reference summary features overfit the small number of training episodes for
  UNKNOWN prediction.

Leakage audit: explicit test-taint trace
----------------------------------------
Raw tainted inputs begin at `test` and each test WAV.

- test -> early placeholder: grouping is used only to satisfy the mandatory valid-artifact
  contract. It does not fit, tune, or calibrate a model.
- each test WAV -> test_f1/test_f2: deterministic transform of that WAV only. The path cache
  only memoizes repeated supplied references; it computes no corpus statistic.
- test_f1 -> test_pair_x: each gallery row is compared only with references contained in that
  row's own reference_json.
- test_f2 -> test_context_x -> trained context predict: batched prediction is row-separable;
  no test-row normalization or fitted state occurs.
- test model probabilities -> caller/unknown/joint/context probabilities: every score is
  produced with models already fitted from training data. Candidate normalization is only
  across alternative nodes/targets for the same gallery call.
- probabilities from calls in one test episode -> trained rate/edge regressors -> integer
  target quotas -> Hungarian assignment: this is the one deliberate across-row inference
  operation. It never crosses episode boundaries, and all fitted parameters, rate shrinkage,
  and the selected edge-head family come from train-only OOF data. The assignment changes
  addressees only; it cannot change callers or contexts.
- each call's probabilities/margins -> trained call calibrator -> that call's confidence:
  row-local prediction only.
- one episode row's supplied node/reference fields -> trained row calibrator -> row
  confidence. It does not inspect other test episodes.
- final decoded calls -> graph Counter: task-required lossless count serialization.
- final objects -> JSON strings -> submission.csv: serialization and schema validation only.

There is no train+test concatenation. Train and test feature caches are separate. No test
value is passed to fit. Every classifier, regressor, encoder, blend, class bias, threshold,
quota hyperparameter, or calibration choice is fitted/searched using training data and OOF
predictions only. Test is used for deterministic transform, trained-model predict, and the
documented within-episode constrained assignment. There is no cross-test frequency, mean,
scale, vocabulary, clustering, pseudo-label, class-balance matching, sorting, threshold, or
prediction-distribution calibration.

Hardcoding / real-ML audit
--------------------------
No discovered generation pattern, phrase/ID mapping, source reconstruction, global bat
identity, class answer, template answer, fixed output family, or offline-tuned decode value is
hardcoded. No regex is used. All predictive thresholds, tree leaves, ranker temperatures,
model/seed ensemble weights, context log biases, social blend weights, episode-rate
shrinkage, edge-head family, and confidence mappings are learned or searched in-script on
train-only grouped OOF data. Fixed values for random seed, FFT resolution, neural layer

Dictionaries/lookups/branches that can reach output:
- audio path -> feature dictionaries are memoization caches, not path-to-label lookups;
- node -> reference arrays are constructed from that row's public reference_json and are
  inputs to trained matchers, not asserted identity mappings;
- taxonomy label -> training index is a reversible encoder loaded from taxonomy.json;
- node -> caller probability dictionaries contain trained model outputs;
- candidate construction enumerates schema-valid node pairs plus UNKNOWN, after which trained
  joint and episode/edge models choose the answer;
- class-bias, ensemble, rate, and graph-decoder arrays are results of in-script OOF search,
  not pasted answers;
- quota dictionaries contain predictions from the trained episode and edge regressors;
- Counter dictionaries in make_graph serialize the final model-decoded calls exactly;
- validity `if` branches only reject impossible labels, self-edges, non-finite confidence,
  or malformed audio. They do not overwrite any valid model prediction;
- the mandated early placeholder and exceptional corrupt-row fallback contain only a valid,
  zero-information schema record. They are written before training for crash safety and are
  overwritten after successful model inference. They are not the scoring strategy.

Strip-the-ML test: with every trained classifier/regressor removed, the pipeline cannot
produce a learned caller, addressee, context, episode quota, edge count, or calibrated
confidence and therefore cannot produce a usable social ledger or graph. Only the explicitly
mandated zero-confidence emergency placeholder/schema serializer remains. Retrieval
distances, references, feature caches, candidate enumeration, assignment machinery, and
graph counting do not independently produce answers. The trained models produce the call
answer; the final graph code losslessly summarizes it.

Housekeeping / robustness audit
-------------------------------
- Random seeds are fixed for NumPy, PyTorch, ExtraTrees, LightGBM, and KFold.
- solution.py reads only files under public_dir and writes only submission_out.
- It imports no local module. solution.py is the only .py file in this challenge directory.
- A schema-valid placeholder is written immediately after reading test/taxonomy and before
  audio extraction or training.
- Individual bad audio files become finite zero descriptors so one noisy row cannot abort the
  run. Final per-call validity branches preserve valid model predictions and provide a valid
  fallback only for corrupt values.
- Missing required train.csv, test.csv, or taxonomy.json is the only intentional hard input
  failure.
- Search stops at 2400 seconds. A 3000-second wall-clock guard preserves the already-written
  placeholder instead of launching late final training, leaving inference/output margin.
- The final validator checks columns/order, episode IDs/count, exact call-ID coverage, exact
  call keys, label domains, finite bounded confidences, and graph/call equality before the
  real submission overwrites the placeholder.
