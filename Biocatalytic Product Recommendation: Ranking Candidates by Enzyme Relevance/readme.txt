Biocatalytic Product Recommendation: Ranking Candidates by Enzyme Relevance

1. Problem contract

Domain classification
---------------------
This is a Biology/Chemistry/Other challenge under Solver Guidebook section 5.6. PROBLEM.md does not categorize it as a Fine-tuning challenge and does not require a pretrained backbone. The final method is a fully trained chemical learning-to-rank stack trained from the supplied train.csv on every invocation. It loads no pretrained weights, external reaction data, reaction database, web result, or cached artifact.

Exact submission schema
-----------------------
The output is a CSV with exactly these columns and this order:

id,score

- id: a string copied verbatim from test.csv. Every test id appears exactly once, in test row order.
- score: one finite decimal numeric float in [0, 1], expressing candidate-product relevance. It is written as the standard pandas CSV decimal representation, with no list, JSON, label, or other wrapper.
- There is one data row per test.csv row and no index column or extra field.

The platform invocation is:

python3 solution.py <public_dir> <submission_out>

Both paths come from sys.argv. The output parent is created before writing. A complete id,score placeholder with score 0.5 is written immediately after test.csv is read and is replaced first by a trained half-split model, then by the full OOF stack.

Exact evaluation metric
-----------------------
For binary feedback y and submitted relevance s clipped to [1e-9, 1-1e-9]:

rank = clip(2 * ROC_AUC(y, s) - 1, 0, 1)
NLL = -mean(y * ln(s) + (1-y) * ln(1-s))
cal = clip(1 - NLL / ln(2), 0, 1)
hard_rank = clip(2 * ROC_AUC(y, s; positives plus hard negatives only) - 1, 0, 1)
quality = 0.45 * rank + 0.20 * cal + 0.35 * hard_rank

gate(q) = [tanh(300 * (q - 0.995)) - tanh(300 * (0 - 0.995))]
          / [tanh(300 * (1 - 0.995)) - tanh(300 * (0 - 0.995))]

final = 0.10 * quality + 0.25 * quality^2 + 0.65 * gate(quality)

The script implements this metric directly for in-script model, feature-set, and calibration selection. For training validation, a negative is in the hard subset only when its substrate/candidate reaction has an observed positive EC, the supplied EC is different, and it shares the positive EC's top-level family. This reproduces the hard-negative definition in PROBLEM.md; it affects validation/HPO only and is never used as a test-time answer rule.

Challenge-specific restrictions followed
----------------------------------------
- The model ranks supplied candidates; it does not generate molecules.
- No external reaction/enzyme lookup, chemical literature, API, or pretrained model is used.
- id is used only to copy rows into the submission. It is never a feature.
- No row order, class-balance correction, feedback counting on test, or private file is used.
- The observed-product-disjoint requirement is reflected in candidate-grouped validation.
- All training and HPO happen inside solution.py from raw train.csv.

2. Approach

Model architecture / algorithm
------------------------------
The stack models the task as learned compatibility between a reaction representation and the supplied EC hierarchy.

A. Generic reaction representation

Each row is represented independently in several complementary views:

1. Character TF-IDF: role-separated character n-grams for substrates and candidate.
2. Lexical SMILES TF-IDF: role-separated n-grams of bracket atoms, element tokens, aromatic tokens, ring indices, bonds, and punctuation.
3. Reaction-difference text: the candidate is aligned to the most similar substrate component by generic multiset/string comparison. Gained/lost character n-grams and local edit contexts become tokens.
4. Molecular graph fingerprints: the SMILES parser builds atom/bond adjacency, then three rounds of Weisfeiler-Lehman neighborhood hashing describe candidate atoms, substrate atoms, and gained/lost neighborhoods relative to the closest substrate and to all substrates.
5. Numeric reaction features: per-row length, component count, character/token overlap, sequence alignment, bond-symbol changes, and candidate-minus-substrate character-count differences.

The character alphabet is learned from train only. Every TF-IDF vocabulary and IDF statistic is fit on the current training partition, then only transformed on validation/test. The parsers contain molecular syntax but no EC-to-transformation or reaction-to-label rule.

B. Learned enzyme compatibility

Positive training reactions teach multiclass models to predict EC prefixes from the reaction views:

- LinearSVC models over the character, lexical, and graph-difference views predict EC levels 2 and 3.
- MultinomialNB and ComplementNB graph models predict EC levels 2, 3, and 4.
- Train-positive prototypes at EC levels 2, 3, and 4 measure maximum, mean, top-three, rank, support, and confidence of graph, character, and reaction-difference similarities.

For the EC supplied on a row, each classifier emits learned compatibility features: supplied-class score, gap from the best class, within-row rank and normalized confidence, known-class indicator, top score, and across-class spread. All reductions are across enzyme classes or train-fitted prototypes for one record, never across test records.

C. Direct relevance models and meta-ranker

An SGD logistic classifier is trained on all labeled fit rows over the tensor-product interaction between graph-difference TF-IDF and train-fitted EC prefix codes. Five regularization values become separate learned relevance features. This directly learns whether a reaction graph change is compatible with its supplied enzyme; hard negatives receive additional training weight.

The final binary meta-ranker receives:

- all EC compatibility, prototype, and direct-relevance outputs,
- generic per-row reaction features,
- positive/negative EC hierarchy support counts learned from the training reference partition,
- EC levels 1-4 as categorical variables.

The primary meta-ranker is CatBoostClassifier; HistGradientBoostingClassifier is the no-CatBoost fallback. Counts are inputs to a trained model, never emitted as scores. Every test EC lookup uses maps fitted from train only.

D. In-script search and final fitting

The script evaluates:

- three LinearSVC regularization values per character, lexical, and graph view,
- five direct graph/EC SGD regularization values,
- three smoothing values for each of two graph Naive Bayes families,
- six CatBoost depth/regularization/hard-negative-weight/feature-count configurations,
- three feature families and four HistGradientBoosting configurations as fallbacks,
- seven calibration slopes and seven calibration biases.

Selection uses the exact challenge metric on a product-disjoint train validation partition. A separate train-only feature-importance fit ranks numeric meta-features for the feature-count candidates. The chosen model, feature count, hard-negative weight, and calibration are recomputed in-script; no locally selected output parameter is pasted into the answer path.

After HPO, four candidate-grouped folds produce out-of-fold train features and per-row test predictions. The final meta-ranker trains on all 14,714 OOF rows. Test base outputs are averaged only across trained models for the same row, which is ordinary model ensembling and not a statistic across test rows.

3. Validation

Strategy
--------
StratifiedGroupKFold uses candidate SMILES as the group. All records for one candidate stay in one fold, so validation products are disjoint from base-model training products. This mirrors the stated test condition and also keeps each positive/hard-negative reaction family together.

The clean HPO path uses four disjoint candidate folds:

- folds 1-2: fit reaction vectorizers, graph/prototype models, and EC-prefix compatibility models,
- fold 3: train the binary relevance meta-ranker and train-only feature selector,
- fold 4: select the meta configuration and calibration against the exact metric.

Thus the reported validation labels are not used by the reaction/EC models or the meta-ranker. Production then performs separate four-fold OOF fitting over all train rows.

Measured local validation
-------------------------
Observed from the required local end-to-end command on the supplied public data:

- final challenge score: 0.258380
- quality: 0.836109
- ROC AUC: 0.949762
- normalized rank: 0.899524
- hard-negative ROC AUC: 0.948279
- normalized hard rank: 0.896558
- calibration component: 0.587637
- NLL: 0.285828

The previous implementation measured 0.221956 on this same untouched product-disjoint validation fold. The integrated graph/prototype/direct relevance stack gains 0.036424 absolute. The improved run selected CatBoost configuration 2 with slope 0.95 and bias 0.4, completed all four production folds in 1,161.3 seconds on the local Apple M4 Pro workstation, and wrote 3,286 predictions.

What worked / what did not
--------------------------
Worked:
- Product-grouped validation prevented candidate/product memorization from appearing as generalization.
- Weisfeiler-Lehman reaction differences added molecular-neighborhood information missing from raw character edits.
- Positive EC prototypes complemented multiclass boundaries, especially at the sparse third and fourth EC levels.
- The direct graph-by-EC logistic interaction improved enzyme-conditioned ranking over chemistry similarity alone.
- CatBoost combined heterogeneous margins, similarities, support, and categorical EC evidence more effectively than the previous histogram model.
- Exact-metric search improved calibration without touching test outputs or their distribution.

Did not work well enough:
- Candidate/substrate character similarity alone tied positives and hard negatives sharing the same reaction pair.
- Flat text relevance models that ignored the EC/reaction interaction were weak under product-disjoint validation.
- A fine-tuned compact pretrained chemical text encoder and a pairwise neural hard-negative objective were both weaker on the untouched fold and were removed completely from the final script.
- A standalone pairwise CatBoost ranker and additional boosted-model seed blends did not improve the exact metric and were not integrated.

4. Leakage audit

Train/test separation
---------------------
No train+test concatenation occurs anywhere. No transform, vocabulary, alphabet, count, class encoder, scaler, classifier, calibration, threshold, model choice, or hyperparameter is fit from test.

Every fitted object and statistic:
- character alphabet: fit from train only,
- candidate groups/folds and hard-negative masks: train labels only,
- TF-IDF vocabularies/IDFs: fit on the relevant base-training partition only,
- LabelEncoder EC classes and prototype class sets: fit on positive base-training rows only,
- LinearSVC and Naive Bayes weights: fit on positive base-training rows only,
- direct SGD relevance weights: fit on labeled base-training rows only,
- positive/negative EC counts: computed from the reference training partition only,
- CatBoost/HistGradientBoosting meta-ranker and CatBoost feature importance: fit on labeled train partitions/OOF features only,
- model/calibration selection: evaluated on a labeled train validation partition only.

Test taint trace
----------------
Raw test-derived variables and every operation applied to them are:

1. test: fill missing cell values and cast each cell to str; no cross-row statistic.
2. test_signatures and test_graph_signatures: independent generic parsing/hashing for each row.
3. test_numeric: numeric_row_features independently for each row, using a train-fitted alphabet.
4. query matrices: transform(test) with train-fitted character, lexical, graph, and difference vectorizers.
5. classifier score matrices: predict(test); class reductions operate across train-fitted EC classes within that row only.
6. prototype similarities: each test row is compared independently with positive training-partition reactions, then reduced within train-fitted EC classes.
7. direct graph/EC interactions: each row combines its own transformed graph tokens with train-fitted EC codes before predict(test).
8. base/advanced test features and counts_test: per-row trained outputs plus lookups from train-fitted maps.
9. meta_test: per-row concatenation of that row's learned, numeric, categorical, and train-count features.
10. raw_test/interim_scores/final_scores: predict(test), then elementwise sigmoid calibration and clipping.
11. test feature sums: elementwise ensembling across trained folds for the same row, divided by model count rather than a test-row statistic.
12. submission: each id is paired with its own final score.

There is no sorting, bincount, quantile, mean/std across test rows, distribution matching, class-balance adjustment, pseudo-labeling, clustering, adaptation, or calibration on test. len(test) is used only for output allocation/completeness.

5. Hardcoding / real-ML audit

Mappings, regexes, constants, and templates
------------------------------------------
- SMILES_PATTERN and ATOM_TOKEN_PATTERN are generic lexical/syntax tokenizers. BOND_LABELS records bond syntax for graph adjacency; none maps chemistry or EC to relevance.
- Weisfeiler-Lehman hashes encode only the parsed neighborhoods present in the current row.
- BASE_FEATURE_INDEX, class_index dictionaries, and feature-set arrays are column layouts or train-fitted class maps, not answer mappings.
- reaction_diff_signature and graph_reaction_signature generically record per-row differences. They contain no enzyme, EC, functional-group, reaction-family, or label mapping.
- No dictionary, if-chain, regex, template, ID rule, candidate table, or sequential arithmetic maps an input to an answer.
- Model and calibration candidates are searched in-script against the exact train-only metric. The locally observed CatBoost configuration, slope 0.95, and bias 0.4 are logged outcomes; the script searches them again every run.
- Every data-dependent prediction/decode value is learned or selected in-script. Remaining constants define representation ranges, generic optimization grids, numerical safety, deterministic seeds, and the mandated wall-clock guard; they assert no output class or score.

Strip-the-ML test
-----------------
With the trained LinearSVC, Naive Bayes, SGDClassifier, CatBoostClassifier, and HistGradientBoostingClassifier removed, the remaining code can parse molecules, create vectors/descriptors, compute train-reference similarities/counts, and write the mandatory 0.5 crash placeholder only. It cannot produce usable ranked answers; the constant placeholder has challenge score 0 by definition. All meaningful relevance scores come from trained models.

Cold-reviewer findings
----------------------
Question: Where is the hardcoded generation pattern?
Finding: none. The source contains no EC-to-transformation, molecule-to-label, phrase-to-class, ID-to-label, or generation-rule mapping. The only reaction alignment is a generic feature transform consumed by trained models.

Question: What produces the answer?
Finding: LinearSVC and Naive Bayes models learn reaction-to-EC compatibility, SGD learns direct graph/EC relevance, and the trained CatBoost meta-ranker converts those outputs plus supporting features to a relevance probability. Parsers, hashes, vectorizers, similarities, and counts alone do not emit an answer.

Question: Which test-derived variable is read across rows?
Finding: none. Test arrays are processed row-wise. Matrix reductions are within a row across enzyme classes, and summation is within a row across independently trained models. No operation derives state from multiple test samples.

6. Robustness and housekeeping

- Random seed is fixed at 1729.
- The script reads only train.csv and test.csv under public_dir.
- It writes only submission_out and creates only its parent directory.
- It imports no local module and loads no cache, external data, or external weights.
- A valid placeholder is written before train.csv is read or heavy work starts.
- Per-row string/signature/numeric parsing catches malformed values and emits neutral feature vectors rather than raising.
- Individual base-model failures are skipped; an outer guard preserves the latest valid submission.
- The 3,000-second wall-clock guard stops launching new production folds and proceeds to trained inference.
- Local fallback verification deliberately supplied an undersized/malformed training scenario: the process returned normally and retained a complete two-row id,score placeholder.
- Local final output verification: columns exactly [id, score], 3,286 rows, all 3,286 ids unique and identical to test order, no missing/non-finite values, all scores in [0,1], min 0.0009338059756197, max 0.998089375131757, mean 0.5224884415076877, standard deviation 0.3904761256482194, and 3,249 unique scores.
- solution.py is the only Python source file in the challenge directory and is self-contained.
