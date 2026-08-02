Animal Habitat Profile Extraction & Inference
==============================================

Problem classification and rules
--------------------------------
Guidebook domain: Biology/Chemistry/Other, Section 5.6, implemented as a supervised NLP text-classification problem. PROBLEM.md does not categorize this as a Fine-tuning challenge and imposes no fine-tuning requirement. Its challenge-specific closed-book restrictions do apply: this solution uses no network access, pretrained weights, external datasets, biological databases, gazetteers, species re-identification, ID signal, row-order signal, or private-key information. It trains from scratch on public/train.csv on every run.

Exact submission schema
-----------------------
The CSV columns and their order are exactly:

id,activity,locomotion,social,reproduction,trophic_guild,habitats,climate

There is exactly one row for each test.csv id, in test order, with no duplicates.

Single-valued fields contain exactly one token:
- activity: nocturnal, diurnal, crepuscular, cathemeral, or unknown
- locomotion: terrestrial, arboreal, aquatic, semiaquatic, fossorial, volant, or unknown
- social: solitary, social, or unknown
- reproduction: viviparous, oviparous, ovoviviparous, or unknown
- trophic_guild: carnivore, herbivore, omnivore, insectivore, piscivore, or unknown

Multi-valued fields contain unique tokens joined by a literal semicolon with no surrounding spaces. An empty string denotes an empty set. Habitat token case follows the required vocabulary: Agricultural; Caves; Coastal; Forest; Freshwater; Grassland; Marine; Mountains; Rainforest; Rocky areas; Savanna; Shrubland; Wetlands. Climate tokens are tropical; temperate; cold; arid; polar. Set order is immaterial to the grader; the script emits tokens in vocabulary order.

Exact evaluation metric
-----------------------
For each single-valued field, the field score is exact normalized-match accuracy (1 or 0). For habitats and climate, the score is set Dice:

    Dice(A, B) = 2 * |A intersect B| / (|A| + |B|)

It is 1 when both sets are empty and 0 when exactly one is empty. The item score is the field-weighted average with habitats weight 20, climate weight 2, and each of the five single fields weight 1, for total weight 27. The final score is the tier-weighted mean of item scores, using each row's difficulty tier 1-4. There is no additional tail, worst-group, or subgroup term. The script implements this exact metric for all hyperparameter and decoder selection.

Approach
--------
1. Read train.csv and test.csv from sys.argv[1]. Immediately write a schema-valid unknown/empty placeholder to the exact sys.argv[2] path.
2. Build word TF-IDF features (unigrams and bigrams) and character-within-word TF-IDF features (3-5 grams) from each complete record. A second independently normalized word TF-IDF block represents only the record's Realms and Continents sections. The section parser is generic and only creates model inputs.
3. Fit five multiclass logistic-regression models for the extraction fields.
4. Fit independent binary logistic-regression heads for every habitat and climate label. A target column with only one observed training value retains that training-observed constant; no input phrase is mapped directly to an answer.
5. Generate primary out-of-fold predictions with GroupKFold grouped by the Order parsed from each training record. Entire taxonomic orders are therefore absent from each fold's model fit, matching the challenge's test construction.
6. Search logistic regularization values in-script. Search two multilabel decoder families in-script against the exact tier- and field-weighted metric: coordinate-searched per-label thresholds, and a per-row expected-Dice decoder whose temperature, bias, and empty-set choice are searched. No selected C, threshold, temperature, bias, ensemble size, or cardinality rule is pasted into the final predictions.
7. For habitat only, train up to five additional five-fold order-grouped ensembles from fixed shuffled group partitions. After each repeat, average train OOF probabilities, retune the decoder, and select the repeat count only when train-only weighted OOF Dice improves. The selected fold models transform and predict each test row independently, and their probabilities are averaged per row.
8. Refit the selected extraction and climate models on all training rows. Use the selected habitat fold ensemble when its OOF Dice beats the primary model, otherwise fit the selected habitat model on all training rows. Decode each test row independently and write the final CSV.

Model architecture / algorithm
------------------------------
The solution uses sparse linear text models: one word TF-IDF block, one character TF-IDF block, and an extra normalized geography block feed genuinely trained logistic-regression classifiers. Habitat uses a probability ensemble of order-grouped logistic-regression fold models when train-only OOF search selects it; all other fields use one selected full-training model per target. It does not use a frozen embedding model, pretrained language model, retrieval answer, template, or rule-based label extractor. Logistic regression produces all seven predicted fields. The mandated vocabularies are only label encoders and schema validators.

Validation strategy and observed score
--------------------------------------
Primary validation uses five-fold GroupKFold on train.csv only, grouped by parsed Order. Every row receives a prediction from a model that was fit without any row from that order. TF-IDF vocabularies and IDF values are refit inside each fold on that fold's training partition only. Difficulty tiers weight the exact challenge metric but are never input features.

Habitat ensemble selection evaluates cumulative ensembles over five further repeated five-fold partitions of the unique training orders. Within every repeat, each row is predicted exactly once by models fit without its order; cumulative OOF probabilities are averaged across repeats. The script searches repeat counts 1-5 and all decoder parameters against train-only tier-weighted habitat Dice; the observed optimum is an interior three-repeat ensemble. Because the overall metric is an additive weighted sum of field scores, replacing the primary habitat component with the repeated-OOF habitat component gives the exact reported aggregate metric.

Observed local end-to-end run:
- extraction C selected in-script: 64
- tier-weighted mean accuracy across the five extraction fields: 0.989342
- inference C selected in-script: 1
- primary habitat Dice: 0.636595
- selected three-repeat habitat Dice: 0.650876
- climate Dice: 0.885456
- primary exact order-held-out metric: 0.720353
- improved exact grouped-OOF metric: 0.730931
- selected habitat decoder: searched per-row soft-Dice decoder
- selected climate decoder: searched per-label threshold decoder

These are OOF model-selection scores: the candidate C values, ensemble repeat count, and decoding parameters are selected on grouped OOF predictions themselves, so they are useful for configuration comparison but are not described as untouched final-test estimates. The final script repeats every search rather than loading these observed selections.

What worked / what did not
--------------------------
Word and character features were complementary for inconsistent spelling, synonyms, repeated attributes, and structured prose. Giving Realms and Continents their own normalized block improved habitat inference without converting geography into an asserted answer rule. Stronger regularization for the multilabel heads and weak regularization for the nearly explicit extraction fields were selected consistently by grouped validation. Metric-searched soft cardinality decoding helped habitat Dice; climate favored searched per-label thresholds. Repeated order-grouped probability bagging was the strongest compliant habitat improvement, raising habitat OOF Dice from 0.636595 to 0.650876 and the aggregate metric from 0.720353 to 0.730931.

Development comparisons that did not consistently improve order-held-out validation included over-weighting separately parsed ecology blocks, taxonomy removal, class-conditional local models, classifier chains, label-powerset decoding, sparse tree ensembles, boosted trees, CatBoost text models, nonlinear kernels, MLPs, and neural multilabel losses. They are omitted from the final script rather than retained as unused complexity.

Leakage audit: test-taint data flow
----------------------------------
Raw test-derived variables and every downstream operation are:
1. test DataFrame: read once from public_dir/test.csv. Its id column is copied to the placeholder/final output and checked for schema completeness. IDs never enter a feature matrix or model.
2. test record strings -> test_records: missing-value cleanup and string conversion, independently per row.
3. test_records -> geography strings: generic section splitting, independently per row.
4. test_records/geography strings -> test_text and test_inference: transform only with TF-IDF objects fit on training records. For the habitat ensemble, each vectorizer is fit on one grouped training-fold partition; for full models, it is fit on all train rows. TF-IDF row normalization is per row.
5. test feature rows -> single-field, habitat, and climate probabilities: predict_proba from models fit on train only. Habitat fold probabilities are stacked with shape (models, test rows, labels) and averaged only on the model axis. An output row therefore reads predictions for that same row from multiple trained models, never features or predictions from another test row.
6. single probabilities -> labels: argmax across classes within each row only.
7. habitat/climate probabilities -> decoded sets: comparisons, sorting, cumulative sums, sums, products, maxima, and argmax all use the label axis within each row. No decoder reads another test row.
8. decoded rows -> submission rows: model-column labels are serialized through the mandated output vocabularies, independently per row.
9. final submission audit: row count, exact ID sequence/uniqueness, column order, and vocabulary validity are checked. These integrity checks do not modify, calibrate, rebalance, rank, or otherwise influence predictions.

There is no train+test concatenation. No statistic is computed across test rows. No test-derived vocabulary, IDF, scaler, encoder, threshold, temperature, class frequency, label frequency, cardinality, count, quantile, calibration, or ensemble size is fit or estimated. Every vectorizer and every model is fit on train only; HPO, repeat-count selection, and decoding search use order-held-out training predictions only. Test is used only for per-row transform and predict, per-row averaging across trained models, schema validation, and writing.

Hardcoding / real-ML audit
--------------------------
Output-influencing dictionaries and mappings were inspected:
- SUBMISSION_COLUMNS, SINGLE_VOCABS, HABITAT_VOCAB, CLIMATE_VOCAB, and FIELD_WEIGHTS reproduce the mandatory schema, controlled vocabularies, and metric from PROBLEM.md. They do not map record content to answers.
- The multilabel label-to-column dictionary and model class-to-column dictionary are label encoders. The value in each column is produced by a fitted model (or learned from a truly single-valued training target column).
- The generic section parser names Realms and Continents for a model feature block and Order for leakage-safe grouping. It never emits or overwrites an output label.
- Candidate C values, decoder grids, and bagging seeds are ordinary search/protocol constants. The selected regularization, decoder family, per-label thresholds, temperature, bias, empty-set behavior, and ensemble repeat count are all selected in-script from grouped train-only predictions against the exact metric. The five-repeat search extends beyond the selected three-repeat optimum, so the final ensemble size is not a pasted search boundary. None of the seeds or candidates maps an input to an output.
- The unknown/empty placeholder and per-row unknown/empty exception fallback exist only to satisfy the required robustness contract. They are not used on the observed successful path and contain no discovered dataset answer pattern.

There is no input-phrase/token/taxon-to-label dictionary, output if-chain, regex-driven answer mapping, habitat template, frequency-mined answer table, nearest-neighbor answer, ID arithmetic, or external biological fact. No discovered generation pattern is hardcoded. The trained logistic-regression models and their selected probability ensemble produce every normal-path answer; TF-IDF and generic parsing only provide features.

Strip-the-ML result: with all trained models removed, the normal prediction pipeline produces no answers. Ensemble averaging cannot operate without those genuinely fitted fold models. Only the deliberately non-informative schema-valid emergency placeholder (unknown for single fields and empty for set fields) remains. It is not a usable solution. The models are therefore the answer producers, not selectors over hand-built answers.

Housekeeping and runtime
------------------------
Random seeds are fixed at 42 for estimators and 13, 97, 211, 307, and 401 for the searched grouped habitat repeats. solution.py is self-contained, imports nothing local, reads only train.csv/test.csv below public_dir, and writes only submission_out. It neither reads sample_submission.csv nor writes caches/model artifacts. Its 2400-second wall-clock guard stops launching additional primary validation folds or habitat repeats; a 3000-second warning records if final training starts unusually late. The observed full five-repeat search and end-to-end prediction run completed in 869 seconds. solution.py writes the early placeholder before vectorization/training and catches individual output-row failures without aborting other rows.
