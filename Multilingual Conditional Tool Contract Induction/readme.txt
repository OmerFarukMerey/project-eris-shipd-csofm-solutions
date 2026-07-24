Multilingual Conditional Tool Contract Induction
================================================

Submission contract
-------------------
The output is a CSV with exactly two columns, in this order:

contract_id,induced_contract

Every test contract_id is copied exactly once and in input order. induced_contract is a CSV-escaped JSON string containing exactly these keys in this order:

{"target_tool":"...","routing_rule":{"argument":"...","operator":"required|forbidden"},"required_arguments":[...],"optional_arguments":[...],"peer_tool":"..."}

target_tool and peer_tool are non-empty raw tokens matching [a-z0-9_]{1,64}. routing_rule.argument is one name from that row's argument registry. routing_rule.operator is exactly required or forbidden. required_arguments and optional_arguments are lexicographically sorted arrays of unique registry names, and the two arrays are disjoint. The final writer uses compact JSON and normal CSV quoting.

Metric
------
For each row, target_tool and peer_tool receive exact-string accuracy. Required and optional argument arrays receive exact set F1; two empty sets score 1 and an empty/non-empty pairing scores 0. Routing is the mean of exact gate-argument and exact operator accuracy. Interface is the mean of required-set and optional-set F1.

component_base = 0.10 * target_tool
               + 0.30 * routing
               + 0.20 * required_F1
               + 0.20 * optional_F1
               + 0.10 * peer_tool

balance_multiplier = 0.5 + 0.5 * min(target_tool, routing, interface, peer_tool)
row_score = component_base * balance_multiplier + 0.10 * exact_complete_contract

exact_complete_contract is 1 only for an exact five-field canonical contract. The final metric is mean row score clipped to [0,1]. If one identical non-empty contract occurs in more than 20% of test rows, the final score is capped at 0.10. Every metric-aware threshold and score blend in solution.py is selected by an in-script search over train-only sibling-tool validation data.

Domain and restrictions
-----------------------
This is NLP under Guidebook section 5.1: multilingual semantic induction and constrained text generation. The implementation uses only CPU, caps Torch and numerical libraries at 10 threads, writes an early schema-valid fallback, has wall-clock guards, and uses less than the 62 GB limit. It reads only the two positional-input CSV files and writes only the positional output path. It does not use external datasets, APIs, translations, source-corpus matching, hidden identifiers, or private labels. The only downloaded/cached weights are general-purpose Hugging Face backbones.

Approach
--------
1. Parse each episode as one unit: six accepted requests, four contrasts, and the 55-name registry. Malformed individual values receive conservative structural fallbacks rather than aborting the run.
2. Encode every request and every argument description with sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2. The encoder is used only as a general-purpose backbone transform.
3. Train a multiple-instance ContractModel on train.csv. It learns request-to-argument probabilities from semantic argument prototypes plus a neural residual. A differentiable Poisson-binomial aggregation maps the six accepted request probabilities into absent, optional (one or two), and required (at least three) categories. A learned route head compares sorted accepted/contrast evidence for all arguments and predicts the gate argument and operator. Optional, required, and operator decode offsets are searched against the documented weighted component metric on held-out sibling tools.
4. Train a contrast grouper from train-only positive examples. A supervised discriminant projection learns tool-semantic geometry; a train-only search selects the raw/projected similarity blend. At inference, exactly two contrast requests closest to the accepted behavior are treated as gate violations and the remaining two as peer evidence.
5. Train open-label family, semantic-regression, and candidate-scoring models on train.csv. Family regularization, ridge regularization, and candidate-scorer regularization are selected on tools withheld by family. Final versions are then fit on all train episodes.
6. Fine-tune Qwen2.5-1.5B-Instruct in-script with manually implemented rank-4 low-rank adapters on attention query/value projections. The held-out tools are never generator targets. Per prompt, four related labeled sibling demonstrations are selected from a train-only semantic bank; they support the generator but cannot supply an unseen answer. The adapters are merged into the backbone before inference.
7. Generate target and peer token beams independently for each test episode. Generic normalization and family recombination create candidate spellings from model-generated tokens. A trained semantic scorer ranks them. Its blend with generator order is searched in-script on generated candidates from held-out train tools. If generation fails, rankings fall back to train-label candidates produced by the trained scorer, never an empty field.
8. Enforce route/interface invariants, serialize strict JSON, and audit column order, IDs, row count, key order, registry membership, sorted/disjoint arrays, non-empty tool syntax, and parseability before completion.

Validation
----------
The split is group-aware by raw target tool. One sibling tool from each eligible family is withheld, so semantic examples from a held-out tool cannot enter the corresponding fit partition. The fixed seed is 314159.

Observed final local train-only validation diagnostics:
- Structured contract model: gate argument accuracy 0.3381; operator accuracy 0.6095; required-set F1 0.3699; optional-set F1 0.2678. Its structured-only contribution to the exact challenge formula (with unavailable tool components set to zero) was 0.134838.
- Contrast grouping agreement with independently trained target/peer pseudo-groups: 0.8885; searched raw-embedding weight 0.000.
- Open-label family accuracy: 0.5667.
- Open-label semantic candidate exact accuracy: 0.4571.
- Fine-tuned generator: top-1 exact tool accuracy 0.2857 and three-beam exact recall 0.5714 on 35 views from seven unseen sibling tools.
- End-to-end generated candidate recall: 0.7143; calibrated candidate-ranking exact accuracy: 0.5143; searched generator-order weight: 0.25.

The final local command completed in 2483.8 seconds on CPU and reported: submission audit=True (ok); rows=450. A later identical code-path verification after rank calibration also completed successfully; no score is claimed for unlabeled test data.

Leakage audit: explicit taint trace
----------------------------------
Raw test CSV -> parsed test_records: parsing is row-local.
test_records -> test_positive/test_contrast: the frozen encoder transforms requests independently; no fitting occurs.
test embeddings -> structured outputs: ContractModel predict is episode-local.
test contrast embeddings -> test_peer_indices/test_peer_embeddings: selection compares only the four contrasts with the six positives from that same row.
test episode embedding -> demonstrations: it compares that one row with a bank whose centroids, representatives, labels, and statistics were built exclusively from train.csv.
test prompts -> target_beams/peer_beams: batched model inference has no cross-row aggregation, fitted state, counts, normalization, or calibration.
beams and that row's embeddings -> candidates/rankings: all construction and scoring are row-local against train-fitted models.
rankings plus structured output -> prediction JSON -> CSV: only row-local invariants and schema validation are applied.

No operation computes a mean, count, vocabulary, cluster, threshold, class balance, pseudo-label, calibration, or model state across test rows. There is no train+test concatenation. Every learned vocabulary, prototype, representative, threshold, blend, classifier, regressor, neural head, and adapter is fit on train.csv only. Test is transform/predict only. contract_id is used solely as the required output key and never as a feature.

Hardcoding audit
----------------
There is no phrase-to-label dictionary, intent table, external ontology, test-derived class list, hand-authored argument rule, sequential-ID rule, or discovered generation pattern in solution.py.

The asserted CONTRACT_KEYS list, operator values, token regex, registry checks, JSON ordering, and required/forbidden invariants come directly from the submission specification and only validate/render predictions. Record dictionaries hold parsed data or learned outputs; they do not map phrases to answers. Prompt text is a generic instruction. The demonstration bank and known-token list are derived from train.csv each run. Generic token normalization and candidate recombination only support Qwen outputs; the fine-tuned generator and trained semantic scorer decide the submitted token. The fallback strings unknown and unknown_peer are deliberately unusable schema protection for a catastrophic pre-training failure, not semantic predictions.

All data-dependent output thresholds, regularizations, projection blends, category biases, and generation/ranking blends are learned or searched in-script on train-only held-out tools. Fixed model names, seed, dimensions, batch sizes, optimizer settings, and wall-clock limits are architecture/reproducibility/resource choices, not pasted answer mappings or offline-tuned corrections.

Strip-the-ML result: with all trained/fine-tuned models removed, the pipeline produces only the unusable unknown/unknown_peer emergency contract plus schema formatting; it produces no meaningful target tool, peer tool, gate, or interface answers. Retrieval, templates, regex, and candidate transforms cannot produce a usable answer on their own.

Housekeeping and experiments
----------------------------
Random seeds are fixed. solution.py is the only Python file in this challenge directory and imports no local code. The early fallback and both training/inference wall-clock guards are active. The script is invoked as:

python3 solution.py <public_dir> <submission_out>

A compact seq2seq generator tended to emit generic families rather than exact raw tool tokens. Unadapted semantic similarity also underperformed the learned multiple-instance and candidate models. Six-beam decoding increased held-out candidate recall only from 0.7143 to 0.7429 without improving calibrated exact accuracy (0.5143) and exceeded the conservative local one-hour command budget, so the final train-searched three-beam path was retained. The low-rank Qwen adapter, train-only sibling demonstrations, and metric-calibrated semantic reranking were the strongest compliant combination.