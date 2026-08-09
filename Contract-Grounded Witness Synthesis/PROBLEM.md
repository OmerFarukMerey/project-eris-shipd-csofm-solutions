AI coding systems often produce several nearly identical implementations whose only disagreement is an operator or boundary decision. Your task is to decide which implementation satisfies a released semantic requirement card and return a compact, executable certificate for that decision.

Each episode contains:

one semantic acceptance card derived from a real programming requirement;
four complete, alpha-normalized Python implementations;
a bank of complete Python equality assertions.
Exactly one implementation satisfies the acceptance card. Select one to three assertions from the released bank such that all selected assertions pass that implementation and, together, reject the other three. Submit the complete assertion strings, not candidate or witness indices.

This challenge is derived from the 427-task sanitized Mostly Basic Python Problems (MBPP) corpus. MBPP was created through internal crowdsourcing at Google; each task has a description, a solution, and three tests, and the sanitized subset received an additional description-verification pass. The creator generates several mutation-site episodes from a source task when possible. Every episode from one source task stays in one partition.

To prevent direct source matching, public requirement text is released as a controlled semantic cue card rather than a verbatim MBPP sentence. Function/local identifiers are normalized, docstrings are removed, source IDs are not public, and each episode receives a candidate-invariant opaque release marker. The correct candidate is balanced across the four positions.

The behavior matrix used by the grader is private. Solvers must derive candidate behavior through static program analysis or sandboxed CPU execution, then learn or reason which behavior agrees with the semantic card. The grader never executes contestant-controlled text.

The final release contains 394 training episodes and 158 test episodes from 162 and 66 disjoint source families, respectively.

Dataset
All public files are CSV files. JSON-valued columns contain serialized JSON strings and should be decoded with a JSON parser.

train.csv
Contains 394 labeled episodes with these columns:

episode_id: string. Unique opaque identifier beginning with cw_.
contract: string. Controlled semantic acceptance card describing required behavior without reproducing the source sentence.
candidate_implementations_json: string containing a JSON array of exactly four complete Python implementation strings.
witness_bank_json: string containing a JSON array of distinct complete Python equality assertions.
gold_witness_suite: string containing a JSON array of one to three assertions. This is the deterministic smallest suite that preserves the correct implementation and rejects all alternatives.
test.csv
Contains 158 unlabeled episodes with these columns:

episode_id: string. Unique opaque test identifier.
contract: string. Semantic acceptance card with the same construction as train.
candidate_implementations_json: string containing four complete implementations.
witness_bank_json: string containing the legal assertion bank.
test.csv does not contain the target, source family, mutation metadata, canonical candidate, or behavior matrix.

sample_submission.csv
Contains 158 format-only predictions:

episode_id: string copied exactly from test.csv.
predicted_witness_suite: string containing a JSON array of one to three distinct assertions copied exactly from that episode's witness bank.
Sample values are structurally valid but are not guaranteed to certify the correct implementation.

Evaluation
The private grader uses a precomputed binary behavior matrix. For a submitted suite, a candidate survives when it passes every selected assertion. The suite certifies a candidate only when exactly one of the four candidates survives. Zero or multiple survivors means that the row has no certified candidate.

1. Chance-adjusted identity skill — 60%
Let canonical_accuracy be the fraction of test episodes whose uniquely certified candidate is the hidden correct candidate.

identity_skill = max(0, (canonical_accuracy - 0.25) / 0.75)

The 0.25 floor is removed because canonical positions are balanced across four slots. A fixed-position or random guess therefore receives approximately zero identity skill.

2. Correct certificate quality — 25%
For each row, this component is zero unless the suite uniquely certifies the hidden correct candidate. For a correct certificate:

row_certificate_quality = min(1, hidden_minimum_suite_size / submitted_suite_size)

The component is the mean row certificate quality. It rewards complete, compact certificates and pays nothing for a certificate attached to the wrong behavior.

3. Exact canonical suite — 15%
This row component is 1 only when the submission uniquely certifies the correct candidate and the submitted JSON array exactly equals the hidden gold suite. Otherwise it is 0. Gold assertions are lexicographically sorted. Ties between minimum suites are resolved by the lexicographically smallest assertion tuple.

Final formula
score = 0.60 * identity_skill + 0.25 * mean_certificate_quality + 0.15 * mean_exact_suite

The score is finite, clipped to [0,1], and maximized. A perfect submission scores exactly 1.0.

Submission Format
Submit submission.csv with exactly these columns in this order:

episode_id,predicted_witness_suite

Requirements:

episode_id is a string and every test ID must appear exactly once.
Missing, duplicate, unknown, or extra IDs are rejected.
predicted_witness_suite must be valid JSON representing an array of one, two, or three distinct strings.
Every string must exactly match one assertion in that episode's witness_bank_json, including whitespace, quotes, and punctuation.
Sort selected assertions lexicographically to be eligible for exact-suite credit.
A correctly formatted two-row example is:

episode_id,predicted_witness_suite  
cw_example_a,"[""assert candidate_fn(4) == 16""]"  
cw_example_b,"[""assert candidate_fn([3, 1]) == 3"",""assert candidate_fn([7]) == 7""]"  

The example IDs and assertions illustrate CSV/JSON escaping only.

Requirements
Use CPU computation only.
Maximum runtime is 90 minutes on 10 CPU cores and 62 GB RAM.
Fit learned representations, classifiers, thresholds, and calibration using training labels only.
Episode-local static analysis or sandboxed execution of the released candidate implementations is allowed.
Write the final file to working/submission.csv when using the supplied layout.
What Not To Use
Do not retrieve or match original MBPP tasks, descriptions, solutions, tests, IDs, or labels from Hugging Face, GitHub, mirrors, search engines, memorized tables, or other external sources.
Do not use hosted inference APIs, GPUs, CUDA-only libraries, private answers, hidden behavior matrices, source family keys, or mutation metadata.
Do not fit representations on the complete test corpus or manually annotate test rows.
Do not submit candidate indices, witness indices, probabilities, or assertions absent from the row's released bank.
If executing released code, use process isolation and strict time/resource limits. Never execute submitted assertion text received from an untrusted party.
 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.