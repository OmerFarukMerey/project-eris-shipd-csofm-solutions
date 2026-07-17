Overview
Each row represents a small local neighborhood from an evolutionary tree. The original tree connected several endpoint taxa through internal branches. In the released example, one internal branch has been hidden and replaced by the placeholder node X. The endpoint names are anonymized as row-local tokens such as u03 or u11, but the row still provides partial relationship evidence between endpoints: integer hop counts and optional within-row branch-length ranks.

Your task is to recover the exact set of endpoint tokens that lie on the same side of the hidden branch as anchor_taxon. In other words, if the hidden branch were restored, which visible endpoints would be grouped with the anchor before crossing that branch? A prediction is a variable-length set of row-local endpoint tokens, not a class label or scalar value. The token vocabulary is independently reassigned in every row, and evaluation rows come from held-out source groups, so solvers must infer the local branching pattern rather than reuse token identities.

Dataset
train.csv: 3,442 rows with inputs and answer_json.
test.csv: 1,758 rows with inputs only.
sample_submission.csv: 1,758 deterministic schema-only predictions in submission format. Each row contains the anchor and one pseudo-random incident endpoint; it is not a model or evidence-based baseline.
train.csv has five columns; test.csv has the same first four columns:

id (string): Globally shuffled opaque row identifier.
collapsed_context_json (JSON object encoded as a string): Contains collapsed_node (string, always X), incident_taxa (array of 6 to 16 row-local endpoint-token strings), and observed_pair_count (integer).
anchor_taxon (string): Endpoint token that must be included in the submitted side.
distance_evidence_json (JSON array encoded as a string): Sparse local relationship evidence. Each object contains a and b (endpoint-token strings), hops (integer number of edges between them in the original local tree), and length_rank (integer 0 through 7 or JSON null). Ranks preserve only within-row ordering of branch lengths when that information exists; null means the clue carries hop structure only.
answer_json (train only; JSON object encoded as a string): Ground-truth object with exactly one key, side, whose value is the anchored child-side list.
Every endpoint token occurs in at least one observed pair. The target side always contains anchor_taxon, has at least two tokens, and has no more than half of the incident tokens.

Submission Format
Submit a CSV with exactly these two columns, in either column order:

id: Every test id exactly once.
answer_json: A JSON object with exactly the key side.
side must be a nonempty list of unique endpoint-token strings from that row. It must contain anchor_taxon and must omit at least one incident endpoint token. No other JSON keys are allowed.

A complete valid two-row CSV example is:

id,answer_json  
clade_d23a9881c7b8de10b8,"{""side"":[""u06"",""u03""]}"  
clade_d4eec008e521559582,"{""side"":[""u04"",""u02""]}"  

The full submission must contain all 1,758 test ids, not only the two example rows.

Evaluation
For row i, let P_i be the submitted endpoint-token set and T_i the true anchored side set. The row score is exact side-set accuracy: it is 1 when P_i = T_i and 0 otherwise.

The final score is the unweighted arithmetic mean across all N = 1,758 test rows:

score = (1 / N) * sum_i 1[P_i = T_i].

The score is bounded by 0 and 1 and is maximized. Partial overlap does not receive credit because the target is one exact branch-side set, not an independent per-token tagging task. Submission row order does not matter because rows are aligned by id.

Expected Methods
Suitable CPU methods include row-local graph feature extraction, constrained set search, path-consistency scoring, rank-aware clustering, lightweight message passing implemented with installed libraries, and compact models trained only on the supplied training rows. Models should exploit the interaction between hop clues, rank clues, the anchor token, and missing-edge geometry rather than memorizing token identities.

What Not To Use
GPU or accelerator computation of any kind.
Hosted APIs, remote inference services, or network access during solution execution.
External datasets, source archives, endpoint-name databases, or extra training data.
Challenge-specific pretrained or fine-tuned checkpoints presented as general pretrained assets.
Reverse lookup of rows, source-paper matching, original endpoint-name recovery, or source-record retrieval.
Hardcoded answers, test-id rules, manual per-row answer tables, or reconstruction from leaked identifiers.
Solutions must be reproducible on CPU using only the provided files and libraries already installed in the runtime.