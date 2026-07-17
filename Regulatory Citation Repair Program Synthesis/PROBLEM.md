Regulatory Citation Repair Program Synthesis
Overview
Regulations often refer to nearby provisions: a rule may cite one section, a range of sections, a list of exceptions, or an entire containing scope. When regulations are reorganized or locally amended, a citation can become stale: its stored target is no longer the intended section, range, list, or scope.

This challenge turns that situation into a structured-generation task. For each row, you receive:

an anonymized local regulatory hierarchy;
several citation cards in their current, possibly stale state;
a short amendment note naming the citation to inspect and describing the kind of local consistency pressure.
Your job is to output an executable JSON patch program that repairs the hidden scored citation. A repair means changing the citation's form and/or target nodes so that it matches the hidden post-amendment citation state.

The source hierarchy comes from official eCFR title-structure data. Solver-facing examples are counterfactual and anonymized: source title numbers, section numbers, headings, URLs, and identifiers are removed. The benchmark evaluates legal/regulatory graph reasoning and constrained program generation, not source lookup.

Task
Submit predicted_program, a JSON object with exactly one key, ops. ops is a list of zero to five patch operations. Do not include any keys other than ops; auxiliary keys such as explanations, confidence, comments, or metadata make the row invalid.

Allowed operations:

{"op":"SET_UNIT","citation":"C1","target":"N4"}

{"op":"SET_RANGE","citation":"C2","start":"N3","end":"N7"}

{"op":"ADD_TARGET","citation":"C3","target":"N8"}

{"op":"DROP_TARGET","citation":"C3","target":"N9"}

{"op":"SET_SCOPE","citation":"C3","scope":"N1"}

All citation aliases and node aliases must appear in the row. The amendment note does not directly state the final target. It gives a broad repair pressure, such as range reconciliation, exception-list pruning, or scope-level consolidation. The exact repair must be inferred from the training examples using local hierarchy order, current citation state, node tags, and size buckets.

Files
train.csv columns:

| column | type | description |
|---|---|---|
| `id` | string | Row identifier. |
| `node_cards` | JSON list | Local hierarchy nodes with row-local aliases, parent links, ordinals, coarse heading tags, and size buckets. |
| `citation_cards` | JSON list | Initial citation objects before repair. |
| `amendment_note` | string | Noisy amendment note naming a citation alias and broad repair pressure. |
| `max_ops` | integer | Maximum allowed submitted operations. Always 5 in this release. |
| `target_program` | JSON object | Training-only canonical patch program. |

test.csv has the same columns except target_program.

sample_submission.csv has the required submission columns and uses an empty dummy program for every test ID.

Example row
A row may contain node cards like this:

[
  {"node":"N1","kind":"scope","parent":"","ordinal":0,"heading_tags":["definition","requirement"],"size_bucket":"scope"},
  {"node":"N4","kind":"section","parent":"N1","ordinal":1,"heading_tags":["definition"],"size_bucket":"short"},
  {"node":"N2","kind":"section","parent":"N1","ordinal":2,"heading_tags":["len8"],"size_bucket":"medium"},
  {"node":"N7","kind":"section","parent":"N1","ordinal":3,"heading_tags":["requirement"],"size_bucket":"short"}
]

and citation cards like this:

[
  {"citation":"C2","source":"N4","form":"unit","targets":["N2"]},
  {"citation":"C5","source":"N7","form":"scope","targets":["N1"]}
]

If the amendment note says:

Citation C2 is flagged for nearby unit substitution; anchor N4; repair only that citation.

then a syntactically valid non-trivial prediction could be:

{"ops":[{"op":"SET_UNIT","citation":"C2","target":"N2"}]}

This example is illustrative. The true post-amendment repair must be learned from the training rows.

Input structures
Each node_cards item has:

| field | type | description |
|---|---|---|
| `node` | string | Row-local node alias such as `N4`. |
| `kind` | string | `scope` for the containing local scope or `section` for an active local provision. |
| `parent` | string | Parent alias, or empty string for the scope root. |
| `ordinal` | integer | Local order within the scope. Ordinals are meaningful; alias numbers are not. |
| `heading_tags` | list[string] | Coarse tags derived from source headings, such as `definition`, `requirement`, or length-based tags like `len8`. |
| `size_bucket` | string | One of `scope`, `short`, `medium`, `long`, or `very_long`. |

Each citation_cards item has:

| field | type | description |
|---|---|---|
| `citation` | string | Row-local citation alias such as `C2`. |
| `source` | string | Node containing the citation. |
| `form` | string | Citation shape: `unit`, `range`, `list`, or `scope`. |
| `targets` | list[string] | Node aliases currently targeted by the citation. A `range` uses two targets: start and end. |

target_program, present only in train.csv, has this structure:

{"ops":[{"op":"ADD_TARGET","citation":"C4","target":"N9"}]}

Hidden repair families and subgroup scoring
The generator creates five hidden repair families: unit retargeting, range retargeting, list expansion, list reduction, and scope promotion. These labels are deliberately not included in public train.csv or test.csv, because exposing them allows shallow shortcut rules. They are stored only in the private answer file so the grader can compute a worst-family robustness term.

You do not need to submit or predict the family label. It only affects the final score through subgroup robustness.

Evaluation
The grader parses predicted_program, applies the operations to the row's initial citation cards, and scores the hidden scored citation. All non-scored citation cards are also checked for collateral damage: after execution, every non-scored citation must remain structurally identical to its initial state, including citation alias, source alias, form, and ordered target list. If a submission changes any non-scored citation, that row scores 0.

Triple construction for SemanticF1
For the hidden scored citation, the grader converts the repaired citation into triples:

one form triple: (citation, "FORM", form);
one target-position triple for each target: (citation, position_index, target_alias).
For example:

{"citation":"C2","form":"range","targets":["N3","N7"]}

becomes:

(C2, FORM, range)
(C2, 0, N3)
(C2, 1, N7)

SemanticF1 is the standard F1 score between predicted triples and hidden final triples for the scored citation.

For each row:

ExactFinal = 1 if the repaired scored citation exactly matches the hidden final citation, else 0. The comparison is ordered: target lists are compared directly, not as unordered sets.

ExactOps = 1 if the submitted operation list equals the canonical operation list, else 0.

Efficiency = min(1, optimal_op_count / submitted_op_count) when ExactFinal = 1; otherwise 0.

row_score = 0.25 * SemanticF1 + 0.55 * ExactFinal + 0.10 * ExactOps + 0.10 * Efficiency

Overall:

tail_mean = mean of the lowest-scoring 25% of row_score values

final_score = 0.70 * mean(row_score) + 0.20 * worst_family_mean + 0.10 * tail_mean

worst_family_mean is the lowest mean row score among the five hidden repair families listed above. This prevents solutions from ignoring one type of repair.

The tail_mean term separates brittle solutions from robust ones: a solver that gets the same average score but collapses on many hard rows receives a lower final score.

Invalid JSON, auxiliary top-level keys, unknown aliases, malformed operations, duplicated operations, empty programs, collateral edits to non-scored citations, or more than five operations score zero for that row. Structural submission errors such as missing IDs, duplicate IDs, unknown IDs, extra columns, or wrong column order are rejected.

What not to use / prohibited shortcuts
Do not use or assume any of the following:

Raw eCFR title numbers, CFR section numbers, source identifiers, source URLs, or source headings. These are intentionally absent from solver-facing rows.
Lookup against the raw source files to recover answers. Test rows are anonymized counterfactual citation-repair instances, not real eCFR citation records.
Hardcoded mappings from row IDs, row order, fixed citation aliases, or fixed node aliases. IDs and aliases are row-local and not semantically meaningful.
The private repair-family label. Repair families are not present in public train or test files and are used only for hidden subgroup scoring.
Editing unrelated citation cards to gain partial credit. The grader compares all non-scored citations against their initial state after execution; any collateral citation change gives that row a score of 0.
Submissions with extra columns, missing IDs, duplicate IDs, unknown IDs, more than five operations, invented aliases, or operations outside the allowed grammar.
A strong solution should learn how amendment pressure, local hierarchy order, citation form, current targets, node tags, and size buckets interact.

Submission format
The submission must have exactly two columns in this order:

| column | type |
|---|---|
| `id` | string |
| `predicted_program` | stringified JSON |

Non-trivial example:

id,predicted_program
0a12bc34de56f789,"{""ops"":[{""op"":""SET_UNIT"",""citation"":""C2"",""target"":""N4""}]}"

The sample submission uses empty programs and is expected to score 0.0.

Benchmark boundary
Unlike legal classification, legal QA, or citation retrieval benchmarks, this task requires executable graph edits under noisy amendment pressure and local regulatory hierarchy. Unlike direct legal cross-reference extraction, the target citations are counterfactual and anonymized; direct lookup of public regulation text cannot recover hidden programs.