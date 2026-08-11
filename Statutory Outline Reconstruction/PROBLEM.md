A United States public law is not a flat sequence of paragraphs. It is a tree. A section states an operative rule, a subsection carves out a condition, a paragraph enumerates an item in a list, a subparagraph splits that item, a clause qualifies a single term. Congress builds that tree as it drafts, and the designators printed down the left margin — Sec. 4., (a), (3), (B), (iv) — are how a reader sees it.

You receive an act with the tree taken away: its provisions as prose, in document order, designators removed. XML nesting, indentation, whitespace layout, headings, and empty designator remnants are not participant features. Rebuild the tree.

For each provision you name its parent — the provision it nests inside — by index, or -1 if it sits at the act's top level. Getting one provision right means deciding what it is subordinate to, and that decision constrains its neighbours, so the act has to come out consistent as a whole rather than one line at a time.

Data
Three public files, all derived from official U.S. Statutes at Large XML. One row is one act.

train.csv has three columns. act_id identifies the public law, provisions_json is a JSON array of the act's provisions as strings in document order, and parents_json is a JSON array of the same length holding the answer.
test.csv has act_id and provisions_json. The parents_json column is withheld.
sample_submission.csv shows the required columns with every provision chained to the one before it. It is a formatting example, not a prediction; it is structurally valid and obviously wrong.
Hidden acts are public laws that appear nowhere in train.csv. Provisions of one act share vocabulary and drafting style, so the split is by act rather than by provision.

Acts hold between 4 and 80 provisions, with a median of 14.

act_id values are arbitrary act_###### identifiers assigned after a deterministic hash shuffle. They do not encode the source volume, date, public-law number, or source-file order; the original provenance mapping is kept outside participant data.

Each provision string contains only that provision's own whitespace-normalised prose, capped at 1,200 characters. During preparation, recursively copied child text and long truncated child prefixes are removed, orphaned designators and sub-six-word stubs are pruned, children of a pruned stub are reattached to the nearest surviving ancestor, and the build fails if meaningful descendant text remains in a parent. Ordinary statutory cross-references inside a provision are retained because they are legitimate language signals.

Submission Format
Submit one CSV with exactly the two columns below, in either order.

act_id must match one test row. Every test act must appear exactly once.
parents_json must be a JSON array of integers, the same length as that act's provision list. Entry i is the index of provision i's parent, or -1 for a top-level provision.
A parent must be an earlier provision: entry i must be -1 or an integer from 0 to i - 1. This makes the tree well-founded and leaves no cycle representable.

Example:

act_id,parents_json
act_000123,"[-1,0,1,1,0,-1,5]"
act_000456,"[-1,0,0,0,-1,4,4]"

Evaluation
Higher is better, and the score is bounded between 0.0 and 1.0.

Each act is scored on three views of the tree, then normalised against that act's own degenerate answer.

The three components are:

Parent link accuracy, weight 0.55. The share of provisions attached to the correct parent.
Depth agreement, weight 0.25. Each provision's depth follows from the parent links. This component is 1 - |predicted_depth - true_depth| / span averaged over provisions and floored at zero, where span = max(maximum true depth, 1). The reference tree alone fixes this denominator; a contestant cannot improve other depth errors by predicting an artificially deep chain.
Sibling grouping, weight 0.20. Pair-counting F1 over provisions that share a parent. An act broken into the right groups under the wrong parents still earns this component.
Those three give a raw score. The reported score for the act is then

score = max(0, (raw - trivial) / (1 - trivial))

where trivial is the same raw formula computed for the submission that declares every provision top-level, on that same act.

This normalisation is the point of the metric. Declaring no structure at all scores exactly 0.0 on every act, by construction rather than by a chance constant measured on the hidden set. Reproducing the act's tree exactly scores 1.0. Partial structure earns the fraction of the available headroom it actually recovered.

The final score is the mean over hidden acts.

For reference, the supplied chain-format example scores about 0.019 under this corrected metric and prepared dataset.

The grader fails closed. An unreadable file, missing or extra columns, missing or extra rows, duplicate act ids, or ids that do not match the test set all produce 0.0 overall. A single row whose parents_json is unparseable, the wrong length, or not a well-founded forest scores 0.0 for that act alone and does not void the rest of the submission. Row order does not affect scoring.

Runtime And Environment
Solutions run in a Kaggle-style Python Docker environment. Read released files from the platform-mounted data directory, write ./working/submission.csv, use only permitted preinstalled libraries, and finish within 90 minutes on 10 CPU cores and 62 GB RAM.

Rules
Do not call external APIs, hosted models, or network services at inference time.
Do not use hardcoded lookup tables keyed on test ids, row order, or file hashes.
Do not attempt to retrieve the original Statutes at Large source, or any other publication of these acts, in order to look up the hidden trees.
Do not manually annotate hidden test acts.
Do not reverse engineer the grader or private answer files.
Predictions must derive only from released participant files and permitted bundled dependencies.
General-purpose pretrained weights or embeddings already bundled in the permitted environment are allowed. External corpora, downloaded weights, and models or indexes specifically built from the Statutes at Large, USLM, or the source acts are not allowed.
 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.