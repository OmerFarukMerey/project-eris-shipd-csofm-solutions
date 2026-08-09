Recommend a six-sensor polling portfolio from 12 case-local candidates. For each case, also identify four candidate pairs that would provide redundant evidence and return a complete priority ranking of the 12 candidates.

Each input is a fixed evidence packet recorded before the downstream recovery measurements become available. Long-running electronic-nose systems cannot continuously poll every element at maximum duty cycle, so a controller must recommend which candidates deserve a limited follow-up budget. Labeled historical cases reveal how visible response evidence relates to later usefulness.

The six-sensor recommendation carries most of the score and is evaluated by hidden downstream utility, not only by equality with one reference mask. A different portfolio can receive strong credit when it covers difficult recovery modes while avoiding redundant candidates. The graph and ranking outputs test whether the recommendation is supported by coherent evidence.

Predict these three fields for every case:

| Output | Required prediction |
|---|---|
| `polling_mask` | A 12-character mask selecting exactly six sensors with `K`. |
| `recovery_collision_set` | Four sensor pairs expected to have highly redundant recovery trajectories. |
| `clearance_order` | A priority permutation of all 12 aliases, ordered by expected recovery clearance. |

The underlying measurements are real. Every case uses one recording and one disjoint bank of 12 sensor channels. All cases derived from the same acquisition day remain in one split, and the local aliases S01 through S12 are independently assigned within each case.

Dataset
Files
| Path | Description |
|---|---|
| `train.csv` | 2,780 labeled cases with packet paths and the three target fields. |
| `test.csv` | 720 unlabeled cases containing only public inputs. |
| `sample_submission.csv` | A schema-valid baseline containing every test ID. |
| `sensor_packets/` | 3,500 Float16 NumPy arrays referenced by `sensor_packet_path`. |

CSV Columns
| Column | Data type | Train | Test | Description |
|---|---|---:|---:|---|
| `case_id` | string | yes | yes | Opaque identifier with no source, date, analyte, or sensor-family meaning. |
| `sensor_packet_path` | path string | yes | yes | Relative path to one `.npy` packet under `sensor_packets/`. |
| `polling_mask` | fixed-length string | yes | no | Canonical six-sensor portfolio used as the labeled reference. |
| `recovery_collision_set` | token-set string | yes | no | Four canonical undirected edges between local sensor aliases. |
| `clearance_order` | ordered token string | yes | no | All 12 aliases ordered from early to late recovery. |

train.csv contains the two input columns followed by all three targets. test.csv contains only case_id and sensor_packet_path.

Sensor Packet Layout
Each packet has shape 12 x 3 x 64.

| Axis | Meaning |
|---|---|
| first axis | Local sensors S01 through S12 in submission order. |
| second axis, channel 0 | Baseline-normalized log-resistance response. |
| second axis, channel 1 | First difference of the normalized response. |
| second axis, channel 2 | Phase marker, `-1` for baseline and `1` for exposure. |
| final axis, positions 0 through 23 | Resampled end of the baseline phase. |
| final axis, positions 24 through 63 | Resampled exposure phase. |

The post-exposure recovery samples are not present in the public packet. Per-sensor gain variation and mild measurement noise are applied after normalization. Each source recording contributes five cases whose 12-channel banks do not overlap.

Target Grammar
polling_mask contains exactly 12 characters from K and .. Position 1 represents S01, position 12 represents S12, K selects a sensor, and exactly six positions must be K.

recovery_collision_set contains exactly four unique edges joined by |. An edge uses Sxx~Syy, with the lower alias first. Edge tokens must be in ascending lexical order.

clearance_order contains each alias S01 through S12 exactly once, joined by >.

Example labeled record from train.csv:

| Field | Example value |
|---|---|
| `case_id` | `rb_0020bace574f2262e1c932` |
| `sensor_packet_path` | `sensor_packets/e239dfd530101630e8167ff5fb.npy` |
| `polling_mask` | `.K..KKK.K.K.` |
| `recovery_collision_set` | `S03~S08\|S03~S12\|S08~S12\|S09~S11` |
| `clearance_order` | `S10>S12>S03>S08>S01>S06>S04>S05>S07>S09>S11>S02` |

Training Target Distribution
Each polling position is selected in roughly half of the training portfolios.

| Alias | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 | S11 | S12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected rows | 1,311 | 1,270 | 1,283 | 1,293 | 1,373 | 1,362 | 1,411 | 1,402 | 1,427 | 1,497 | 1,492 | 1,559 |

All 66 possible undirected alias pairs occur in the training collision targets. Individual edge counts range from 136 to 209. The 2,780 training rows contain 859 distinct optimal polling masks and 2,726 distinct collision sets; every training clearance order is unique.

Evaluation
The metric is the Polling Portfolio Recommendation Score. Its complete aggregation formula is Score = 0.55 x PortfolioUtilityScore + 0.25 x CollisionGraphScore + 0.20 x ClearanceOrderScore.

PortfolioUtilityScore
For a submitted six-sensor set A, the hidden recovery data provides:

a risk value r_i for each sensor,
a four-value recovery descriptor d_i,
a pairwise recovery similarity s_ij.
For a submitted set A, define base(A) as the mean r_i over selected sensors. Define coverage(A) as the mean across the four descriptor columns of the largest selected d_i,k in each column. Define redundancy(A) as the mean s_i,j over the 15 unordered pairs inside the six-sensor set. The utility formula is utility(A) = 0.55 x base(A) + 0.30 x coverage(A) + 0.15 x (1 - redundancy(A)).

Let u_min and u_max be the minimum and maximum utility among all 924 valid six-sensor portfolios for that case. Compute normalized utility = clip((utility(A) - u_min) / (u_max - u_min), 0, 1). The case score is PortfolioRowScore = normalized utility cubed.

PortfolioUtilityScore is the mean row score. The cubic term rewards portfolios close to the hidden optimum while still allowing more than one useful answer. A malformed mask or a mask selecting a number other than six receives 0 for this component.

CollisionGraphScore
Let T be the four true undirected edges and P the four submitted edges. Because both valid sets contain four edges, edge F1 = 2 x |T intersection P| / (|T| + |P|). The case formula is CollisionRowScore = 0.85 x edge F1 + 0.15 x exact set match.

CollisionGraphScore is the mean row score. A malformed or noncanonical edge set receives 0 for that case.

ClearanceOrderScore
For one case, compare all 66 unordered pairs of aliases. pairwise_agreement is the number of pairs whose submitted relative order matches the answer, divided by 66. The case formula is ClearanceOrderRowScore = 0.90 x pairwise agreement + 0.10 x exact order match.

ClearanceOrderScore is the mean row score. A malformed order receives 0 for that case.

Minimum score: 0.0
Maximum score: 1.0
Higher scores are better.

Submission Format
Write the final submission to exactly ./working/submission.csv.

The CSV must contain exactly four columns in this order: case_id, polling_mask, recovery_collision_set, clearance_order.

| Column | Required format |
|---|---|
| `case_id` | Every test identifier exactly once, without whitespace padding. |
| `polling_mask` | Exactly 12 characters from `K` and `.`, with exactly six `K` values. |
| `recovery_collision_set` | Exactly four unique, canonical, sorted edges; maximum 160 characters. |
| `clearance_order` | A `>`-joined permutation of S01 through S12; maximum 100 characters. |

Two correctly serialized example rows are shown below. An actual submission must use every ID from test.csv exactly once.

| `case_id` | `polling_mask` | `recovery_collision_set` | `clearance_order` |
|---|---|---|---|
| `rb_0020bace574f2262e1c932` | `.K..KKK.K.K.` | `S03~S08\|S03~S12\|S08~S12\|S09~S11` | `S10>S12>S03>S08>S01>S06>S04>S05>S07>S09>S11>S02` |
| `rb_003ced12986c5b1008adc1` | `..KKK..KKK..` | `S01~S06\|S01~S07\|S02~S11\|S06~S07` | `S07>S01>S06>S11>S02>S12>S04>S09>S08>S10>S03>S05` |

A wrong column set or order, extra columns, duplicate IDs, unknown IDs, omitted IDs, or an incorrect row count rejects the submission. Malformed target values receive 0 for their corresponding component.

What Not To Use
Do not derive predictions from case IDs, packet filenames, byte sizes, hashes, row order, or source ordering.
Do not identify, retrieve, or match public packets against external copies of the original recordings, source filenames, analyte labels, acquisition dates, or precomputed source features.
Do not use exact or approximate lookup tables that map public packet fingerprints to target outputs.
Do not exploit hidden answer structure, grader behavior, malformed CSV handling, duplicate rows, or submission ordering.
Do not call hosted or closed-model APIs during inference. Local CPU signal processing and locally executed learned models are allowed.
 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.