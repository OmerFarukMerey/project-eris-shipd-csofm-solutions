Problem Description
Deadzone: Hidden-State Transduction from Paired Symbol Sequences
Overview
Each record in this challenge is a pair of parallel strings over a 254-symbol alphabet. Your job is to emit a third string, over a different 8-symbol alphabet, aligned one-to-one with the scored positions of the input.

That is the whole contract: two input strings in, one output string out. The actor_codes string and the foe_codes string describe two agents acting in the same episode; the states string you produce says which of eight behaviour classes the actor was in at each scored position.

This is a transduction problem, not a forecasting one. Nothing is extrapolated beyond the end of the string, and no value is predicted before it is observed. The entire input pair is handed to you at once and the output is a labelling of it, so both left and right context are fair game — the strong reference model below is bidirectional. There is no timestamp, no sampling rate, no clock, and no numeric measurement anywhere in the data: every field you receive is either a symbol string or a categorical code. The task is closest in shape to sequence tagging or transliteration, where an input string is rewritten into an aligned output string over a different alphabet.

What makes the rewriting hard is that it is not a symbol-to-symbol mapping. The alphabet was emitted by a hidden automaton, and the same input symbol licenses completely different output symbols depending on which state that automaton is in — a state you are never shown. The identical code yields atk from one state, air from another, and nothing at all from a third. So the mapping cannot be tabulated; the automaton has to be inferred from the parallel corpus in train.csv and then decoded, carrying the inferred state along the string.

The second difficulty is that the two input strings are coupled. Roughly 28% of the output symbols hit, evd, rsp) are states the actor did not choose at all: they are forced by what the other agent did, and whether that agent's action took effect depends on a hidden relationship between them that appears in neither string. No amount of modelling actor_codes alone will explain them. They are recoverable only by reading both strings jointly and inferring the interaction that must have produced the pair.

The symbols are opaque and salted. c000 does not tell you what it denotes, and nothing here reveals the mapping; which symbols mean "attack", which mean "block", and which mean "nothing at all" is itself something to learn from the training split. On top of that, 22% of the positions in every string are blanked to [X], so the evidence is lossy as well as opaque. Blanking removes evidence only — the output symbols are the source system's own recorded states, exactly as observed, never perturbed.

Provenance. The strings are real recordings: two people operating hand controllers through matches of a competitive video game, and the output symbols are the game engine's own internal action states for one of the two characters. Everything the engine displayed on screen — positions, velocities, damage, collisions — is withheld, which is precisely why the automaton's state must be inferred rather than looked up.

The intended first-order approach is to learn the association between short spans of surrounding symbols and the actor's class, then improve on it by decoding the pair as a sequence: propagating the inferred hidden state along the string, using the unscored prefix to initialise it, and reading the foe's symbols to explain the states the actor did not choose. A classifier that looks only at the symbol at the current position is measured below and scores poorly.

Evaluation
Pool the scored positions of every test record together and let C be the 8x8 confusion matrix over those positions, where C[i][j] counts the scored positions whose true class is i and whose predicted class is j.

Define:

N = total number of scored positions = the sum of all entries of C
c = the sum of C[i][i] over i — the number of correct positions
t_i = the sum of C[i][j] over j — how often class i is the truth
p_j = the sum of C[i][j] over i — how often class j is predicted
Term 1 — multiclass correlation. The classification term is the Matthews correlation coefficient generalised to 8 classes:

MCC = (c × N − S_tp) / ( sqrt(N² − S_pp) × sqrt(N² − S_tt) )
where

S_tp = the sum over i of t_i × p_i
S_pp = the sum over j of p_j × p_j
S_tt = the sum over i of t_i × t_i
If the denominator is 0, then MCC = 0. MCC is 1 for a perfect prediction, 0 for any constant prediction, and negative for anticorrelated predictions. It is chance-corrected, so predicting the majority class everywhere earns nothing.

Term 2 — boundary agreement. This term asks where the output string changes symbol. For one record with true classes y_1 ... y_n and predicted classes d_1 ... d_n, define at each position k from 2 to n:

true boundary B_k = 1 if y_k ≠ y_(k−1), else 0
predicted boundary D_k = 1 if d_k ≠ d_(k−1), else 0
Pooled over every scored position of every record:

TP = the sum of B_k × D_k
FP = the sum of (1 − B_k) × D_k
FN = the sum of B_k × (1 − D_k)
BoundaryF1 = 2 × TP / (2 × TP + FP + FN), and BoundaryF1 = 0 when TP = 0
Final score.

Score = 100 × clip(0.70 × MCC + 0.30 × BoundaryF1, 0, 1)
clip(x, 0, 1) returns 0 below 0, 1 above 1, and x otherwise. The minimum score is 0 and the maximum is 100, awarded only for reproducing every scored position of every record exactly. Higher is better.

The output string changes symbol at 4.8% of positions, so neither trivial boundary strategy pays: never predicting a change gives BoundaryF1 = 0, and predicting a change everywhere gives BoundaryF1 = 0.091.

Measured reference scores
All figures were produced by the published grader on the published test.csv, against the held-out key:

| Approach | Score |

|---|---|

| Perfect reconstruction | 100.00 |

| sample_submission.csv — the constant symbol gnd | 0.00 |

| Uniform random symbol per position | 2.71 |

| Constant most-frequent training symbol | 0.00 |

| Per-symbol lookup: each code to its most common class in train | 19.31 |

| Gradient-boosted trees over a span of surrounding symbols from both strings | 43.34 |

| Bidirectional recurrent sequence model over both strings | 58.95 |

Both constant baselines score exactly 0 because MCC is chance-corrected and a constant string has no boundaries.

Dataset
The public/ directory contains three files.

train.csv* — 8,826 rows, one row per record:

seq_id — integer, opaque identifier of the record; encodes nothing
arena — string, salted code for the setting the episode took place in
actor_kind — string, salted code for the actor's archetype
foe_kind — string, salted code for the foe's archetype
context_len — integer, always 60; the length of the unscored prefix, given as context only
n_scored — integer, the number of scored positions in this record; varies from 160 to 260
actor_codes — string, space-separated symbols for the actor, of length context_len + n_scored; each symbol is either c000 through c253 or the blanked marker [X]
foe_codes — string, space-separated symbols for the foe, same length and alphabet
states — string, space-separated ground-truth classes for the actor, of length n_scored
test.csv* — 3,174 rows, identical to train.csv but without the states column.

sample_submission.csv* — 3,174 rows, a valid but deliberately worthless submission in the exact required format.

Both input strings span all context_len + n_scored positions, but only the last n_scored are scored. The 60-symbol prefix exists purely to let you warm up your estimate of the hidden state before scoring begins.

The eight output classes are:

rsp — the actor has been knocked out: dead, respawning, or re-entering
gnd — grounded movement: standing, walking, dashing, running, turning, crouching, landing
air — airborne movement with no attack: rising, falling
atk — attacking: any ground attack, aerial attack, or special move, including its recovery lag
shd — blocking: raising, holding, or dropping a shield
grb — grappling: grabbing, holding, throwing, or being grabbed and thrown
hit — being struck, knocked back, or tumbling
evd — evasive and recovery states: rolls, dodges, teching, knockdown, get-up, and hanging on a ledge
Classes are not balanced; their frequencies are those of real play, from 28.3% for atk down to 3.6% for grb.

Submission
Submit a CSV with a header row and exactly 3,174 data rows — one per seq_id in test.csv.

seq_id — integer — the record identifier, copied from test.csv. Every id must appear exactly once.
states — string — space-separated class symbols for the actor, in order. Must contain exactly n_scored symbols, each one of rsp, gnd, air, atk, shd, grb, hit, evd.
Example, using real ids from test.csv — record 0 has n_scored = 184 and record 3 has n_scored = 183, so their strings are 184 and 183 symbols long respectively; both are truncated here with an ellipsis:


seq_id,states

0,air air air air air gnd gnd atk atk ...

3,atk atk atk hit hit hit evd evd gnd ...

Validity rules:

Row order does not matter, and extra columns are ignored.
Duplicate, missing, or unrecognised seq_id values are rejected, as is a wrong row count.
A row whose states value is empty, non-string, or does not contain exactly n_scored recognised class symbols is scored as maximally wrong for that record: every position is counted incorrect, and its boundaries are counted as both missed and falsely raised. An unparseable row is therefore never cheaper than a wrong one — there is no benefit to abstaining.
What Not to Use
A constant or majority-class string. MCC is chance-corrected, so every constant submission scores exactly 0, not the 28% you might expect from the frequency of atk.
A per-position classifier on the current symbol alone. This is the obvious shortcut and it is measured above at 19.31. It cannot work, because the symbol-to-class mapping is one-to-many by construction: the identical symbol yields atk, air, hit or nothing depending on hidden state you must carry along the string yourself.
Any approach that ignores foe_codes. Roughly 28% of output symbols are hit, evd or rsp — states the actor did not choose. They are forced by the foe, and no amount of modelling the actor's own string will explain them.
Prior knowledge of any particular device, layout, or archetype. The alphabet is salted and the archetype codes are anonymised. c000 is not something you already know. The mapping exists only in the training split.
Blanket smoothing as a substitute for state tracking. Blurring the output raises apparent accuracy but destroys BoundaryF1, which is 30% of the score and rewards placing each transition at the correct position.
What is expected instead: infer the hidden automaton from the training split, then decode each test record as a sequence — propagating the hidden state along the string, using the unscored prefix to initialise it, and reading both strings jointly to explain the symbols the actor did not choose.

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.