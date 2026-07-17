Polyphonic Vocal Passage Event Recovery — solution notes
=========================================================

APPROACH
--------
The solution separates the structured output into rhythm, pitch/rest, and tie
recovery while decoding one test passage at a time.

1. Rhythm generation: a train-fitted variable-order duration model performs
   k-best dynamic programming over contiguous duration sequences. The decoder
   enforces missing_event_count and, where possible, tiles tick 0 through the
   first target_after onset. It returns ten candidates using train-only
   duration transitions plus the row's own meter, chord grid, and repetitions.
2. Rhythm selection: a CatBoost QuerySoftMax ranker learns to distinguish the
   true training rhythm from hard alternatives produced by that DP. Its score
   is fused with the DP score and the best independent pitch-path score.
3. Pitch/rest: an independent CatBoost query ranker scores each candidate pitch
   and rest. A second conditional query ranker is trained with the preceding
   event pitch and supplies a learned transition-emission matrix. Viterbi
   combines the conditional scores with twice the independent emissions.
4. Boundary ties constrain the first or last pitch when the visible context
   starts or ends a tie. Serialization emits exactly missing_event_count
   ordered events with the supplied voice token.

The implementation is CPU-only, self-contained, deterministic, and trains from
train.csv on every run. The measured end-to-end local runtime was 2,362.85
seconds for 2,985 training rows and 1,605 test rows.

MODEL / ALGORITHM
-----------------
Three CatBoostRanker models use QuerySoftMax loss:

  independent pitch: 550 iterations, depth 8, learning rate 0.07, seed 127
  conditional pitch: 450 iterations, depth 8, learning rate 0.07, seed 223
  rhythm reranker:    250 iterations, depth 8, learning rate 0.07, seed 219

All use CPU only, 8 threads, l2_leaf_reg 5.0, random_strength 0.3, and disabled
artifact writing.

Each labeled missing event is a pitch query. Candidates cover every semitone
within eight semitones of the surrounding vocal range plus rest; the correct
candidate is positive. The independent and conditional models share the same
musical features. Conditional training adds previous-pitch/rest and interval
features using the preceding true training event. During inference those scores
are evaluated for every previous/current candidate pair and decoded jointly.

Each contiguous training passage is also a rhythm query. Its alternatives are
the top 24 candidates from the train-fitted symbolic DP plus the true sequence.
The fitted rhythm model reranks the ten candidates retained at inference.

The candidate matrices are allocated directly as float32 and released after
fitting. No model consumes row IDs or voice-token identity.

FEATURE ENGINEERING
-------------------
CatBoost candidate features are transposition-aware and include:

- all six before and six after events: relative onset, duration, pitch/rest,
  and tie state;
- missing-event count, gap span, event index, candidate onset and duration;
- candidate distance from the last before pitch, first after pitch, and their
  linear interpolation across the gap;
- candidate membership in surrounding context pitches and distances to those
  pitches;
- sounding chord at event onset, event end, and the tick immediately before
  the end: exact pitch, pitch-class membership, top/bass distance, and nearest
  chord tone;
- chord-onset distances, candidate endpoints on the accompaniment grid, common
  metrical residues, and context events at lags 2, 3, 4, 6, 8, 12, 16, and 24;
- per-row key estimate, voice range, rest flag, and local repetition features.
- rhythm-candidate durations/onsets, duration changes, context-sequence
  agreement, chord-grid boundary distances, and lagged rhythm matches;
- conditional previous-pitch/rest state, interval size/direction, repeat,
  step, leap, and pitch-class-repeat indicators.

No row identifier or voice-token identity is a model feature. voice_token is
copied only into the required output record.

VALIDATION STRATEGY AND RESULT
------------------------------
Validation uses train.csv only. Rows are shuffled with Python random seed 42;
fold 1 (every fifth shuffled row starting at index 1) is held out: 2,388 train
and 597 validation rows. The official maximum-matching/LCS metric is
reimplemented exactly. Every duration table and all three CatBoost rankers are
refit on the 2,388-row partition.

Observed result for the final 550/450/250-tree pipeline:

  row_score       0.200244
  exact_sequence  0.072027
  event_F1        0.324503
  edit_similarity 0.271281

The previous symbolic decoder scored 0.1518 on this fold. The earlier submitted
independent-ranker pipeline scored 0.1819 on the leaderboard. The enhanced
validation gain comes from learned rhythm reranking and conditional pitch
decoding; leaderboard performance remains unseen until submission.

The challenge allocation holds out composers, but composer labels are not
provided. This random-row validation may share a voice source across train and
validation and is therefore optimistic. The model omits voice identity and
uses transposition-relative musical features to reduce that generalization
risk.

LEAKAGE AND COMPLIANCE STATEMENT
--------------------------------
All model fitting uses train.csv only. Test data is inference-only.

The complete test-data touch list in solution.py is:

1. pd.read_csv(public_dir / 'test.csv');
2. parse_rows(test, with_answer=False), which parses each row independently;
3. pitch_ranker.score_rhythms(row), rhythm_ranker.predict(row), and
   pitch_ranker.predict_conditional(row) for that same row;
4. JSON serialization and writing to submission_out.

Per-row key, meter, chord, range, and interpolation features use only that
single row's supplied score context. They are production-style transforms, not
cross-test statistics. No test rows are counted, fitted, calibrated, clustered,
joined, or used to select a feature, threshold, vocabulary, or hyperparameter.
There is no train+test concatenation, pseudo-labeling, test-time adaptation, or
cross-row test aggregation. Unknown test passages require no fitted categorical
vocabulary.

Random seeds are fixed. CatBoost artifact writing is disabled. The script reads
only train.csv and test.csv beneath public_dir and writes only submission_out.
It imports no local project files and uses no network, external data, pretrained
challenge checkpoint, GPU, Metal, CUDA, or package installation.

WHAT WORKED / WHAT DID NOT
--------------------------
Worked:

- candidate ranking rather than direct absolute-pitch classification;
- transposition-relative pitch and explicit chord-candidate relations;
- a conditional pitch ranker plus Viterbi sequence decoding;
- learned reranking of hard rhythm-DP alternatives;
- event-count-specific fusion selected on train-only validation;
- deterministic boundary tie constraints;
- keeping test processing strictly per row.

Did not improve validation:

- replacing the symbolic rhythm candidate generator with independent
  per-position duration predictions reduced exact rhythm recovery;
- an unconstrained absolute-pitch multiclass model underperformed ranking;
- a second deep independent pitch-ranker ensemble added substantial runtime
  without improving validation;
- an internal-tie classifier slightly reduced the official score at every
  useful threshold, so only visible boundary ties are emitted;
- expanding inference from ten to 24 rhythm candidates added only about
  0.0005 validation score and was not worth the runtime.

PRE-SUBMIT AUDIT
----------------
- Full command completed: python3 solution.py dataset/public working/submission.csv
- Output: 1,605 rows; columns exactly id and answer_json.
- IDs: unique and exactly equal to the test ID set.
- Every answer parses as JSON, has exactly missing_event_count events, uses the
  required fields/types/tie values/voice token, and is ordered by onset.
- No train+test concatenation or test-derived fitted state exists.
- solution.py is the only Python source file in the challenge directory.
