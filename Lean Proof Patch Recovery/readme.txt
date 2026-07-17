Lean Proof Patch Recovery — solution writeup
=============================================

Task: given an anonymized broken Lean proof, compiler feedback, an approximate
failing line and a proof-state hint, predict a line patch: replace_start_line,
delete_line_count, and the insert_lines list.

Metric: 0.18*location + 0.58*text (normalized Levenshtein of joined insert
lines) + 0.14*line-LCS + 0.10*exact-patch, averaged over rows.


Key dataset facts discovered from train (all verified on train only)
--------------------------------------------------------------------
- delete_line_count = 1 in 99.4% of rows; insert_lines is ALWAYS exactly one
  line. So we always predict delete_line_count=1 and a single insert line.
- The insert line's indentation ALWAYS equals the replaced line's indentation.
- The replaced line is the last non-empty line of the snippet in 57% of rows,
  and is otherwise near the end; line_hint sits ~3-5 lines before the true
  replacement line.
- Aliases (x###, T###) are numbered by first appearance in the row
  (broken_proof, then feedback/state). Identifiers that the true fix
  introduces are therefore always the NEXT sequential numbers after the row's
  max context alias (~50% of insert aliases; the other ~50% come from the
  row's own context, mostly from the replaced line itself).
- The insert is usually a moderate edit of the replaced line (mean char
  similarity ~0.47) but never identical to it.


Approach / architecture
-----------------------
Four stages, all fit exclusively on train.csv:

1) Start-line model (location).
   A HistGradientBoostingClassifier scores every non-empty line of each
   snippet as "is this the replaced line". Features: offset from line_hint,
   offset from end / last non-empty line / the `:= by` line, indentation,
   line length, first-token id (exact/rw/simp/...), alias overlap with
   compiler feedback and state hint, neighboring-line token/indent context,
   feedback-category id (top-40 alias-normalized first lines of feedback),
   number of goal cases in feedback. Per row the top-2 scoring lines are kept
   as start candidates. Top-1 accuracy is ~74% (vs 57% for the
   last-non-empty heuristic); a top-3 diagnostic reaches ~95% recall.

2) Insert-line candidate generation, per (row, start-candidate):
   - copy of the line being replaced;
   - generic tactics at the same indentation: "simp", "rfl",
     "simp [<next new alias>]";
   - k-NN retrieval (k=12): char-TFIDF (2-4 grams) over the alias-normalized
     concatenation of replaced line + feedback head + state hint, cosine
     nearest train rows. Each neighbor's insert is used raw, x-remapped, and
     fully alias-remapped. The full remapper maps both x### and T### aliases,
     preserves deleted-line positional correspondence, and assigns one stable
     target alias to every repeated source alias. New aliases continue after
     the target row's train-independent context maximum;
   - slotted templates: train insert lines with aliases encoded as
     deleted-line-position slots / context slots / new-sequential slots
     (e.g. "rw [D0, N0]"); the 50 globally most frequent templates plus the
     8 most frequent per feedback category, instantiated with the target
     row's aliases.

3) Semantic tactic-family model.
   A multinomial LogisticRegression uses the train-fitted character TF-IDF
   representation to predict the repaired line's first-token family. Its
   candidate-specific probability and rank are ranker features. Ranker
   training uses five-fold out-of-fold probabilities, so each train row's
   semantic features come from classifiers that did not see its answer.

4) Joint (start x insert) ranker.
   A CPU CatBoostRegressor predicts each candidate pair's expected metric
   contribution
   (0.18*loc + 0.58*text_sim + 0.14*exact-line + 0.10*exact-line*correct-start).
   A lightweight HistGradientBoostingRegressor supplies a complementary 15%
   blend. Features include semantic tactic probabilities, candidate source
   flags, neighbor rank/distance/votes, template and insert frequencies,
   similarity to the replaced line and strongest remapped neighbor, trigram
   consensus, alias counts, feedback category, and start-line probability.
   Train-side neighbors exclude the row itself. The highest scoring pair over
   the top-2 start lines is submitted.

Validation strategy
-------------------
- Validation uses KFold splits built from train only (shuffle, seed 123).
  Each validation fold is treated exactly like test: every vectorizer,
  classifier, retrieval index, frequency table, and ranker is refit using
  only that fold's training partition.
- The semantic + CatBoost model scored 0.4307, 0.4192, and 0.4225 on fixed
  folds 0-2 (mean 0.4241).
- Second-round ablations used a conservative 300-tree CatBoost configuration.
  Full alias remapping plus the lightweight ranker blend scored 0.4312,
  0.4182, and 0.4235 on the same folds: mean 0.4243. Mean components were
  location 0.9321, text 0.4338, exact-line 0.0211, exact-patch 0.0195.
- Final training restores 800 CatBoost trees; the reduced configuration was
  used only to compare second-round variants on train-derived folds.

Leakage statement
-----------------
Every transform, statistic, and learned model (TF-IDF vectorizer, k-NN index,
template/frequency tables, feedback categories, tactic and start classifiers,
CatBoost ranker, and histogram ranker) is fit on train.csv only. Test rows are
used strictly for per-row transform + predict:
parsing that row's own text, applying the train-fitted vectorizer/models, and
retrieving neighbors from the train-only index. No statistic is computed
across test rows; there is no train+test concatenation anywhere; no
pseudo-labeling; no test-derived selection of features, thresholds or
hyperparameters. Random seeds are fixed (42 for models, 123 for final CV).
The script reads only the provided public dir and writes only the submission
path. Categories/aliases unseen in train fall back to id 0 / heuristic
defaults, never to test-fitted values.

What worked / what didn't
-------------------------
Worked:
- Exploiting the anonymization scheme: sequential "new" alias numbering and
  exact indentation transfer are deterministic structure the models exploit.
- Joint ranking of (start line, insert candidate) pairs against a target that
  mirrors the actual metric, instead of optimizing stages independently.
- Stable x/T alias remapping adds candidates that preserve repeated aliases
  and type aliases instead of accidentally copying a neighbor's vocabulary.
- A diverse candidate pool: copy / retrieval / templates each win a large
  share of rows (oracle over the pool ~0.58 text vs 0.39 for any single
  source).
- Out-of-fold tactic-family probabilities improved candidate ranking without
  leaking a train row's answer into its own ranker features.
- CPU CatBoost regression improved the fixed three-fold mean over the prior
  ranker while directly optimizing a target shaped like the challenge metric.
- A small histogram-ranker blend consistently improved the 300-tree CatBoost
  model on the checked folds while keeping the final runtime practical.
Didn't work / not used:
- Out-of-fold start-line candidates reduced fold-0 score from 0.4299 to
  0.4202; the final ranker therefore retains full-train start features.
- Deterministic deleted-template -> insert-template transform rules: the
  mapping is too diffuse (modal target share ~7%).
- Retrieval keyed on goal text or heavier feedback windows: slightly worse
  than the combined deleted-line + feedback + state-hint document.
- Whole-proof near-duplicate matching between train and test: essentially no
  duplicates exist (<0.5%).

Reproducibility
---------------
python3 solution.py <public_dir> <submission_out>
Pure CPU, deterministic; uses pandas, numpy, scikit-learn, and CatBoost.
RapidFuzz speeds up Levenshtein when installed; an exact numpy fallback is
included.
