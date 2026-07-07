GUI Widget Grounding and Interaction Prediction
================================================

Problem
-------
For each test screenshot + instruction pair, predict 6 columns: the target widget's
bounding box, the action type (click/type/select), the element role (text_field/tab/
button/link/widget), a blocker type describing interface uncertainty (none/covered_modal/
disabled/wrong_page/ambiguous), a canonical repair-step sequence, and the resulting
interaction status. Scored by a weighted average: box IoU+center (0.32), repair-sequence
edit distance (0.20), and four inverse-frequency-weighted macro-F1 categoricals (0.12
each: action_type, element_role, blocker_type, interaction_status). Sample submission
baseline: 0.116865.

Approach
--------
EDA on the 1900 labeled train rows first, to find out which columns are actually free
and which need real modeling:
  - repair_sequence and interaction_status are EXACT deterministic functions of
    (blocker_type, action_type) in all 1900 rows -- verified, not assumed. No model is
    fit for these two; they're a lookup applied after blocker_type/action_type are
    predicted (prefix by blocker_type: none->"", ambiguous->inspect_duplicates,
    covered_modal->dismiss_modal, disabled->enable_control, wrong_page->navigate_back;
    suffix is always {action_type}_target; status is a fixed 1:1 map of blocker_type).
  - target_box == [0,0,0,0] iff blocker_type == 'wrong_page', exactly, both directions.
  - action_type (~98%) and element_role (~99%) are recoverable straight from
    user_instruction text: three fixed wrapper phrases ("Type into the control described
    by this request:", "Click the widget described by this request:", "Select the option
    or control described by this request:") are 100% deterministic for action_type and
    cover ~70% of rows. The remaining "raw" instructions (bare labels, "click left button
    for X", etc.) are handled by a small TF-IDF(char n-gram)+LogisticRegression classifier
    trained on the public train split, which picks up recurring vocabulary a fixed regex
    can't (e.g. a literal "SELECT ALL" button label -> select, or a recurring blog-post
    title that is always typed rather than clicked in this dataset). element_role uses an
    ordered keyword regex (text field/checkbox/list/tab/link/button), plus one learned
    quirk: instructions mentioning "Tableau/Tableur" are always role=tab in this dataset
    even when they also contain the word "link".
  - ui_context_note and prior_action_trace (5 fixed templates each) show no measurable
    association with any target column (chi-square/Cramer's V ~0.06 on train) -- treated
    purely as decoys, not used as features.

That leaves blocker_type and target_box as the only two columns that genuinely require
looking at the screenshot. Both are solved with OCR-based grounding:
  - Tesseract (already installed on this machine via Homebrew) is run via subprocess on a
    grayscale + autocontrast + 2x-upscaled copy of each screenshot -- this preprocessing
    was necessary: default settings missed small, low-contrast form labels (verified on a
    concrete example) that the upscale+contrast pass reads at >90% OCR confidence.
    Word-level boxes are cached to working/ocr_cache.json.
  - The instruction's target label is isolated by stripping the wrapper/verb/role-word
    prefixes, then fuzzy-matched (difflib, accent-stripped) against sliding-window phrase
    candidates built from OCR lines (plus multi-line concatenation for wrapped labels).
  - target_box is NOT the raw matched OCR text box -- measured mean IoU between raw OCR
    box and the true box is only ~0.03-0.06. Instead, a per-element_role center-offset and
    width/height scale correction (fit from train rows with a confident OCR match) is
    applied, since e.g. a text field's true box sits offset to the right of its label, and
    a checkbox/radio icon sits just left of its label. Browser tabs get a further special
    case: matches landing in the top chrome band snap to that row's near-constant y/height
    (measured from train), since only x/width actually vary there.
  - blocker_type is a single RandomForestClassifier (not a hand-written rule cascade) over
    OCR-derived features (match score, runner-up score, duplicate-match cluster count) and
    classical pixel features: a connected-components blue-button detector (the
    covered_modal overlay in this dataset is a very consistent synthetic "Review required /
    Dismiss" panel with a solid blue button, which OCR usually fails to read at all because
    it's small, but which a plain blue-hue pixel mask finds near-perfectly) plus local vs.
    global saturation/edge-density/brightness/red-fraction stats (disabled controls in this
    dataset show up as desaturated+cross-hatched, redacted, or otherwise higher local edge
    density than a normal control). A single joint classifier was used deliberately after
    a first attempt at an isolated wrong_page-vs-rest first stage was measured to fail: OCR
    match quality is suppressed both when a target is genuinely absent (wrong_page) AND
    when it's merely hidden behind the modal (covered_modal), so those two overlap heavily
    on match-score alone and an isolated binary gate over-fired wrong_page across every
    other class. Letting one model see all the features together (so the blue-button
    signal can override a weak match score) fixed most of that.
  - target_box and blocker_type are graded as independent columns, so they don't have to
    agree: rather than always zeroing the box exactly when the argmax label says
    wrong_page, a separate threshold on the model's P(wrong_page) is tuned on out-of-fold
    predictions specifically to maximize OOF box score.

Local validation
-----------------
5-fold StratifiedKFold (on blocker_type) over the 1900 train rows, scored with a local,
from-scratch implementation of the exact grading formula (box IoU+center with the
[0,0,0,0] sentinel rule; token-level Levenshtein for repair_sequence; inverse-frequency-
weighted macro F1, weights capped at 5x the median and renormalized, for the four
categoricals). Since repair_sequence/interaction_status are deterministic given
blocker_type+action_type, the real story is 3 numbers: blocker_type accuracy, box IoU, and
action/role accuracy -- the run prints all of them, not just the aggregate, precisely so
iteration effort goes to blocker_type and target_box first (they carry 0.32 + 0.12 direct
weight, plus most of repair_sequence's 0.20 and all of interaction_status's 0.12).

Out-of-fold result on the public train split:
  target_box      (w=0.32): 0.3346
  repair_sequence (w=0.20): 0.6059
  action_type     (w=0.12): 0.9986
  element_role    (w=0.12): 0.9932
  blocker_type    (w=0.12): 0.5207
  interaction_status(w=0.12): 0.4826
  OVERALL WEIGHTED SCORE       : 0.5877
  (sample_submission.csv baseline: 0.116865)

blocker_type confusion is heavily concentrated on covered_modal, which the blue-button
detector separates almost perfectly (~98% recall); the remaining 4-way confusion between
none/ambiguous/disabled/wrong_page is the genuinely hard residual -- OCR match strength
alone can't cleanly tell "target absent from this page" apart from "target present but
poorly read by OCR", and no pretrained vision-language model was used to go further (see
Requirements). A no-OCR fallback path was also tested (Tesseract made unavailable) and
produces a complete, valid submission with only a modest drop (0.5839 local score) --
losing OCR mainly hurts target_box (falls back to a per-role positional prior), while
blocker_type is scored almost entirely from the pixel features either way.

Requirements
------------
numpy, pandas, scikit-learn, scipy, Pillow -- all already used elsewhere in this repo.
OCR grounding additionally shells out to the `tesseract` CLI binary (no pip OCR package
needed); if it isn't on PATH, OCR_AVAILABLE is detected at import time and the pipeline
falls back to positional priors for target_box and pixel-only features for blocker_type
instead of crashing. No pretrained deep vision/VLM weights are downloaded or used --
everything is fit from scratch on the public train split, per the task's stated
constraints (no lookups from sample_id/row-order/filenames, no matching against an
external copy of the source dataset).

If running on a laptop, keep it awake during the run (e.g. `caffeinate -s` on macOS) --
OCR itself only takes ~1-2 minutes over all 2286 images with an 8-thread pool, but the
rest of the pipeline (feature engineering + RandomForest training across 5 CV folds, then
a final refit) is another several minutes of CPU time; system sleep pauses the process
without failing it but can stretch wall-clock time considerably.

How to run
----------
    cd "GUI Widget Grounding and Interaction Prediction"
    python3 solution.py

First run builds working/ocr_cache.json (cached for fast re-runs). Writes
./working/submission.csv (386 rows: sample_id, target_box, action_type, element_role,
blocker_type, repair_sequence, interaction_status) and prints the local CV score
breakdown.
