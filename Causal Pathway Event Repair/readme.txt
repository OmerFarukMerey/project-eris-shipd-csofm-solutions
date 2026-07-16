Causal Pathway Event Repair solution
====================================

Run
---
From this directory:

    python solution.py

The script reads dataset/public/train.csv and dataset/public/test.csv and writes
working/submission.csv. It uses at most 10 CatBoost CPU threads and does not use
a GPU, network access, external pathway records, public-ID lookup, candidate
position, or row order as a predictive feature. The measured runtime on the
provided 1,800/600-row release was 35.4 minutes.

Approach
--------
The solution treats every candidate card as a possible replacement node and
learns a supervised candidate score. It jointly learns the auxiliary targets
rather than reducing the task to event classification:

1. A base candidate ranker scores each event card from the visible typed causal
   graph, the candidate's entity descriptors, and local causal continuity.
2. A participant-role model learns input/output/catalyst/activator/inhibitor
   probabilities. Its role probabilities are fed back into a second candidate
   ranker, so candidate selection can distinguish forward causal continuity
   from descriptor overlap in the wrong direction.
3. A contextual multi-label role head emits one or more roles for every alias in
   the selected card. Deterministic constrained decoding only enforces schema
   validity and full participant coverage.
4. A regulation head predicts activation, inhibition, or neutral using the
   selected event and predicted role distribution.
5. A separate ambiguity head predicts genuine branch ambiguity and emits the
   exact ABSTAIN bundle at a cross-validated threshold.
6. A final MAE regression head estimates normalized task correctness for the
   confidence column, matching the calibration term in the metric.

Model architecture / algorithm
------------------------------
All learned heads use compact CatBoost models on CPU:

- Four-fold cross-fitted binary classifiers for base and role-aware candidate
  scoring.
- A multi-label MultiLogloss participant-role classifier.
- A three-class signed-regulation classifier.
- A binary abstention classifier.
- A robust MAE regressor for confidence calibration.

Four deterministic pathway-grouped folds keep rows with substantially
overlapping candidate-card pools together while preserving event types, branch
topology, missing-node degree, and abstention prevalence. Out-of-fold
predictions are used for every stacked training feature. Test predictions are
averages of the four fold models. The role-aware second stage receives the base
candidate score as an input; no candidate ID or list index enters either model.

Feature engineering
-------------------
The serialized JSON is parsed into typed entities, events, roles, and directed
links. Features include:

- Missing-node in/out degree, graph position, branch/merge degrees, neighboring
  event types, compartments, regulation, and two-hop ancestor/descendant data.
- Candidate event type, compartment, participant count, and multisets of entity
  type, compartment, state class, and component bin.
- Multiset matches between candidate entities and predecessor outputs,
  successor inputs, catalysts, regulators, and reverse-direction controls at
  several descriptor resolutions.
- Candidate-card overlap graph degree, weighted degree, component size,
  triangles, same-type overlap, and neighboring event-type counts.
- Alias recurrence across cards, without using the lexical or numeric value of
  an alias.
- Learned participant-role probabilities and role-weighted forward-versus-
  reverse causal continuity.
- Regulation and confidence features derived only from cross-fitted model
  probabilities and decoded structured outputs.

Role decoding allows multiple roles for one entity. Thresholds were selected
on out-of-fold set F1. If every role probability for an alias is below its
threshold, the highest-probability role is emitted, guaranteeing that the
selected candidate's aliases are covered exactly.

Validation strategy
-------------------
Validation is pathway-isolated four-fold stratified group cross-validation.
Rows sharing four or more coarse event-card signatures are assigned to the same
fold. Stratification preserves branching, missing-node in/out degree,
abstention, and selected event type. Every stacked feature used to score a
training row is produced by a model that did not fit its related-fragment group.

The local scorer reproduces the published metric: coupled event/role repair,
regulation, abstention, calibration to partial correctness, and full-row
consistency. The final reproducible run reported:

- non-ABSTAIN candidate accuracy: 0.5935
- abstention accuracy: 0.9517
- estimated complete structured score: 0.6669

The generated submission was additionally checked for exact column order, exact
ID order, row count, candidate membership, complete alias coverage, allowed
roles, consistent ABSTAIN bundles, allowed regulation values, and finite
confidence in [0,1].

What worked
-----------
- Exact typed multiset continuity from predecessor outputs to candidate inputs
  and candidate outputs to successor inputs was the strongest candidate signal.
- Cross-card alias recurrence and card-overlap graph structure broke many ties
  left by coarse entity descriptors.
- Feeding learned role probabilities into the candidate ranker improved local
  candidate accuracy over the base graph ranker.
- Combining intrinsic role templates learned from all visible known events with
  context-specific role predictions improved role-set F1.
- A dedicated ambiguity model was substantially better than treating every
  branch as ambiguous.
- Direct MAE calibration against normalized partial task correctness improved
  the metric over candidate-probability confidence.

What did not work
-----------------
- Pure causal-overlap scoring produced many ties because coarse descriptors are
  intentionally shared by several real candidates.
- CatBoost pairwise ranking objectives and LightGBM rankers underperformed the
  grouped binary CatBoost candidate model in local validation.
- Event-type-only role templates missed state transitions, catalysts, and
  regulated synthesis cases.
- Fixed neutral regulation was strong on frequency but missed the systematic
  activation/inhibition signal captured by predicted participant roles.
- A fixed rule such as "abstain at every visible branch end" produced too many
  false abstentions; learned graph and score-margin features were safer.
