Biocatalytic Product Recommendation: Ranking Candidate Products by Enzyme-Reaction Relevance
Overview
You are building a learning-to-rank / recommendation model for enzymatic chemistry. Each record pairs
an enzyme-reaction context — a set of substrate molecules together with the catalyzing enzyme class
(its EC number) — with a candidate product molecule:

substrates: CCCCCCCCCCCCCCCCCCO.NC(=O)c1ccc[n+](...NAD+ cofactor...)c1 ec: 1.1.1.71 candidate: CCCCCCCCCCCCCCCCC=O
For each record you output a single relevance score in [0, 1]: how strongly this enzyme-context
supports candidate as its reaction product. Products the enzyme actually makes should be ranked at the
top (high relevance); products it does not make should sink to the bottom (low relevance). Here the
candidate is the context's real product (an alcohol dehydrogenase, EC 1.1.1.x, oxidizes the primary alcohol
of the first substrate to an aldehyde; the second substrate is the NAD+ cofactor), so it should rank high.
No molecule is generated — you rank/score candidates, you do not produce them.

The training data carries implicit relevance feedback: each enzyme-context appears paired with its
observed product (feedback 1) and with sampled negatives (feedback 0) — candidate products that
this specific enzyme-context does not make. You learn a ranker from this feedback and apply it to enzyme
contexts and products you have not seen.

The ec field is the four-level EC number (Enzyme Commission number, e.g. 1.1.1.71); its leading
digit is the top-level enzyme class (1 oxidoreductases, 2 transferases, 3 hydrolases, 4 lyases,
5 isomerases, 6 ligases, 7 translocases). The substrates field is a single dot-separated SMILES string
and includes any cofactors (NAD(P)H, ATP, coenzyme A, ...) the enzyme uses.

The sampled negatives are engineered to be informative: many are hard negatives — an enzyme's real
product paired with the wrong enzyme class from the same top-level family (same EC first digit) — so the
candidate is a chemically plausible product for that kind of enzyme and surface similarity between
candidate and substrates is uninformative; only modelling the specific enzyme's activity ranks them
correctly. A good ranker both orders observed products above negatives and reports well-calibrated
relevance. The intended apparatus is a trained learning-to-rank / recommendation model — for example a
fine-tuned chemical language model used as a cross-encoder, a graph neural network, or a gradient-boosted
model over reaction-difference features.

Data
All files live under public/ and share a single identifier column, id (present in train.csv,
test.csv, sample_submission.csv, and your submission). The dataset contains about 18,000 records in
total, split into 14,714 training records (train.csv) and 3,286 test records (test.csv); each
split is an even mix of observed products and sampled negatives.

Files provided:

train.csv — the 14,714 training records with feedback, one row per record (columns id, substrates,
ec, candidate, label).
test.csv — the 3,286 records to rank, one row per record (columns id, substrates, ec,
candidate; the label is withheld and is what your relevance score orders).
sample_submission.csv — an example submission in the exact required shape (columns id, score), one
row per test.csv id. Its score values are a weak character-similarity baseline (the overlap
between the candidate and substrate SMILES); it ranks only the easy negatives correctly and scores low
(see Scoring). Replace each score with your own relevance.
The columns present across the dataset, with their data type and meaning, are:

id — type str. A unique record identifier, an opaque string (e.g. rxv-16tvhnh8x). Present in
every file (train.csv, test.csv, sample_submission.csv, and your submission).
substrates — type str (a dot-separated SMILES string). The enzyme-context's substrate molecules
joined by . (e.g. CCO.O=O), including any cofactors. Present in train.csv and test.csv.
ec — type str. The four-level Enzyme Commission number d.d.d.d of the catalyzing enzyme (leading
digit = enzyme class 1–7). Together with substrates it defines the enzyme-context. Present in
train.csv and test.csv.
candidate — type str (a SMILES string). The candidate product molecule to score for relevance to
the enzyme-context. Present in train.csv and test.csv.
label — type int (0 or 1). The implicit relevance feedback for a training record: 1 if
candidate is the enzyme-context's observed product, 0 if it is a sampled negative. It is the
training signal you learn your ranker from — present in train.csv and withheld from test.csv
(where your relevance score orders the records).
Example train.csv rows (header + an observed-product and a sampled-negative record):

id,substrates,ec,candidate,label rxv-16tvhnh8x,CCO.O=O,1.1.3.13,CC=O,1 rxv-9a1b2c3d4,CCO.O=O,1.1.1.1,CC=O,0
(the second row is a hard negative: the same substrates and the same candidate aldehyde, but a different
oxidoreductase ec from the same family that does not produce it under these conditions — so it should be
ranked below the observed product.)

The test candidates are drawn from a product-disjoint pool: every observed product molecule in the test
set is absent from the training records, so you must rank enzyme-contexts and products you have never seen
in training rather than memorize them.

Submission format
Submit a CSV named submission.csv with exactly two columns, id and score, one row per test record:

Column	Type	Description
id	str	A test record id; copy the id values verbatim from test.csv (each test id must appear exactly once).
score	float	Your relevance score in [0, 1] for the record (how strongly the enzyme-context supports the candidate as its product). A missing or non-numeric value is treated as 0.5; values outside [0, 1] are clipped.
Include the header row id,score and exactly one row per test id. The submission must have exactly these
two columns, in that order (id then score) — any extra/missing column or different column order is
rejected. Concrete example (submission.csv, header + three rows):

id,score rxv-7c4e1a9d2,0.97 rxv-1b2c3d4e5,0.04 rxv-2c3d4e5f6,0.61
Scoring
This is a learning-to-rank problem: your relevance scores are judged on how well they order
observed products above sampled negatives and how well-calibrated they are, blended into a single
quality in [0, 1] and passed through a strict difficulty curve. Writing y_i in {0,1} for the relevance
feedback of record i and score_i for your relevance score:

1. Ranking quality rank. The fraction of (observed-product, negative) record pairs your scores order
correctly — the Mann–Whitney / ROC statistic, where 0.5 is a random ordering and 1 is a perfect one:

rank = clip( 2 * AUC(score, y) - 1 , 0 , 1 )
2. Calibration cal (a log-loss proper scoring rule). With s_i your relevance clamped to
[1e-9, 1-1e-9], the mean log-loss and its normalization against the uninformative constant-0.5 baseline
ln 2:

NLL = - mean over i of [ y_i * ln(s_i) + (1 - y_i) * ln(1 - s_i) ] cal = clip( 1 - NLL / ln(2) , 0 , 1 )
cal = 1 for perfectly-calibrated relevance (s_i = y_i), cal = 0 for scoring 0.5 everywhere, and a
confidently-wrong relevance (e.g. s = 0.99 on a negative) is penalized heavily — being sure and wrong
costs you.

3. Hard-negative ranking hard_rank. A subset of the negatives is hard: an observed product paired
with the wrong enzyme class from the same top-level EC family, so surface cues do not help. Restricting the
ranking statistic to observed products versus these hard negatives:

hard_rank = clip( 2 * AUC(score over {observed products} u {hard negatives}, y) - 1 , 0 , 1 )
4. Blend and difficulty shaping. The three parts are blended into a single quality, then mapped to the
final score by a strict-but-smooth curve:

quality = 0.45 * rank + 0.20 * cal + 0.35 * hard_rank score = 0.10 * quality + 0.25 * quality**2 + 0.65 * gate(quality)
where gate(q) is a smooth "near-perfect" curve built from the hyperbolic tangent, normalized so
gate(0) = 0 and gate(1) = 1:

gate(q) = ( tanh( 300 * (q - 0.995) ) - tanh( 300 * (0 - 0.995) ) ) / ( tanh( 300 * (1 - 0.995) ) - tanh( 300 * (0 - 0.995) ) )
What not to use
This is a learning-to-rank / recommendation challenge. Each record pairs an enzyme-reaction context
(substrates, ec) with a candidate product, and your solution outputs a single relevance score in
[0, 1] for how strongly the enzyme-context supports that candidate as its product. Your solution must be a
trained ranker / recommender that learns from the provided implicit relevance feedback (label 1 =
observed product, 0 = sampled negative) and generalizes to the product-disjoint test set (whose observed
product molecules never appear in training).

Not allowed
Retrieval / look-up of the feedback. Do not rank a record by looking the reaction up in any external
reaction or enzyme database, the chemical literature, a web search / API, or a pretrained model's
memorized reactions. The relevance must be modeled from the given record, not retrieved. (The test
observed products are held out precisely to reward generalization, not recall.)
A purely hand-written rule engine as the whole solution. A fixed table of "EC class X + functional
group Y -> transformation Z" with no learned, calibrated component does not satisfy the intent.
Hand-built rules are welcome as features or as a re-ranking / post-processing layer around a trained
ranker.
Any use of the test feedback or private files. Do not attempt to recover answers.csv, the withheld
label/hard columns, or the true relevance for the test set by any means.
id / metadata hacking. The id is an opaque token; it encodes nothing about the relevance. Do not
attempt to infer relevance from ids, row order, feedback-balance counting, or file statistics.
Degenerate constant scores. Scoring 0.5 (or any constant) on every record scores 0 by construction —
the score rewards ordering observed products above negatives and calibrated relevance, not a lucky
threshold. Hard-coding outputs or otherwise gaming the grader rather than modeling the chemistry is
disallowed.
 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.