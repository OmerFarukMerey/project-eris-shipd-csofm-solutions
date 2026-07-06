# DNA Barcode Family and Artifact Detection

## The problem

`solution.py` reads `dataset/public/train.csv` and `test.csv`, trains a model from scratch on the raw
nucleotide barcode text, and writes predictions for the test set to `working/submission.csv`
(columns: `id, label`), where `label` is one of `F01`–`F20`, `NOVEL`, or `ARTIFACT`.

## The solution

The classifier combines two complementary signal sources computed directly from the raw sequence text.

### 1. Engineered reading-frame / codon features (the ARTIFACT signal)

These barcodes are insect COI mitochondrial sequences, so a genuine barcode should translate cleanly under
the invertebrate mitochondrial genetic code (stop codons = TAA/TAG only, not TGA as in the standard code).
For each sequence all 6 reading frames are scanned (3 forward strand + 3 reverse complement, since
orientation isn't normalized in this data) and stop codons counted in each. A genuine barcode has one clean
frame (near-zero stop codons); a pseudogene-like ARTIFACT usually does not, even in its best frame, because
frameshifts/degradation disrupt the true reading frame. This single idea is validated empirically on the
training data: the minimum stop-codon count alone separates ARTIFACT from everything else with **AUC ≈ 0.77**.

The winning frame is also used to canonicalize each sequence (reorient to the correct strand, trim to a
codon boundary) — fed to the neural network below — and to compute a small set of derived features: codon
usage (64-dim), GC content overall and by codon position, dinucleotide frequencies (16-dim), and a second,
independent scan under the standard genetic code as a hedge. All engineered features are whole-sequence
summary statistics (101 dimensions total).

This same feature set carries essentially no signal for distinguishing NOVEL from the 20 known families
(confirmed empirically, AUC ≈ 0.5) — NOVEL sequences are genuine, intact barcodes from an unlisted family, so
they pass the reading-frame test the same as any known family. NOVEL has thousands of labeled training
examples, so it's treated as an ordinary class for the network to learn, not an open-set/anomaly-detection
problem.

### 2. A 1D residual CNN over the sequence (the family / NOVEL signal)

The canonicalized nucleotide sequence (one-hot A/C/G/T, fixed length with padding/masking) is passed through
a small residual 1D CNN (stem + 4 residual blocks, kernel sizes that are multiples of codon length 3) with
masked attention pooling, producing a fixed-size embedding. This branch learns the family-specific sequence
patterns.

The CNN embedding and the 101-dim engineered feature vector are concatenated (late fusion) and passed
through a small MLP head to produce the final 22-way softmax.

## Training

Stratified 3-fold cross-validation, AdamW optimizer, a class-weighted (inverse-sqrt-frequency) and
label-smoothed (0.08) cross-entropy loss to handle the ~36× class imbalance and the modest label noise
mentioned in the problem statement, early stopping on validation macro-F1, and a wall-clock training budget
guard. Predictions across the 3 fold models are averaged (probability ensembling) for the final test
predictions.

A class-balanced batch-sampling scheme (oversampling the rarest classes) was tried and rejected: it
consistently reduced overall macro-F1 by hurting the majority classes (especially ARTIFACT) more than it
helped the smallest ones. Natural sampling with per-sample loss weighting performed best in direct comparison
and is what's used.

## Fallback

If `torch` is unavailable, or the neural path fails for any reason, the script automatically falls back to
a scikit-learn `HistGradientBoosting` classifier trained on the 101-dim engineered feature table alone (same
fold splits and class weighting), so a valid submission is always produced.

## Results (out-of-fold, on the training data)

Overall OOF macro-F1: **~0.745** — ARTIFACT F1: ~0.91, NOVEL F1: ~0.95. Most families score F1 in the
0.73–0.97 range. The two smallest classes (F16: 162 rows, F20: 123 rows, out of ~27.6K total) score near zero
— a genuine data limitation (too few examples to learn a reliable decision boundary) rather than a fixable
modeling gap; attempts to rebalance training toward them cost more overall accuracy than they gained.

## How to run

Requires Python with `numpy`, `pandas`, `scikit-learn` (`torch` optional but strongly recommended for the
primary CNN path):

```bash
cd "DNA Barcode Family and Artifact Detection"
python3 solution.py
```

Runtime is roughly 25–30 minutes on a machine with GPU/MPS acceleration for the primary path (dominated by
the 3-fold CNN training); the sklearn fallback path is much faster but only used if `torch` is unavailable
or the CNN path errors out.
