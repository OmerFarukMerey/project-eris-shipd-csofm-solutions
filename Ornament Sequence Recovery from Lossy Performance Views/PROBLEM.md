Ornament Sequence Recovery from Lossy Performance Views
Overview
Traditional bowed-string performances contain rapid ornaments, simultaneous notes, resonating strings, and timing that does not follow a perfectly regular pulse. Each row in this challenge represents one real performance excerpt as an ordered sequence of 20–32 latent note events. You must reconstruct the complete event sequence from two complementary symbolic views and a compact acoustic feature array.

The pitch view reveals relative pitch and simultaneous-note multiplicity at some positions. The timing view independently reveals inter-onset-gap and duration bins. Withheld positions are explicit question-mark tokens, and neither view is sufficient alone. The accompanying acoustic row supplies spectral and energy evidence for every ordered event. The intended first-order approach is to align the two visible views, use acoustic cues and neighboring events to recover withheld components, and decode a consistent sequence.

Each target event has the form P±dd_Gd_Dd_Md:

P±dd is relative pitch in semitones, clipped to -24 through +24.
Gd is the inter-onset-gap bin from G0 through G5.
Dd is the duration bin from D0 through D5.
Md is simultaneous-note multiplicity from M1 through M4.
Evaluation Metric
Let the predicted token sequence be \hat y=(\hat y_1,\ldots,\hat y_n) and the target be y=(y_1,\ldots,y_m).
Let d_{\mathrm{lev}}(\hat y,y) be token-level Levenshtein distance. The exact edit term is   E=\max\left(0,1-\frac{d_{\mathrm{lev}}(\hat y,y)}{\max(n,m,1)}\right).  
For valid events a=(p_a,g_a,d_a,m_a) and b=(p_b,g_b,d_b,m_b), pair similarity is   s(a,b)=0.50e^{-|p_a-p_b|/2}+0.20e^{-|g_a-g_b|/1.25}+0.20e^{-|d_a-d_b|/1.25}+0.10\mathbf{1}[m_a=m_b].   Invalid events have pair similarity zero.
Let W be the maximum sum of s over every order-preserving one-to-one matching between predicted and target events. Define P_A=W/n, R_A=W/m, and   A=\frac{2P_AR_A}{P_A+R_A},   with A=0 when either sequence is empty or the denominator is zero.
Form the multisets of adjacent token bigrams. If their multiset overlap count is c, define P_B=c/\max(n-1,1), R_B=c/\max(m-1,1), and   B=\frac{2P_BR_B}{P_B+R_B},   with B=0 when either sequence has no bigram or the denominator is zero.
The final score is   100\times\operatorname{clip}(0.40E+0.40A+0.20B,0,1).  
Minimum 0 means no usable sequence agreement. Maximum 100 means every event and transition is reconstructed exactly.
Measured public-data-only references: sample submission 0.000000; constant sequence 16.116568; mode-filled view fusion 47.295947; capable CPU gradient-boosted sequence decoder with transition decoding 54.994396; perfect 100.000000.
Dataset
train.csv - 1,714 labeled examples.
sample_id - int64 - Content-free submission identifier.
pitch_view - string - Ordered P±dd:Md tokens and P?:M? withheld tokens.
timing_view - string - Ordered Gd:Dd tokens and G?:D? withheld tokens.
capture_profile - string - Balanced nuisance profile A or B; it is not a target.
target_sequence - string - Complete ordered training target.
test.csv - 1,130 query examples with the same query columns and no target column.
train_features.npz - acoustic, float16, shape (1714, 32, 18).
test_features.npz - acoustic, float16, shape (1130, 32, 18).
sample_submission.csv - Correct submission header and all required test IDs.
CSV rows and feature arrays are positionally aligned: the first data row maps to acoustic[0], the second maps to acoustic[1], and so on. The number of populated event rows equals the number of tokens in either symbolic view; remaining rows are zero padding. Acoustic channels 0–11 are rotated chroma-band energies. Channels 12–17 are local pre-onset, center, post-onset, energy-change, variation, and mean-energy observations.

Submission
sample_id - int64 - Must match one test ID exactly.
target_sequence - string - A variable-length, space-separated sequence of valid P±dd_Gd_Dd_Md event tokens.
Submit exactly 1,130 data rows plus the required header. The exact column order is sample_id,target_sequence. Extra or reordered columns, duplicate IDs, missing IDs, unknown IDs, and extra rows are rejected. Row order may differ because scoring aligns by ID. Missing, non-finite, empty, or malformed prediction values are scored as invalid events and never provide an abstention advantage.

Example using real test IDs:


sample_id,target_sequence

1231,P+00_G0_D2_M1 P+02_G3_D1_M1

1510,P-05_G0_D3_M2 P+00_G4_D2_M1

2194,P+00_G0_D1_M1 P+07_G5_D2_M2

What Not to Use
A constant or majority event sequence fails because tune, length, pitch contour, timing, and polyphony vary naturally.
A pitch-view-only method cannot recover timing and duration components hidden independently in the timing view.
A timing-view-only method cannot recover relative pitch or simultaneous-note multiplicity.
Treating positions independently leaves performance on the table because ornament transitions and local timing form ordered motifs.
capture_profile is deliberately balanced and non-informative; using it as a target shortcut does not generalize.
External source matching is not expected and is undermined by tune-disjoint splitting, cropped windows, relative pitch, chroma rotation, feature normalization, and stripped source metadata.
Successful systems should fuse both symbolic views with the acoustic event rows and model dependencies across the output sequence.