Overview
This is a sequence-to-sequence spectral decoding task. Each input is a continuum-normalized

near-infrared flux sequence with 7,514 ordered wavelength positions. The sequence contains the

blended light of two unresolved stars whose absorption patterns overlap and are shifted relative

to one another.

For every input sequence, produce an ordered output sequence of exactly eight tokens encoding four

conceptual quantities: the brighter component's three-token atmospheric state, the fainter

component's three-token atmospheric state, the fainter component's one-token light contribution,

and the one-token signed velocity separation.

The output order is fixed:

primary effective-temperature token;

primary surface-gravity token;

primary metallicity token;

secondary effective-temperature token;

secondary surface-gravity token;

secondary metallicity token;

secondary light-fraction token;

signed relative-velocity token.

The primary is always the component contributing more light to the released composite. This

removes the usual component-swap ambiguity.

This is not single-star parameter fitting. A successful sequence model must identify two

overlapping line systems, preserve wavelength order, infer which features move together, and emit

the eight output symbols in the correct semantic order.

What The Task Requires
Solving an item well exercises four related capabilities:

Long-sequence representation: capture temperature-, gravity-, and metallicity-sensitive

structure across thousands of ordered flux values.

Component separation: distinguish the brighter line system from the weaker secondary pattern.

Sequence alignment: infer the signed displacement between the two systems while respecting

detector gaps and missing intervals.

Ordered decoding: emit eight valid vocabulary tokens in exactly the documented order.

Data Provenance
The challenge examples are derived composite observations prepared from a licensed astronomical

source collection. Full attribution and license information are maintained on the associated

source-dataset record.

Source identifiers, sky coordinates, timestamps, field names, telescope metadata, and original

component spectra are absent from participant files. Parent observations are assigned to only one

side of the split before composites are constructed.

File Structure
train.csv - training identifiers, spectrum indices, and target token sequences.

test.csv - test identifiers and spectrum indices, without target sequences.

train_spectra.npy - the training input-sequence matrix.

test_spectra.npy - the test input-sequence matrix.

wavelength.npy - the shared ordered wavelength positions in Angstrom.

sample_submission.csv - a correctly formatted token-sequence submission.

The spectrum matrices have shape (number_of_examples, 7514) and use float16 storage. Convert

them to float32 during model computation when appropriate.

The row selected by spectrum_index is the corresponding input sequence. For example, a training

row whose spectrum_index is 17 uses train_spectra.npy[17].

Input Sequence
Each flux row is ordered by wavelength.npy. The sequence spans three detector segments. Gaps

between segments remain as jumps in the wavelength values rather than artificial samples.

Composite sequences contain continuum variation, noise, broadened features, and neutral-valued

missing intervals. Wavelength position is meaningful; randomly permuting the input destroys the

line geometry needed for decoding.

Output Vocabulary
target_sequence contains exactly eight whitespace-separated tokens. Each position has its own

prefix and an ordinal, zero-padded bin index.

PTE000 to PTE127 - primary effective temperature, 128 uniform bins from 2,500 to 9,000 K.

PLG000 to PLG127 - primary log surface gravity, 128 uniform bins from -1.5 to 6.0 dex.

PMH000 to PMH127 - primary metallicity, 128 uniform bins from -3.0 to 1.0 dex.

STE000 to STE127 - secondary effective temperature, using the same 128-bin scale.

SLG000 to SLG127 - secondary log surface gravity, using the same 128-bin scale.

SMH000 to SMH127 - secondary metallicity, using the same 128-bin scale.

SFR000 to SFR063 - secondary light fraction, 64 uniform bins from 0.08 to 0.34.

DRV000 to DRV127 - secondary-minus-primary velocity, 128 uniform bins from -260 to 260 km/s.

Bin endpoints are inclusive. For a vocabulary with B bins spanning lower value L to upper

value U, token index k represents L + k * (U - L) / (B - 1).

Example target sequence:

PTE043 PLG092 PMH074 STE061 SLG105 SMH069 SFR031 DRV088
Token prefixes, positions, and capitalization are mandatory.

Evaluation
The grader parses each output sequence into eight ordinal token indices. For each position, the

corpus-level squared token-index error is divided by the error of the optimal constant token for

that position. The normalized position errors receive these weights:

primary atmosphere positions: 2/30 each;

secondary atmosphere positions: 5/30 each;

secondary light fraction: 3/30;

signed relative velocity: 6/30.

For each sequence position j, its ratio is computed separately across all test rows:

position_ratio_j = sum_rows((predicted_index_row_j - true_index_row_j)^2) / sum_rows((true_index_row_j - mean_true_index_j)^2)
The final sequence skill is the weighted sum over all eight positions:

score = 1 - sum_j(position_weight_j * position_ratio_j), for j = 1,...,8
Higher is better. The score is bounded to [0.001, 1.0]. An exact token sequence scores 1.0,

while constant sequences sit at the floor. Giving more metric weight to the secondary atmosphere

ensures that decoding only the brighter line system is insufficient.

Malformed sequences, missing tokens, wrong prefixes, out-of-range indices, duplicate ids, or an

incorrect id set receive the minimum score.

Submission
Submit a CSV with exactly two columns:

id, target_sequence

Provide exactly one row for every id in test.csv. The target_sequence value must contain all

eight tokens in the documented order.

Example rows:

id,target_sequence TE0123ABCDEF,"PTE043 PLG092 PMH074 STE061 SLG105 SMH069 SFR031 DRV088" TE4567ABCDEF,"PTE071 PLG110 PMH096 STE052 SLG084 SMH058 SFR019 DRV034"
What Not To Use
Do not treat spectrum_index as a feature; it is only an array lookup key.

Do not use the opaque id or row order to predict tokens.

Do not emit physical decimal values directly. The required output is the fixed eight-token

sequence.

Do not collapse the input to a single average star. The fainter component's three atmosphere

tokens carry 15/30 of the total metric weight (and 15/21 of the atmosphere-only weight).

Do not copy the sample sequence as a prediction strategy; it is only a formatting example.

Expected Output
For every test flux sequence, return one valid eight-token output sequence. Strong solutions should

preserve wavelength order, separate the two shifted line systems, and decode all eight positions

rather than only the brighter component.

 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.