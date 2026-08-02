Catalan Administrative Discourse Operator Reconstruction
Overview
Administrative documents depend on short connective phrases to express addition, contrast, consequence, sequencing, explanation, and related discourse transitions. When those phrases are missing, restoring them requires understanding both sides of each boundary and the larger sequence.

This sequence-to-sequence challenge is configured for one NVIDIA GPU with 10 GB of VRAM. Every example contains six ordered sentence boundaries taken from one real public-administration document. The opening discourse operator at each boundary has been removed. All remaining words have been replaced with stable, collision-prone codes, so the task cannot be solved by external text lookup.

Predict the six removed operators in order. Targets are opaque tokens from O00 through O25; the mapping from original phrases is intentionally not public. Repeated operator tokens are allowed.

Every row also supplies a finite-field constraint. For that row, each operator token deterministically generates an invertible 2-by-2 matrix over the field modulo 17. Multiplying the six matrices in predicted order should reproduce the supplied boundary_matrix. Matrix multiplication is order-sensitive, and many different paths share the same boundary, so the constraint assists structured decoding without replacing contextual learning.

Task
For every row in test.csv, predict an operator_path containing exactly six operator tokens.

context_sequence contains six gap sections:

G1 through G6 identify the ordered gap positions.

LEFT introduces the coded words before a gap.

MASK marks the removed discourse operator.

RIGHT introduces the coded words after the removed operator.

NEXT_GAP separates consecutive gap sections.

Codes matching w[0-9a-f]{4} are stable collision-prone lexical buckets.

The same word normally receives the same code across rows, but unrelated words may collide. Structural markers, IDs, row order, seeds by themselves, and filenames do not encode target labels.

Strong solutions should use a compact neural sequence encoder, learn contextual patterns for each gap, model dependencies across the six outputs, and use the matrix boundary during constrained decoding. Models, batches, and cached tensors must fit within 10 GB of GPU memory.

Finite-Field Constraint
The complete public construction is stored in operator_algebra.json.

For an operator token and the row’s constraint_seed:

Starting with counter 0, calculate SHA-256 of the ASCII string FFDOR1|constraint_seed|operator_token|counter.

Reduce the first four digest bytes modulo 17 and interpret them as [a,b,c,d], representing [[a,b],[c,d]].

If the determinant is zero modulo 17 or the matrix is the identity, increment the counter and repeat.

Start from the identity matrix and right-multiply the six accepted matrices in submitted order, reducing every entry modulo 17.

The hidden path’s product is the public boundary_matrix. Because the construction is instance-conditioned and highly many-to-one, boundary lookup across rows does not reveal the answer.

A context-free meet-in-the-middle search can nevertheless enumerate all 26-cubed three-token prefixes and suffixes and always find at least one six-token path whose product equals the supplied matrix. The matrix-agreement component, worth 15% of the total score, can therefore be earned in full without learning discourse context. This is an intentional decoding aid, not evidence that the recovered path is the labelled path: many paths satisfy the same matrix, and the position, edge, edit, and exact-path components comprising the other 85% distinguish them using the hidden contextual target.

Dataset
train.csv contains 2,108 labelled rows.

test.csv contains 702 unlabelled rows.

sample_submission.csv is a valid training-only scaffold. It uses the per-position majority token from train.csv, never test labels.

operator_algebra.json defines the allowed tokens, modulus, matrix layout, matrix-generation procedure, and ordered-composition rule.

train.csv columns
sample_id

Type: string.

Format: FFDOR_TR_ followed by six decimal digits.

Meaning: synthetic row identifier assigned after train shuffling; it has no predictive meaning.

context_sequence

Type: string.

Format: six space-separated masked-gap sections using the markers and word-code grammar described above.

Meaning: anonymized left and right context for six ordered missing discourse operators.

constraint_seed

Type: string.

Format: exactly 16 lowercase hexadecimal characters.

Meaning: row-specific input to the public operator-matrix construction.

boundary_matrix

Type: string containing a JSON array.

Format: [a,b,c,d], with four integers from 0 through 16.

Meaning: ordered finite-field matrix product of the hidden six-token path.

operator_path

Type: string.

Format: exactly six space-separated tokens from O00 through O25.

Meaning: training-only target sequence in gap order.

test.csv columns
sample_id

Type: string.

Format: FFDOR_TE_ followed by six decimal digits.

Meaning: synthetic row identifier assigned after an independent test shuffle; it has no predictive meaning.

context_sequence

Type: string.

Format: the same six-section coded sequence used in train.csv.

Meaning: anonymized input contexts requiring operator reconstruction.

constraint_seed

Type: string.

Format: exactly 16 lowercase hexadecimal characters.

Meaning: row-specific matrix-generation input.

boundary_matrix

Type: string containing a JSON array.

Format: [a,b,c,d], with four integers from 0 through 16.

Meaning: target path’s ordered matrix product modulo 17.

sample_submission.csv columns
sample_id

Type: string.

Format: one exact identifier from test.csv.

Meaning: test row being predicted.

operator_path

Type: string.

Format: exactly six space-separated allowed operator tokens.

Meaning: predicted discourse-operator sequence.

operator_algebra.json fields
name — string; algebraic constraint name.

modulus — integer; prime modulus 17.

path_length — integer; required output length, 6.

identity_matrix — array of four integers; flattened identity matrix.

operator_tokens — array of strings; allowed output vocabulary.

matrix_layout — string; flattened matrix interpretation.

generation — string; deterministic instance-conditioned matrix procedure.

composition — string; ordered right-multiplication procedure.

The train/test split is performed by whole source-document families before independent shuffling. Documents sharing an exact participant-visible gap or six-gap window are grouped together. Preparation requires exactly 2,108 training rows and 702 test rows, exactly six visible gap sections per row, and zero train/test overlap for both complete contexts and individual masked-gap sections. It also verifies that all 26 operators occur in both partitions, with at least 30 training occurrences and 5 test occurrences per operator.

Evaluation
Each structurally valid submission receives the mean row score:


score = 0.45 * A_position

      + 0.20 * F_edge

      + 0.15 * S_edit

      + 0.15 * A_matrix

      + 0.05 * I_exact

A_position is the fraction of the six positions whose predicted token equals the hidden token.

F_edge is multiset F1 over the five adjacent directed token pairs. For a path p1 p2 p3 p4 p5 p6, its edge multiset is (p1,p2), (p2,p3), (p3,p4), (p4,p5), (p5,p6). Direction is significant, so (O01,O02) differs from (O02,O01). Absolute edge position is not included: the same directed pair may match at a different edge position. Repeated pairs retain multiplicity, and overlap uses the minimum predicted and hidden count for each directed pair.

S_edit is 1 - token_Levenshtein_distance / 6.

A_matrix is the fraction of the four flattened product-matrix cells that equal the hidden product.

I_exact is 1 when the complete six-token path is exactly correct and 0 otherwise.

A perfect submission scores 1.0. Higher is better.

An individual row receives zero if its operator_path is missing, contains anything other than O00 through O25, or does not contain exactly six space-separated tokens. Duplicate tokens are valid and are not an error.

The entire submission is rejected when:

the columns are not exactly sample_id,operator_path in that order;

an ID is empty or duplicated; or

the submitted ID set differs from the test.csv ID set.

Submission
Submit a UTF-8 CSV file with exactly these columns, in this order:


sample_id,operator_path

FFDOR_TE_000001,O08 O08 O08 O08 O08 O08

FFDOR_TE_000002,O08 O08 O08 O08 O08 O08

Each test ID must appear exactly once. Spaces separate tokens inside operator_path; do not use commas or JSON syntax inside that field. sample_submission.csv is the exact file-format template.

Rules
Use only the supplied challenge files and preinstalled runtime, including the preinstalled GPU libraries.

Solutions must run on one NVIDIA GPU with at most 10 GB of VRAM and finish within 1.5 hours.

GPU memory use must remain within the platform limit; compact encoders, mixed precision when supported, and bounded batches are recommended.

Do not access the internet or install packages during execution.

Do not use external corpora, source-text copies, reverse lookup, source-specific dictionaries, hidden files, IDs, or row order to recover targets.

 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.