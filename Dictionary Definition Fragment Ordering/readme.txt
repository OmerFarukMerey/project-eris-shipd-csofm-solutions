Dictionary Definition Fragment Ordering
=======================================

Entry point
-----------

The submitted program is solution.py.

Platform invocation:

    python3 solution.py <public_dir> <submission_out>

For the repository's local convention, this also works:

    python3 solution.py

The no-argument form reads dataset/public/{train,test}.csv relative to the script and writes working/submission.csv. The implementation is deterministic, CPU-only, performs no network access or downloads, and uses only the supplied CSV content plus NumPy and scikit-learn already available in the runtime.

The effective execution path first runs one complete held-out ensemble validation fold with lexical-family proxy groups kept disjoint, selects latent-role weights using only that fold, releases that model, and only then refits on every labeled row for submission. The observed full local execution time for grouped selection, refit, prediction, and output was 208 seconds.

Approach
--------

The cards are contiguous and internally correct, so only joins, endpoints, and global order need to be scored. The solution reconstructs every training definition from answer_json, trains several complementary sparse sequence models, scores all legal card transitions, and finds the globally best permutation with exact subset dynamic programming.

Opaque aliases are treated as words. Their stability across rows is useful: frequent aliases acquire grammatical behavior even though the source words are never recovered. The model never attempts dictionary lookup, source-record matching, row-ID decoding, or manual test annotation.

Authentic parent definition_tokens are already supplied features and have known token order. The solution uses parent definitions from both train and test as extra unlabeled language sequences. This is transductive feature extraction from allowed inputs, not answer access. Parent definitions increase the amount of observed syntax by roughly one third and are especially valuable for sparse boundary n-grams.

Model architecture / algorithm
------------------------------

1. Interpolated trigram boundary language model

   Reconstructed training definitions and all supplied parent definitions are padded with begin/end markers. Unigram, bigram, and trigram counts are collected. A trigram probability backs off smoothly through a bigram and then a smoothed unigram probability. Only terms affected by a permutation are scored:

   - the first two tokens of the first card,
   - the first two tokens after every card join,
   - the end marker after the final card.

   Since every card has at least two tokens, these trigram boundary terms depend on at most the two cards at a join.

2. Multiclass absolute-position models

   A separate sparse logistic regression is trained for each possible card count (4, 5, 6, or 7). For each card it returns log probabilities for every exact output position. This learns strong endpoint and broad syntactic-role signals that a local language model alone cannot enforce.

3. Fragment-adjacency classifier

   A binary sparse logistic regression is trained on every directed pair of cards in a training row. The positive class means that the right card immediately follows the left card in the authentic order. This discriminatively learns which exact suffix/prefix patterns form a join, including fragment-level length and part-of-speech effects.

4. Token-adjacency classifier

   Every token split with two tokens on each side in every reconstructed definition and supplied parent definition becomes a positive local boundary example. Deterministically mismatched right contexts from the same part of speech provide two negative examples per positive. A high-capacity sparse logistic regression learns valid cross-boundary bigrams, trigrams, and four-grams. This exposes all known internal token adjacencies to the classifier, not only the random boundaries that happened to become training cards.

5. Pairwise precedence classifier

   Another binary logistic regression predicts whether one card occurs anywhere before another. Unlike adjacency models, it can preserve global ordering when a particular local join is unseen. It uses directional token bags, card endpoints, lengths, part of speech, and lexical-context overlaps.

6. Transductive latent-role models

   Exact alias n-grams are weak for lexical-family-held-out rows, especially when a content alias never occurs in labeled training definitions. Every evaluation alias still occurs inside one or more internally ordered cards. The solution collects its known left/right neighbors, definition position bins where authentic positions are available, and part-of-speech usage. A deterministic signed BLAKE2 feature projection produces a 64-dimensional context signature; a version-independent Lloyd implementation clusters these signatures into 192 latent grammatical roles.

   Three class-level models use those roles: a smoothed class trigram model, a token-boundary classifier trained from all authentic internal joins, and a fragment-adjacency classifier trained on labeled cards. Evaluation-card internals participate only as unlabeled positive internal contexts; no proposed cross-card join or answer is used. This particularly improves rows with low exact-alias coverage.

7. Exact constrained decoding

   The final score is the sum of:

   - exact-alias trigram start, join, and end scores,
   - 9.0 times exact-position log probability,
   - 2.0 times exact fragment-adjacency logit,
   - 3.0 times exact token-adjacency logit,
   - 0.25 times every earlier/later pairwise precedence logit,
   - validation-selected latent-role language, token-join, and fragment-join scores.

   The effective grouped fold compares seven conservative latent-role weight triples, including a zero-weight sparse reference. The verified run selected (1.0, 4.0, 16.0) for class language, class token adjacency, and class fragment adjacency respectively.

   A Held-Karp-style subset DP keeps the best partial path for each (used-card subset, final card). Appending a card fixes its exact position, immediate transition, and precedence relative to every card already in the subset. Complexity is O(2^m * m^2), with m at most 7, so decoding is exact rather than greedy or beam-pruned. Score ties are broken by displayed card index for reproducibility.

Feature engineering
-------------------

Absolute-position card features:

- token alias counts inside the card;
- aliases at the first three and last three offsets;
- first and last internal bigrams;
- raw and log token length;
- source part of speech;
- overlap count with the row's lemma aliases;
- whether parent context exists;
- overlap counts with parent lemma aliases and parent definition aliases.

Fragment adjacency features:

- last two aliases of the left card;
- first two aliases of the right card;
- exact cross-boundary bigram;
- both cross-boundary trigrams;
- two skip-style suffix/prefix conjunctions;
- card lengths, card count, and part of speech.

Token adjacency features add the exact four-token window spanning a proposed join. Training from all authentic sequence splits gives this model much denser positive coverage than card boundaries alone.

Pairwise precedence features:

- directional alias bags for both cards;
- first/last aliases at three offsets on both sides;
- proposed edge alias conjunction;
- both card lengths and card count;
- part of speech;
- directional overlap with target lemmas, parent lemmas, and parent-definition tokens.

Validation strategy
-------------------

The hidden lexical-family identifier is not present in the CSV, so ordinary random splitting can accidentally be optimistic. Development validation therefore built conservative connected groups from rows sharing rare target or parent lemma aliases. Aliases occurring in at most two rows were unioned into a component, and StratifiedGroupKFold kept every such component wholly in one fold while approximately balancing part of speech and card count. This is stricter than row-random validation for every family relationship observable from the supplied columns; relationships with no shared supplied alias cannot be reconstructed.

The local metric exactly matched the challenge metric:

    0.85 * mean exact-position accuracy
    + 0.15 * complete-order exact rate

Earlier development folds showed that exact sparse features degrade sharply as labeled-token coverage falls: below 80% coverage, the sparse ensemble scored roughly 0.47 on one held-out slice, while the first latent-role ensemble reached roughly 0.51. This motivated replacing additional exact n-grams with transductive grammatical-role features rather than increasing sparse model capacity.

The same family-safe procedure is part of solution.py's effective path rather than existing only in exploratory code. It constructs observable lexical components deterministically, uses StratifiedGroupKFold, asserts that fit and holdout group sets are disjoint, trains every ensemble component only on fit rows, predicts held-out rows, and computes the official metric for each allowed latent-weight triple. The verified run's sparse reference scored 0.542398; selected latent weights raised the same held-out fold to 0.553448. Exact-position accuracy was 0.579104 and complete-order exact was 0.408065. The validation model is then released and the final ensemble is newly fit on all labeled rows.

All training, transductive role induction, and validation operations use only supplied data. Supplied parent definitions and internally ordered evaluation-card spans are unlabeled input context. Holdout fragment orders and all cross-card evaluation joins remain inaccessible during fitting.

What worked
-----------

- A smoothed trigram boundary model was a strong first baseline: about 0.431 on the development row split.
- Combining local boundary coherence with an exact-position assignment model was substantially better than either model alone.
- Parent definition sequences materially improved boundary modeling. Adding them to the language corpus raised the development ensemble by several points; including the test rows' supplied parent definitions was also consistently useful.
- Training token adjacency from every known authentic split, rather than only observed card joins, gave another clear gain.
- The global precedence model improved difficult six- and seven-card rows where a chain of locally plausible joins was not globally coherent.
- Latent grammatical roles reduced dependence on exact aliases. Their stable hashed projection and custom clustering avoid numerical and version drift across CPU runtimes.
- Selecting only three latent score weights on the grouped fold improved the verified hard fold from 0.542398 to 0.553448 without test labels.
- Exact dynamic programming is both faster and safer than enumerating up to 7! permutations, while optimizing the complete ensemble objective without approximation.

What did not work
-----------------

- The displayed card order contains no useful authentic-position prior; treating it as an ordering baseline is close to random.
- A pure exact-position classifier scored only about 0.357. It finds broad grammatical regions but cannot reliably connect neighboring fragments.
- The trigram model alone frequently chose a fluent local join but misplaced a whole clause.
- A ridge regression on normalized fragment position duplicated weaker parts of the multiclass position and precedence models and did not improve the ensemble.
- Extra standalone start/end classifiers produced only a very small, unstable validation change, so they were excluded.
- Features based on the normalized location of aliases inside the parent definition were noisy because common grammatical aliases match at many positions.
- Card-count-specific score weights overfit one split and lost score on grouped validation. The final solution uses conservative shared weights.
- More aggressive high-order n-grams were too sparse for the available corpus; smoothed trigrams plus a discriminative four-token adjacency classifier were more robust.
- A dense SVD/K-means role pipeline looked stronger in one development environment but was numerically unstable across installed BLAS/scikit-learn versions. It was replaced with deterministic feature hashing and elementwise Lloyd clustering.

Output safeguards
-----------------

Before writing each row, solution.py verifies that the predicted order has exactly the displayed number of fragment IDs and that its ID set matches the cards exactly. It writes exactly the columns id and answer_json. Each JSON value contains exactly one fragment_order field, and output row order follows test.csv.
