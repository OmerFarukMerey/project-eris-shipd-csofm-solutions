Swadesh Phoneme Cipher Decoding
Overview
Somewhere in this dataset is a single 'target' language whose pronunciations have been enciphered. Every distinct sound in that language - every IPA segment - has been consistently replaced by an opaque token under one fixed substitution (the same sound always becomes the same token, and two different sounds never share a token). You are given the meaning of each enciphered word, but not its language's identity and not its true pronunciation.

You are also given complete, true-IPA wordlists for every other language in the database, each labelled with its genetic family and subfamily. The target belongs to the Uralic family. Some of the other languages are its relatives, and because the words in this list are core, slowly-changing basic vocabulary, related languages tend to share cognate forms. The regular sound correspondences between the target and its relatives are your way in: line the enciphered words up against their cognates, work out which token stands for which sound, and decode the whole lexicon.

Concretely: suppose the enciphered word for 'water' is 'x10 x4 x1 x5', and across the support languages the word for 'water' shows up as 'v e s i', 'v e t e', and 'v e z i'. Reading the systematic correspondences across many such words lets you infer x10 -> v, x4 -> e, x1 -> s, x5 -> i, and therefore decode 'x10 x4 x1 x5' as 'v e s i'. The substitution is global, so once you have pinned a token down it decodes that token everywhere - including in words that have no obvious cognate anywhere in the support data.

Evaluation
Each word is scored by the segment-level normalised edit similarity between your decoded pronunciation and the true pronunciation:

similarity = 1 - levenshtein(predicted_segments, true_segments) / max(len(predicted_segments), len(true_segments))
where the two sequences are compared as whitespace-separated IPA segments (so 'v e s i' is four tokens). The final score is the mean of this similarity over every word in the evaluation set and lies in [0, 1]; higher is better. A word for which you submit no prediction, or which you leave enciphered, contributes a similarity near 0. Perfectly recovering the cipher yields a score of 1.

Dataset
Three files are provided.

train.csv
The crib material: true-IPA wordlists for every language in the database except the hidden target. One row per (language, word).

Column	Type	Description
language	string	Language code identifying which language the form belongs to (for example 'fin', 'krl').
family	string	Genetic family of the language - one of roughly 21 families present in the data, such as Uralic, Indo-European, Turkic, Nakh-Daghestanian, Mongolic-Khitan, Tungusic, Dravidian, or Eskimo-Aleut.
subfamily	string	Genetic subfamily or branch - one of roughly 48 values (for example the Uralic subfamily 'Finnic').
concept	string	Concept identifier of the form 'N_gloss', where N is an integer index and gloss is a short English label (for example '1_eye', '2_ear', '3_nose'). The same concept ids appear in test.csv, which is what lets you align words across languages by meaning.
ipa	string	The word's pronunciation, written as IPA segments separated by single spaces (for example 's i l m æ'). Individual segment tokens may be more than one character (for example 'aː').
test.csv
The enciphered target lexicon you must decode. One row per target word.

Column	Type	Description
id	string	Unique row identifier (for example 't00042'). Use it to match your predictions to rows.
concept	string	Concept identifier in the same 'N_gloss' form as in train.csv (for example '1_eye'). This is the meaning of the enciphered word and your key for aligning it to the support languages.
cipher	string	The enciphered pronunciation, written as opaque tokens separated by single spaces (for example 'x0 x5 x0 x2'). Each token is the letter 'x' followed by an integer, and each token maps to exactly one true IPA segment under a single fixed substitution that is shared by every row in the file.
sample_submission.csv
An example submission in the exact format required for grading. It has the same 'id' column as test.csv and a placeholder 'ipa' column; replace the placeholder values with your decoded pronunciations.

The 'concept' ids are consistent between test.csv and train.csv; that shared vocabulary list is what makes cross-lingual alignment possible. A single concept may appear in more than one row (genuine synonyms); each row is scored independently. Some target sounds are rare, or occur only in words with no cognate in any relative, and will be harder to resolve than others.

Submission
Submit a single CSV file with exactly two columns, in this order:

Column	Type	Description
id	string	The row identifier, copied verbatim from the 'id' column of test.csv (for example 't00042').
ipa	string	Your decoded pronunciation for that row: IPA segments separated by single spaces, in the same format as the 'ipa' column of train.csv (for example 'v e s i').
The first line of the file is the header 'id,ipa'. Each following line is one data row, one per row in test.csv, with each 'id' used exactly once. There are no blank lines anywhere in the file: the header line is immediately followed by the first data row, and each data row is immediately followed by the next. Any 'id' from test.csv that is missing from your submission is scored as incorrect.

A correctly formatted submission file for three test rows looks exactly like this:

id,ipa  
t00042,s i l m  
t00043,n i n a  
t00044,v e s i  

Method Requirements
This is an unsupervised decipherment problem: there are no target-language labels to train on. You are expected to recover the phoneme substitution from the data itself by modelling the systematic sound correspondences between the enciphered target and its relatives - for example through statistical alignment of cognates, iterative or EM-style refinement of a token-to-segment mapping, optimisation under the global one-to-one constraint, phonetic-feature or pretrained phoneme representations, or a small sequence model trained from scratch. Because the substitution is a single consistent bijection over the target's sound inventory, strong solutions exploit that global consistency rather than decoding each word in isolation. Solutions must be derived solely from the provided files; approaches that look the target up in outside resources, reconstruct it from a language model's memory, or hard-code the answers are not permitted (see What Not To Use).