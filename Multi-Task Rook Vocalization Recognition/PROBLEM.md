A captive colony of 15 rooks was recorded over many sessions, and every vocalisation was annotated with which individual produced it and which call type it was. This challenge asks you to do three different things at once on each one-second window of audio: decide whether a possibly faint rook call is present, recognise which of the 15 known rooks produced it, and recognise which of the 8 common call types it is.

These are three genuinely different skills. Detecting that a call is present tells you nothing about who made it or what type it is, and the individual and the call type are themselves only weakly related, because each rook uses many call types and the same types recur across individuals. The final score is the average of the three, so being excellent at one skill alone leaves most of the score unclaimed, and you have to detect and identify and categorise.

The test windows come from recording sessions held out of training, so a model cannot lean on a session-specific background or one stereotyped call. It must learn a genuine voice signature and call-type representation that survive a change of session, and it must find faint calls without raising false alarms. This is deliberately hard.

Task
For each test window you output, in one row:

det_score — how likely the window contains a rook call, higher meaning more likely a call.
ind_0 through ind_14 — a score for each of the 15 rooks, higher meaning more likely that individual made the call.
ct_0 through ct_7 — a score for each of the 8 call types, higher meaning more likely that call type.
You are given training windows, each marked as a call or background, and for calls labelled with the individual and, where applicable, the call type, together with the recording session so you can hold out whole sessions when validating. The individual and call-type outputs are only scored on the call windows, but you output all of them for every window.

Required approach. Train a model on the provided windows, for example a single multi-task network with a shared spectrogram body and three heads for detection, individual and call type, or separate specialised models. A learned representation is what carries all three skills, and a solution that only detects, or only identifies, leaves most of the score unclaimed.

Evaluation
Submissions are scored with RookScore, higher is better, in the range 0 to 1. It is the average of three sub-scores, each in 0 to 1:

Detection efficiency. For a fixed false-alarm rate a, set the threshold at the det_score above which a fraction a of the background windows fall. The detection efficiency at that false-alarm rate is the fraction of weak calls, the faint near-threshold ones, scored above the threshold. This sub-score averages the efficiency at a = 0.05 and a = 0.10.
Individual agreement. For each call window, rank the 15 rooks by your ind_ scores. The reciprocal rank of the true individual is 1 divided by its rank, and tied scores share their average rank. This sub-score is the mean reciprocal rank over the call windows.
Call-type agreement. The same mean reciprocal rank for your ct_ scores against the true call type, over the call windows that carry one of the eight scored call types.
RookScore = the mean of detection efficiency, individual agreement and call-type agreement.

Only the ordering of your scores within each group matters, so any monotonic scale is fine.

Dataset
The prepared public dataset:

train_X.npy — training windows, a float16 array of shape Ntrain by 16000: one-second windows at 16 kHz, a mix of calls and background.
train_iscall.npy — an int8 array of length Ntrain: 1 if the window is a call, 0 if background.
train_ind.npy — an int8 array of length Ntrain: the individual id 0 to 14 for calls, and -1 for background.
train_ct.npy — an int8 array of length Ntrain: the call-type id 0 to 7 for calls that carry one of the eight scored types, and -1 otherwise.
train_rec.npy — an int16 array of length Ntrain: a recording-session id. Calls that share a value came from the same session, so use it to hold out whole sessions when validating.
test_X.npy — test windows, a float16 array of shape Ntest by 16000. The id of the window in row i, counting rows from 0, is the letter t followed by i written as a zero-padded 5-digit integer, so row 0 is t00000, row 1 is t00001, and so on.
individuals.txt, call_types.txt — the 15 rook names and the 8 call-type codes, one per line, where the line number is the id.
sample_submission.csv — a valid submission in the required format.
The same 15 individuals appear in training and test, but training and test windows always come from different recording sessions.

Because the test windows are whole recording sessions held out of training, hold out whole sessions when you validate: group by train_rec.npy so no session appears on both sides of your split. A validation that does not hold out whole sessions will overestimate the held-out test score by roughly a tenth, because windows from the same session share a background and a few stereotyped calls and are much easier to match than windows from an unseen session.

Submission
Submit a CSV with exactly these 25 columns: id, det_score, ind_0 through ind_14, ct_0 through ct_7.

Every test id must appear exactly once, and every value must be finite. Output all scores for every window, call or background. Example, abbreviated:

id,det_score,ind_0,...,ind_14,ct_0,...,ct_7  
t00000,0.98,0.03,...,0.01,0.10,...,0.05  
t00001,0.02,0.07,...,0.06,0.14,...,0.09  

Write the final submission to ./working/submission.csv, UTF-8.

Allowed And Prohibited
Allowed:

Train any model on the provided windows: 2-D CNNs on spectrograms, 1-D CNNs on the waveform, temporal-convolutional, recurrent or transformer networks, multi-task shared bodies or separate specialised models, embedding or metric-learning heads.
Any preprocessing of the audio such as spectrograms, band-pass filtering, whitening, normalisation or resampling, plus data augmentation and ensembling.
Standard open-source deep-learning libraries such as PyTorch.
Prohibited:

Do not use external datasets or any information beyond the provided files.
Do not train on, adapt to, or fit statistics from the test windows, whose labels are withheld.
Do not hardcode outputs or use per-id answer tables.
Do not use external LLM APIs or any model-generated labels in your submission.
 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.