Intraoperative ECG Beat-and-Rhythm Transcription
Overview
Each example is a six-second, single-lead ECG recorded during surgery. The task is to

transcribe two complementary parts of the cardiac activity visible in the waveform:

the rhythm family of the window; and
the ordered sequence of heartbeats, including each R-peak position and beat type.
This is not future prediction and it is not classification from hidden metadata. Every

scored event is present in the provided waveform. A successful system must locate the

beats, distinguish their morphology and timing, and use the sequence as context for the

window-level rhythm. The labels were validated by anesthesiologists on real

intraoperative ECG.

The four beat types are:

N: normal beat;
S: supraventricular beat;
V: ventricular beat;
U: unclassifiable beat.
The nine rhythm families are:

sinus_rhythm;
atrial_fibrillation_flutter;
patterned_atrial_ectopy;
patterned_ventricular_ectopy;
sinus_node_dysfunction;
atrioventricular_block;
supraventricular_tachyarrhythmia;
ventricular_tachyarrhythmia;
wandering_multifocal_atrial_rhythm.
The split is disjoint by de-identified patient group. No patient contributes windows to

both train and test. The public waveforms are private crops of longer source windows,

resampled onto 750 points with a mild monotonic local time warp, a smooth amplitude

envelope/filter change, and robust amplitude normalisation. A minority have their

polarity inverted. Beat coordinates are transformed by the exact same time map. These

operations remove source coordinates and simple recording-match shortcuts while

preserving beat order, morphology, and rhythm information.

This dataset is for machine-learning evaluation only. It is not a medical device and

must not be used for patient care.

Evaluation
Scores range from 0 to 1 and higher is better. The final score combines beat detection,

beat typing, rhythm recognition, and performance on the clinically difficult subset:


score = 0.20 * event_micro_F1

      + 0.30 * beat_type_macro_F1

      + 0.30 * rhythm_macro_F1

      + 0.10 * rare_beat_macro_F1

      + 0.10 * rare_rhythm_macro_F1

Beat-event matching
A predicted event matches a gold event only when:

both have the same beat type; and
their sample indices differ by at most 10 samples.
Candidate pairs are considered from smallest timing error to largest. Matching is

one-to-one: one prediction cannot claim two gold beats and one gold beat cannot be

claimed twice.

For any event pool:


event_F1 = 2*TP / (2*TP + FP + FN)

event_micro_F1 pools all four beat types over the complete evaluated test slice.
beat_type_macro_F1 computes event F1 separately for N, S, V, and U, then averages the types present in the ground truth of the evaluated slice. On the full test set all four types are present.
rare_beat_macro_F1 is the unweighted mean of event F1 for V and U. Both are present in the full test set. This prevents accurate normal-beat detection from hiding failure on uncommon ventricular or unclassifiable morphology.
At the nominal 125 Hz grid, the 10-sample event tolerance is 80 ms.

Rhythm matching
rhythm_macro_F1 is the unweighted mean of the ordinary class F1 values for the nine

rhythm families present in the ground truth of the evaluated slice. On the full test set

all nine families are present. Macro averaging prevents the common sinus and atrial

fibrillation families from hiding failures on less frequent rhythms.

rare_rhythm_macro_F1 is the unweighted mean of ordinary class F1 for

atrioventricular_block, supraventricular_tachyarrhythmia,

ventricular_tachyarrhythmia, and wandering_multifocal_atrial_rhythm. All four are

present in the full test set. These classes receive a separate component because they

are clinically important and naturally less frequent, not because test rows are

artificially relabeled.

A perfect transcription scores 1. Predicting no beats receives zero on every beat

component, and unmatched extra events count as false positives.

Dataset
The public package contains 3,221 labeled training windows and 1,107 test windows. Every

signal file is a NumPy .npy array with shape (750,) and dtype float32.

train.csv has one row per labeled window and these columns:

id string): opaque window identifier.
signal string): relative path to the waveform, such as signals/ecg_0123456789abcd.npy.
rhythm_family string): one of the nine rhythm families listed above.
beats string): JSON list of ordered beat events. Each event has the form [sample_index, beat_type], where sample_index is an integer from 0 through 749 and beat_type is N, S, V, or U.
test.csv has the columns:

id string): opaque window identifier.
signal string): relative path to the waveform.
It contains no rhythm or beat labels.

sample_submission.csv is a valid structural example with the exact submission

columns. Its regularly spaced normal beats are illustrative and are not a competitive

baseline.

task_manifest.json documents the signal shape, label vocabularies, submission schema,

split unit, event tolerance, and metric. It contains no per-window hidden data.

signals/ contains one waveform file for every train and test row.

Example labeled row:

undefined

id,signal,rhythm_family,beats

ecg_0123456789abcd,signals/ecg_0123456789abcd.npy,patterned_ventricular_ectopy,"[[71,""N""],[162,""V""],[278,""N""]]"


## Submission

Submit a CSV with exactly these columns in this order:

- `id` `string`): a test id from `test.csv`.
- `rhythm_family` `string`): exactly one allowed rhythm family.
- `beats` `string`): a JSON list of at most 64 ordered events in the form
  `[sample_index, beat_type]`. Sample indices must be integers in `[0, 750)`. Use `[]`
  only when predicting that no valid beat is visible.

Example:

id,rhythm_family,beats

ecg_0123456789abcd,patterned_ventricular_ectopy,"[[71,""N""],[162,""V""],[278,""N""]]"

ecg_fedcba98765432,atrial_fibrillation_flutter,"[[54,""S""],[141,""S""],[229,""S""]]"


The submission must contain every required test id exactly once. Missing ids, extra ids,

duplicate ids, malformed JSON, invalid labels, unordered events, or out-of-range indices

are rejected.

## Appropriate Approaches

- A compact one-dimensional CNN, temporal convolutional network, or transformer can
  share an encoder between an event head and a rhythm head.
- A two-stage system can first detect R-peaks, then classify each beat from local
  morphology and neighbouring RR intervals, followed by a rhythm classifier over the
  resulting sequence.
- Sequence tagging, heatmap regression, or CTC-style decoding are natural ways to avoid
  forcing a fixed number of events.
- Patient-group-aware validation should be approximated by grouping nearby or
  morphology-similar training windows; random window validation can be optimistic.
- Class-balanced sampling or loss weighting is useful because `N` and `S` beats are more
  common than `V` and `U`.

A cheap signal-processing detector establishes a useful floor, but accurate `SVU`

typing and the less frequent rhythm families require learning from the labeled training

waveforms. The score deliberately gives only 20% weight to pooled peak localization;

the remaining components measure typed and rhythm-aware transcription.

## What Must Not Be Used

- Do not use external copies of the source waveform records, annotations, patient ids,
  filenames, or lookup tables to recover hidden labels.
- Do not attempt to align public windows back to an external full-case recording.
- Do not manually inspect or reconstruct private answers from non-public files.
- Train only on the provided public training data.

&nbsp;