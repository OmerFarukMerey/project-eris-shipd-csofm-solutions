An airborne laser scanner sweeps a forest and fires millions of short pulses at the canopy. Most pulses do not stop at the first thing they touch. A pulse that clips the edge of a leaf keeps going, strikes a branch, then understorey, then perhaps the ground, and the instrument records a separate echo for each surface. The sequence of echoes from a single pulse is a vertical transect through the canopy — a direct measurement of what is underneath the outer surface.

You are given a point cloud from which every echo after the first has been deleted. What remains is exactly one point per emitted pulse: the outer skin of the forest, and nothing below it.

Your task is to recover what the deleted echoes would have said.

This is not an obfuscation. The later echoes are not present in your input in any encoded form — they were physically removed. You are inferring the interior of a canopy from its surface.

The task
For each designated query pulse, predict two things.

1. echo_class — how many further surfaces that pulse struck.

This is a class index, not a count. The mapping is:

echo_class	further echoes after the first	total echoes from that pulse
0	1	2
1	2	3
2	3 or more	4 or more
Note that the class index is one less than the number of further echoes, because the classes are numbered from zero and no pulse in the graded set has zero further echoes. Read echo_class == 0 as "the smallest category", not as "no extra returns".

Every graded pulse produced at least one further echo, so there is no "the pulse stopped here" class. The easy discrimination — canopy versus bare ground — has already been removed.

2. depth_class — how far down the standing column the pulse burned before its last echo.

Define, for a query pulse,

r = (z_first - z_last) / max(z_first - g, 1.0)
where z_first is the height of the pulse's first echo (a point you are given), z_last is the height of its last echo (withheld), and g is the ground reference defined below. Then

value	range of r
0	r <= 0.25
1	0.25 < r <= 0.55
2	0.55 < r <= 0.85
3	r > 0.85 — the pulse burned through to the ground layer
The ground reference g is published, not withheld
g is a deterministic function of the point cloud you are given. It is not an estimate of the true terrain and it is not a hidden quantity — it is a definition, and you can reproduce it exactly. Only the numerator of r is withheld.

import numpy as np

def ground_reference(x, y, z):
    # x, y, z are float64 upcasts of the first three float32 columns of the points array.
    # Returns one reference height per point: the 5th percentile of height among all points
    # falling in the same 5 m x 5 m cell.
    ix = np.floor((x - x.min()) / 5.0).astype(np.int64)
    iy = np.floor((y - y.min()) / 5.0).astype(np.int64)
    key = iy * (ix.max() + 1) + ix
    order = np.argsort(key, kind="stable")
    key_s, z_s = key[order], z[order]
    starts = np.flatnonzero(np.append(True, key_s[1:] != key_s[:-1]))
    ends = np.append(starts[1:], len(key_s))
    g = np.empty(len(z), dtype=np.float64)
    for s, e in zip(starts, ends):
        g[order[s:e]] = np.percentile(z_s[s:e], 5.0)
    return g

Every graded pulse satisfies z_first - g >= 2.0.

Files
The dataset directory contains three CSV files and two folders.

train.csv — one row per labelled query pulse.
test.csv — one row per query pulse you must predict.
sample_submission.csv — a correctly formatted submission with placeholder values.
train/ — one .npz file per training item, named <item_id>.npz.
test/ — one .npz file per test item, named <item_id>.npz.
Columns of train.csv
There are five columns.

id — string. Unique identifier of one query pulse. This is the identifier your submission must use.
item_id — string. Identifies the item this pulse belongs to; the corresponding point cloud is train/<item_id>.npz.
query_index — integer. Row index into that item's points array identifying which pulse this is. Zero-based.
echo_class — integer in {0, 1, 2}. The first target, defined above.
depth_class — integer in {0, 1, 2, 3}. The second target, defined above.
Columns of test.csv
There are three columns: id, item_id and query_index, with the same meanings as above. The two target columns are absent.

Columns of sample_submission.csv
There are three columns: id, echo_class and depth_class, in that order. This is exactly the format your submission must have.

Contents of each .npz file
Each .npz contains two arrays.

points — float32 array of shape [N, 4], where N is between 3,000 and 60,000. The four columns are, in order: x (metres, recentred on the item), y (metres, recentred), z (metres, on an arbitrary per-item datum), and intensity (dimensionless, rescaled per item into [0, 1]). Each row is one emitted pulse, represented by its first echo only. Row order is shuffled and carries no meaning.
queries — int32 array of shape [Q]. Row indices into points marking the graded pulses. These are the same values as the query_index column.
Items are square ground patches 40 m on a side. Nothing else is provided: there is no return count, no scan angle, no timestamp, no classification, no georeference and no acquisition metadata.

Evaluation
Both targets are scored by macro-F1, each normalised against uninformed guessing, then combined.

S(F1, K) = clip( (F1 - 1/K) / (1 - 1/K), 0.01, 1.0 ) score = clip( 0.60 * S(macro_F1(echo_class), 3) + 0.40 * S(macro_F1(depth_class), 4), 0.01, 1.0 )
The held-out answers are stratified so that every class of both targets is equally frequent. Consequently uniform-random guessing scores 1/K macro-F1 on each head, and any constant submission scores less than that. Both floor at 0.01. Predicting the training prior does not help.

Higher is better. The maximum is 1.0.

Submission format
Produce a CSV with exactly three columns in this order: id, echo_class, depth_class.

Exactly one row per id in test.csv, no more and no fewer.
echo_class must be an integer in {0, 1, 2}.
depth_class must be an integer in {0, 1, 2, 3}.
No missing, non-integer, or out-of-range values.
The grader rejects a submission outright — it does not partially credit it — if the column names or order differ, if the set of ids does not match test.csv exactly, if any id is duplicated, or if any value is missing, non-integer or out of range.

Notes on the split
Training and test items are separated structurally, not randomly. No test item is drawn from the same locality as any training item, and a buffer distance is enforced between them, so neighbouring patches never straddle the split. Validate accordingly: a random split of the training items will overstate your score, because nearby patches share stand structure.

Query pulses have been sampled so that height above the ground reference and intensity have matched distributions across every target class. A model that reads only those two scalars will score at chance. The information that remains is in the three-dimensional arrangement of the surrounding first returns.

 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.