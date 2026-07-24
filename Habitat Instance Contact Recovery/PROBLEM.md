Habitat Instance Contact Recovery
Overview
Field survey systems must count individual target organisms without merging nearby bodies into one detection. This is especially difficult when organisms overlap, touch, or rest within the same plant cluster. To reduce uplink bandwidth and avoid retaining identifiable raw habitat photographs, the supplied images are three-channel edge-and-texture telemetry views rather than RGB photographs. Your task is to locate every visible target instance and recover a contact graph that indicates which detected instances touch or nearly touch.

The hidden set is separated from training by capture session, not by random image row. It also includes confirmed empty scenes and look-alike organisms. Every telemetry view is deterministically derived from one real photograph; there are no generated or composited records. Good solutions must therefore generalize across recording conditions, reject hard negatives, separate crowded instances, and produce a consistent graph.

Dataset
File descriptions
train.csv: labeled training rows.
test.csv: unlabeled test rows.
train/images/: training JPEG edge-and-texture telemetry views.
test/images/: test JPEG edge-and-texture telemetry views.
sample_submission.csv: valid submission structure with random example predictions.
Column descriptions
train.csv:

id: opaque image identifier.
image_path: path relative to dataset/public.
instances: JSON list of ground-truth instances.
test.csv:

id: opaque image identifier.
image_path: path relative to dataset/public.
Each object in instances contains:

instance_id: unique string within that image.
bbox: [x_min, y_min, x_max, y_max], normalized to [0, 1].
contacts: list of instance_id values for other instances in the same image.
Two ground-truth boxes are contacts when the Euclidean gap between their rectangular boundaries is at most 0.02 of image width/height. Touching and overlapping boxes have zero gap. Each relation is undirected; listing it from either or both endpoints has the same meaning.

The public data contain 1,919 labeled training images. The hidden set contains 691 images from held-out capture sessions.

Evaluation
Predicted boxes are matched one-to-one to ground truth with Hungarian assignment at IoU >= 0.50.

Detection F1 is computed per image and averaged equally across three hidden object-count strata: empty, one instance, and two-or-more instances. Contact edges are mapped through the matched boxes. Topology quality combines edge F1 with matched-node coverage and is averaged equally across multi-instance images with and without contacts.

The final score is the geometric mean of the two components:


detection_score = mean([empty_f1, single_f1, multi_f1])

topology_score = mean([no_contact_topology, contact_topology])

score = sqrt(detection_score * topology_score)

Either component reaching zero forces the final score to zero. The geometric mean prevents strong localization from masking a weak contact graph, while the stratified component averages prevent abundant easy scenes from hiding failures on empty or crowded cases. Higher is better; the range is 0 to 1.

Submission
Submit submission.csv with:

id: every test identifier exactly once.
instances: JSON list using the schema above.
Example:


id,instances

00111dc6a64d8234,"[{""instance_id"":""p1"",""bbox"":[0.112,0.244,0.251,0.396],""contacts"":[]}]"

0050f30778a61e22,[]

Requirements
Include exactly the columns id,instances and every test ID once.
Use finite normalized box coordinates with x_max > x_min and y_max > y_min.
Use unique instance IDs within each image.
Contact references must point to predicted instances in the same row and cannot contain self-links.
Use only the supplied public data. Do not recover the source corpus, filenames, or hidden annotations through external lookup.
 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.