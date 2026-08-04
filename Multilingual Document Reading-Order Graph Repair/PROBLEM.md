Document-understanding systems must recover more than individual words. They

must determine which lines belong to the same paragraph, how lines are ordered

within each paragraph, and how paragraphs should be traversed across a page.

This becomes difficult for multi-column pages, side regions, irregular spacing,

and multilingual scripts. A simple top-to-bottom or left-to-right sort can

produce a graph that is locally plausible but globally incorrect. Such errors

propagate into screen readers, document search, information extraction, and

retrieval-augmented generation systems.

Every page in this challenge is supplied with an image, OCR line proposals,

paragraph-region proposals, and an imperfect reading-order graph. Your task is

to generate a canonical sequence of graph-edit operations that transforms the

observed graph into the intended document hierarchy and reading order.

This is a sequence-to-sequence structured-prediction problem. You are not asked

to transcribe the image or merely decide whether the graph is correct. You must

identify the incorrect relations and emit the minimal repair sequence.

Task
Each page contains two kinds of nodes:

LINE nodes representing recognized text lines.
BLOCK nodes representing paragraph or document-region candidates.
The graph uses three relation types:

CONTAINS: a block contains a line.
NEXT_LINE: one line follows another within the same block.
NEXT_BLOCK: one block follows another in page-level reading order.
The observed graph is a structurally valid alternative produced from the

intended graph by one controlled local corruption. The corruption may:

swap adjacent lines within a paragraph;
swap adjacent paragraph regions in page order;
exchange visually close lines between two paragraph regions; or
combine a cross-region line exchange with a paragraph-order swap.
All nodes remain available. Only graph edges are changed. For each test page,

predict the edge deletions and additions required to reconstruct the intended

graph.

Data
The public package contains page images, JSONL case files, CSV indices, and an

operation catalog. It contains English, Japanese, and Simplified Chinese pages.

File Structure
train.csv — training identifiers, case indices, and target repair sequences.
test.csv — test identifiers and case indices without targets.
train_cases.jsonl — graph-repair cases for the training split.
test_cases.jsonl — graph-repair cases for the test split.
images/ — WebP page images referenced by the case files.
operation_catalog.json — valid operations, relations, and ordering rules.
sample_submission.csv — a correctly formatted example submission.
The case_index column gives the zero-based line number of the corresponding

object in the relevant JSONL file.

CSV Columns
train.csv contains:

id — opaque identifier for one page.
case_index — line index in train_cases.jsonl.
repair_sequence — canonical target graph-edit sequence.
test.csv contains:

id — opaque identifier for one page.
case_index — line index in test_cases.jsonl.
Case Structure
Each JSONL case has this structure:


{

  "image_path": "images/doc_182b74fd891acfe13230.webp",

  "language": "ja",

  "width": 637,

  "height": 900,

  "nodes": [],

  "observed_edges": []

}

Nodes
A line node has this form:


{

  "node": "L017",

  "kind": "LINE",

  "bbox": [0.078125, 0.171875, 0.453125, 0.203125],

  "angle": 0,

  "text": "recognized line text"

}

A block node has this form:


{

  "node": "P004",

  "kind": "BLOCK",

  "bbox": [0.0625, 0.140625, 0.46875, 0.625],

  "angle": 0

}

Bounding boxes use normalized coordinates:


[x_min, y_min, x_max, y_max]

Coordinates lie between 0.0 and 1.0. They are deliberately coarse, noisy,

quantized detector proposals rather than exact rendering boxes. Every proposal

has positive width and height. OCR text may contain the placeholder character

□. The page image therefore remains useful evidence.

Node identifiers are randomly assigned within each page. Their numeric suffixes

do not encode position, order, paragraph membership, or corruption type.

Identical node identifiers on different pages do not imply shared meaning.

Observed Edges
Each edge is represented as one JSON array:


["CONTAINS", "P004", "L017"]


["NEXT_LINE", "L017", "L018"]


["NEXT_BLOCK", "P004", "P009"]

A valid final graph satisfies all of these rules:

Every LINE has exactly one incoming CONTAINS edge.
Every CONTAINS edge has a BLOCK source and a LINE target.
Within each block, NEXT_LINE edges form one directed path through all of that block's lines.
Every NEXT_LINE edge connects lines assigned to the same block.
NEXT_BLOCK edges form one directed path through all non-empty blocks.
No relation contains a self-loop.
The complete graph is acyclic.
The observed graph already satisfies these structural rules, so validity alone

does not reveal the answer. Layout, OCR text, language, and image evidence must

be used to select the intended graph.

Target
The target is a non-empty, semicolon-separated sequence of primitive graph

edits. Every operation uses exactly five pipe-separated fields:


OPERATION|RELATION|SOURCE|TARGET|END

The final field is always the literal token END.

Delete an Edge

DEL_EDGE|NEXT_LINE|L017|L021|END

This removes NEXT_LINE(L017, L021). The edge must exist when the operation is

applied.

Add an Edge

ADD_EDGE|NEXT_LINE|L017|L018|END

This adds NEXT_LINE(L017, L018). The edge must not already exist when the

operation is applied.

Supported Relations
The relation field must be one of CONTAINS, NEXT_LINE, or NEXT_BLOCK.

Source and target node types must match this table:

| Relation | Source | Target |

|---|---|---|

| CONTAINS | BLOCK | LINE |

| NEXT_LINE | LINE | LINE |

| NEXT_BLOCK | BLOCK | BLOCK |

Canonical Order
Operations must appear in this order:

all DEL_EDGE operations;
all ADD_EDGE operations.
Within each category, sort lexicographically by

(relation, source, target). Duplicate operations are not allowed.

A complete repair sequence can look like this:


DEL_EDGE|CONTAINS|P003|L011|END;DEL_EDGE|NEXT_LINE|L010|L014|END;ADD_EDGE|CONTAINS|P006|L011|END;ADD_EDGE|NEXT_LINE|L010|L011|END

Each target is the minimal symmetric-difference repair: every incorrect

observed edge is deleted once, and every missing intended edge is added once.

Dataset Construction
The train/test assignment is grouped by language and coarse layout family.

Pages in the same layout group cannot cross the split. No group contributes

more than 15 percent of the test rows. Each held-out page is scored separately,

and the final score is the mean of those page scores rather than an average over

layout groups.

Each source page is transformed deterministically. Line and block identifiers

are shuffled independently, proposal boxes are quantized and perturbed, OCR

text receives limited character masking, and a geometrically plausible local

alternative graph is constructed. File names, row position, identifier suffix,

operation count, and node numbering do not determine the target.

What Not to Use
Solutions must be developed from the files released in the public challenge

package. To preserve the blind evaluation, do not use:

private answers, hidden test annotations, or information obtained from the grading environment;
external copies of the source pages or their original layout annotations;
reverse-image search, OCR-text lookup, perceptual matching, or record linkage against public document archives to recover a test page's original graph;
manual annotation of individual test pages or hand-written test-specific repair sequences;
internet services or external APIs that retrieve page-level answers; or
assumptions that row order, file names, opaque page identifiers, node-number suffixes, or operation counts encode the target.
Models, algorithms, and feature engineering trained on the released training

split are allowed. General-purpose software may be used as long as it does not

retrieve external page-level labels or matching source records.

Evaluation
Submissions are evaluated using three fully specified components.

Edit-Operation F1
For each page, every predicted operation is represented by the complete tuple:


(operation, relation, source, target)

The tuple is a true positive only when all fields match a required operation

for that page:


operation_precision =

    correct predicted operations / predicted operations

operation_recall =

    correct predicted operations / required operations

operation_f1 =

    2 × operation_precision × operation_recall

    / (operation_precision + operation_recall)

Every test page requires at least one repair.

Final-Edge F1
The grader applies submitted operations to each observed graph. Every resulting

edge is represented by (relation, source, target) and compared with the

intended final graph for that page:


edge_precision = correct final edges / predicted final edges

edge_recall = correct final edges / required final edges

edge_f1 = 2 × edge_precision × edge_recall

          / (edge_precision + edge_recall)

Exact Graph Indicator
A page receives an exact-graph value of 1 only when its reconstructed edge set

is identical to the intended edge set; otherwise it receives 0:


exact_graph = 1 if final edges equal intended edges, else 0

The score for each structurally valid page is:


page_score =

    0.73 × page_operation_f1

  + 0.02 × page_edge_f1

  + 0.25 × page_exact_graph

An inapplicable edit or a resulting graph that violates a structural condition

gives that page a score of 0.0; it does not discard scores from other pages.

The final score is the arithmetic mean of all page scores. It ranges from 0.0

to 1.0, and higher is better. An exact submission scores 1.0.

The complete submission receives 0.0 if it contains any of the following:

missing, extra, or duplicate identifiers;
missing, extra, or reordered columns;
blank, non-string, non-finite, or otherwise unparsable predictions;
surrounding whitespace or empty operations;
unknown operations, relations, or node identifiers;
incorrect field counts or terminal fields;
invalid source/target node types;
duplicate operations; or
noncanonical operation ordering.
An edge deletion or addition that is inapplicable when reached, or a resulting

graph that violates a structural condition, receives 0.0 for that page only.

Submission
Submit a CSV containing exactly these columns in this order:


id,repair_sequence

Provide exactly one row for every identifier in test.csv. Complete example:


id,repair_sequence

doc_0012a4f9,"DEL_EDGE|NEXT_LINE|L003|L009|END;ADD_EDGE|NEXT_LINE|L003|L004|END"

doc_0078c1e5,"DEL_EDGE|CONTAINS|P002|L014|END;DEL_EDGE|NEXT_BLOCK|P002|P005|END;ADD_EDGE|CONTAINS|P006|L014|END;ADD_EDGE|NEXT_BLOCK|P002|P006|END"

Column names, operation names, relation names, and node identifiers are

case-sensitive.

Expected Output
For every test page, return the minimal canonical sequence of graph edits that

reconstructs the intended paragraph hierarchy, within-paragraph line order,

and page-level block order.

 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.