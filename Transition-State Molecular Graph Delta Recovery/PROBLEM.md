Overview
Chemical reactions transform a reactant molecular graph into a product molecular graph by removing edges, inserting edges, or replacing edge orders. The objective is to generate the missing molecular graph delta from compact atom-pair evidence near a reaction's transition state.

Each row contains four independent atom-pair evidence records. A record describes two atoms in the reactant, whether an edge initially connects them, their local chemical environments, and whether their distance contracts or expands while approaching the transition state. Generate a variable-length JSON edit program containing every deposited edge removal and edge insertion, including the exact old and new bond orders.

The underlying records come from a curated computational-chemistry archive containing atom-mapped reactant and product molecular graphs together with calculated three-dimensional reactant and transition-state coordinates. Every output edge is obtained exactly from the deposited reactant and product graphs.

The four evidence records in a row come from four different reactions. They are independent graph-edit contexts packaged into one structured output and do not describe one shared molecule.

Dataset
train.csv: 780 rows with complete molecular graph deltas.
test.csv: 300 rows whose molecular graph deltas are withheld.
sample_submission.csv: 300 deterministic randomized, schema-valid JSON edit programs. It demonstrates serialization only and is not a baseline solution.
CSV Columns
id (string): opaque unique row identifier.
bond_probe_panel_json (JSON object serialized as a string): the four atom-pair evidence records.
answer_constraints_json (JSON object serialized as a string): valid evidence handles, bond orders, output-size limit, and observable equivalence groups.
target_patch_json (JSON object serialized as a string): exact broken and formed bonds. This column appears only in train.csv.
Atom-Pair Evidence
bond_probe_panel_json contains exactly one field:

{  
  "probes": [  
    {  
      "id": "q00",  
      "endpoint_elements": "C-O",  
      "endpoint_formal_charges": "0,0",  
      "endpoint_aromatic_count": 0,  
      "endpoint_reactant_degree_bands": "2|3_plus",  
      "reactant_bond_order": 1.0,  
      "reactant_distance_band": "contact",  
      "transition_motion_band": "extension"  
    }  
  ]  
}  

The probes value is a list of four objects. Every object contains:

id (string): row-local evidence handle, one of q00, q01, q02, or q03.
endpoint_elements (string): unordered element pair such as C-H, C-N, or C-O.
endpoint_formal_charges (string): sorted reactant formal charges, such as 0,0.
endpoint_aromatic_count (integer): number of aromatic endpoints, from 0 to 2.
endpoint_reactant_degree_bands (string): sorted reactant graph-degree bands. Each endpoint is represented by 0, 1, 2, or 3_plus.
reactant_bond_order (number): deposited reactant bond order. 0.0 means no reactant bond; other possible values are 1.0, 1.5, 2.0, and 3.0.
reactant_distance_band (string enum): reactant atom-pair distance: contact for less than 1.6 angstroms, near for less than 2.5, mid for less than 4.0, and far otherwise.
transition_motion_band (string enum): transition-state distance minus reactant distance: large_contraction for at most -1.0 angstrom, contraction for at most -0.25, stable for less than 0.25, extension for less than 1.0, and large_extension otherwise.
Whole molecules, atom maps, source identifiers, SMILES strings, and Cartesian coordinate arrays are not released.

Answer Constraints
answer_constraints_json has exactly these fields:

required_probe_ids (list[string]): the four evidence handles that may appear in the graph delta.
allowed_bond_orders (list[number]): [0.0, 1.0, 1.5, 2.0, 3.0]. The value 0.0 describes absence of a bond but must not be written inside an output bond object.
maximum_changed_probe_count (integer): maximum number of distinct evidence handles that may be used across both output lists; it is 4.
symmetry_groups (list[list[string]]): a partition of the four evidence handles. Handles in the same group have identical released evidence and are interchangeable during grading.
For example, [["q00"],["q01","q03"],["q02"]] means q01 and q03 cannot be distinguished from the released fields. Writing an order for either one contributes the same group-level edge operation. Duplicate operations still matter because comparison uses multisets and therefore preserves multiplicity.

Target Format
target_patch_json contains exactly broken_bonds and formed_bonds. Each is a list containing zero to four objects with:

probe_id (string): one of the row's required evidence handles.
order (number): a nonzero bond order from 1.0, 1.5, 2.0, or 3.0.
Interpret the two lists as follows:

Pure bond cleavage: the old order appears in broken_bonds; the handle does not appear in formed_bonds.
Pure bond formation: the new order appears in formed_bonds; the handle does not appear in broken_bonds.
Bond-order replacement: the old order appears in broken_bonds and the new order appears in formed_bonds for the same handle.
Unchanged pair: the handle appears in neither list.
Examples of the three change types:

{"broken_bonds":[{"probe_id":"q00","order":1.0}],"formed_bonds":[]}  

{"broken_bonds":[],"formed_bonds":[{"probe_id":"q03","order":1.0}]}  

{"broken_bonds":[{"probe_id":"q02","order":1.0}],"formed_bonds":[{"probe_id":"q02","order":2.0}]}  

A row may contain any valid combination of these graph operations, so the serialized edit program varies in length and structure.

Submission Format
Submit a UTF-8 CSV with exactly id and target_patch_json, in either column order. Include exactly one row for every test id. Row order does not matter.

id,target_patch_json  
bondset_2af42ce99f061293,"{""broken_bonds"":[{""probe_id"":""q02"",""order"":1.0}],""formed_bonds"":[{""probe_id"":""q02"",""order"":2.0},{""probe_id"":""q03"",""order"":1.0}]}"  

Missing or extra columns, a wrong row count, duplicate ids, or a mismatched id set rejects the submission. Malformed or schema-invalid participant JSON receives zero for that row while grading continues. A malformed hidden answer raises an evaluation error.

Evaluation
For each row, the grader first replaces every evidence handle by its public symmetry-group index. It then constructs two graph-operation multisets:

(symmetry_group, order) entries from broken_bonds;
(symmetry_group, order) entries from formed_bonds.
Row fidelity is 1 only when both submitted operation multisets exactly equal their corresponding hidden multisets. Otherwise row fidelity is 0.

row_fidelity = 1 if broken_multiset and formed_multiset are both exact, else 0  
final_score = sum(row_fidelity) / 300  

Thus the leaderboard score is the mean complete-graph-delta fidelity over all 300 rows. Scores range from 0 to 1, higher is better, and a perfect submission scores exactly 1.

Expected Methods
Suitable CPU approaches include molecular graph-edit transduction, structured probabilistic models, chemistry kernels over the released evidence, edge-operation scoring, constrained JSON generation, assignment, and beam or dynamic-programming decoding.

What Not To Use
GPU, TPU, Metal, CUDA, ROCm, or any other accelerator
Network access, source lookup, hosted APIs, or external datasets
Runtime package installation or downloads
Vendored external code or remote-code loaders such as trust_remote_code=True and torch.hub.load()
Private, gated, or challenge-specific checkpoints
Hardcoded id-to-answer mappings or manual evaluation annotation
Use only the supplied files and run the complete solution on CPU.