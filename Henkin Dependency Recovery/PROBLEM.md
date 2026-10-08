Henkin Dependency Recovery
Overview
Predict, for each queried existential variable of a Dependency Quantified Boolean Formula (DQBF), the exact set of universal variables it depends on. You see the formula's clause matrix but not its dependency prefix.

A DQBF has the form ∀u1…∀un ∃x1(D1)…∃xm(Dm). φ, where φ is a CNF matrix. Each existential xi is a Boolean function (a Henkin function) that may only read the universal variables in its dependency set Di ⊆ {u1…un}. The dependency sets are the most important part of a DQBF. They decide whether a formula is an ordinary QBF or a genuinely Henkin-quantified (NEXPTIME) problem, and DQBF preprocessors, solvers and certificate checkers all rely on them.

The formulas here are real DQBF benchmarks from the dataset Gallery whose name is not disclosed here. They cover controller synthesis, partial equivalence checking of circuits with black boxes, Ramsey-type encodings, succinct graph problems, lifted SAT instances, and random DQBFs. For every formula we removed the prefix (a/e/d lines) and the comments, renumbered all variables with a secret random permutation, and shuffled the clauses. You are told which variables are universal and which are existential, but not which universals each existential sees.

The test formulas lost their prefix through a simple damage model: each existential's dependency line survived independently with probability 8%. Survivors are given to you as hints. Your job is to reconstruct the lost lines.

Generalization is the point of the task. The test formulas come from designs, circuits, and parameter families that never appear in training. For example, if AMBA bus specifications are in test, then no AMBA specification is in train, in any encoding. Formulas that are near-copies of each other (an identical prefix and at least half of the clauses in common) are always kept on the same side. Every scored generator family appears in both train and test, so its encoding conventions can be learned. The concrete instances cannot be memorized.

Dataset
All files are in ./dataset/public/.

formulas/<formula_id>.dqx.gz — 314 gzip-compressed text files, one per formula (train and test). Format:
line 1: p cnf V C. V is the number of variables and C the number of clauses.
line 2: u <universal variable ids> 0. These are all universal variables, in ascending id order.
line 3: x <existential variable ids> 0. These are all existential variables, in ascending id order.
then C clause lines, one clause per line. Each line is a list of non-zero integer literals ending with 0. A negative literal is a negated variable. Literals inside a clause are sorted by variable id, and clause order is random.
formulas.csv — one row per formula.
formula_id (string) — opaque random id; also the file name stem.
split (string) — train (229 formulas) or test (85 formulas).
family (string) — generator family, one of bounded_synthesis, bloem_synthesis, tentrup_synthesis, partial_equivalence, scholl_henkin, random_dqbf, ramsey, succinct_graph, cnf_lifted. cnf_lifted (SAT instances lifted to DQBF, with only 2–3 existentials each) appears in train only.
n_vars, n_clauses, n_universal, n_existential (int) — size statistics.
train_labels.csv.gz — the complete dependency prefix of every train formula, with one row per existential variable (772,649 rows).
formula_id (string), variable (int, an existential id in that formula's file).
deps (string) — space-separated universal ids of the dependency set, in ascending order, or the literal {} for the empty set.
test_hints.csv — surviving dependency lines of the test formulas (13,393 rows, columns as in train_labels.csv.gz). Every test formula has at least one surviving line.
test.csv — 8,791 queries (lost dependency lines).
id (string) — opaque random query id.
formula_id (string), variable (int).
sample_submission.csv — a valid submission that predicts "depends on every universal of its formula" for every query.
The queries are a sample of the lost lines. In each test formula they include up to 400 existentials with a strict-subset dependency set, plus about half as many (at least 3) existentials that depend on all universals. A query's dependency set is never in test_hints.csv.

Evaluation
For each query existential e:

U is the universal set of its formula.
D is the true dependency set (D ⊆ U).
P is your predicted set.
The query score is the geometric mean of two Jaccard similarities: one for the dependency set, and one for the independence set (the universals e must not see):

s(e) = sqrt( J(P, D) * J(U - P, U - D) )
J(A, B) = |A ∩ B| / |A ∪ B|,  with J(∅, ∅) = 1

So s(e) = 1 only for an exact set. Predicting all of U for a query with a strict dependency set scores 0, and so does predicting the empty set for a query that depends on everything.

The final score is a weighted mean of s(e) in [0, 1], higher is better:

each of the 8 scored families has equal total weight. These are bounded_synthesis, bloem_synthesis, tentrup_synthesis, partial_equivalence, scholl_henkin, random_dqbf, ramsey and succinct_graph. Every one of them has test formulas, and cnf_lifted is train-only and never scored,
within a family, each test formula has equal weight,
within a formula, each query has equal weight.
Reference implementation (this is exactly what the grader computes):

import math

def jacc(a, b):
    u = len(a | b)
    return 1.0 if u == 0 else len(a & b) / u

def query_score(P, D, U):
    if P is None or not P <= U:          # invalid / foreign ids -> worst score
        return 0.0
    return math.sqrt(jacc(P, D) * jacc(U - P, U - D))

def evaluate(rows):
    # rows: iterable of (P, D, U, weight)
    tot = sum(w for _, _, _, w in rows)
    return sum(w * query_score(P, D, U) for P, D, U, w in rows) / tot

The weights follow the three rules above, so the formula is fully defined by formulas.csv (family) and test.csv.

Submission
Write one CSV file with exactly two columns, id and deps, and one row per query in test.csv (8,791 rows). Example using the first three real test ids (the sets are illustrative, built from each formula's own universal ids):

id,deps
q2f40f80f131c0,21921 89569 153431 186301
q2719e23ab04f1,{}
q1836e02148e91,2 12 13 17 22

Requirements
Exactly the columns id and deps. Any other column is rejected.
Exactly one row per test id. Missing ids, duplicated ids, or ids not in test.csv are rejected with an error.
deps is a list of universal variable ids of the query's own formula. Separate the ids with spaces (commas or semicolons are also accepted). Order and repeats do not matter. Use the literal {} for the empty set (none is also accepted).
A blank, NaN, unparseable, or longer than 100,000 characters deps cell, or one that contains any id that is not a universal of that formula, scores 0 for that query.
The scored metric is computed exactly as shown in Evaluation. There is no partial credit for near-miss ids.
What not to use
No external data, no other copies of the Gallery or DQBF benchmark libraries (including the original files and the generators that produced them), and no pretrained models or downloads at run time. Only use the files in ./dataset/public/.
Do not try to undo the variable renumbering by matching formulas against outside copies of the benchmarks.
Do not hand-label test queries. Predictions must come from code that runs end to end within the time limit.
Extra Modelling Information
DQBF research treats the dependency prefix as given. Solvers take it as input. Dependency schemes (for example resolution-path schemes, which also have DQBF variants) do derive dependencies from the matrix, but only one way: they prove which declared dependencies are spurious and can be removed. They never reconstruct a prefix from scratch, and they never predict which universals a Henkin function should see.

Learning work on QBF, such as learned branching heuristics for QBF solvers and GNN-based formula classifiers, uses the full prefix as an input feature. It predicts solver decisions or truth values, not the prefix itself.

This task inverts the problem. It is structured link prediction from a CNF matrix to a set-valued Henkin dependency for each existential, under design-disjoint generalization and partial supervision from surviving lines.