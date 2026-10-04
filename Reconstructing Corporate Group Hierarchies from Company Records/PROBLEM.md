Overview
A corporate group is not one company. It is a tree of separate companies: a parent at the top, holding companies under it, operating subsidiaries under those, special-purpose vehicles, finance subsidiaries registered in another country, funds the group consolidates. Each member reports which entity directly consolidates it in its accounts. Put the reports together and you have the group's structure.

This challenge hands you the group without its structure. For each corporate family you get the list of member entities - registered name, legal form, jurisdiction, cities, registration date - and you are told which member is the ultimate parent. Every link between the members is hidden. Reconstruct the tree: for every member, the member directly above it.

The obvious answer is instructive precisely because it is useless. In the training families, most entities hang directly off the ultimate parent, so "put everything under the top" reproduces most links while recovering none of the structure. The score is built so that answer is worth exactly zero: it counts only the intermediate links - an entity sitting under a holding company, a sub-holding, a regional parent - which are the part of a group's shape that has to be inferred.

Whole families are held out. No test family, and no member of one, appears in training. What transfers is how groups are built: which legal forms sit above which, how holding chains run through jurisdictions, what a sub-holding's name looks like next to the companies under it, which entities were registered together.

The task
For each held-out family, output its tree as a set of links: one parent for every member except the ultimate parent. A parent is always another member of the same family.

Data
train_entities.csv - 1,136 families, 13,383 entities
train_parents.csv - one row per non-root training entity: entity_id, parent_entity_id
test_entities.csv - 479 families, 4,940 entities; the links are withheld
sample_submission.csv - 4,461 rows: every test entity placed directly under its ultimate parent
The entity files share these columns:

family_id - meaning: the corporate family; links never cross families
entity_id - meaning: the entity. Opaque, assigned in random order, carries no signal
is_ultimate_parent - meaning: 1 for the family's ultimate parent, 0 otherwise
legal_name - meaning: the registered legal name, as filed, in its original script
legal_form - meaning: a four-character code for the entity's legal form, or OTHER: followed by the form as filed when it has no code
jurisdiction - meaning: the legal jurisdiction, ISO country or subdivision code
legal_city, legal_country - meaning: the registered legal address
hq_city, hq_country - meaning: the headquarters address
registered_on - meaning: the date the entity's record was first registered
Registry identifiers, company-register numbers and every relationship field are withheld. Families have between 5 and 183 members, 7 at the median, and entities are registered in 130 countries. Training families are 2.4x the test families, and every training link is given.

Evaluation
A link is an (entity, parent) pair. Links to the ultimate parent are never counted; every other link is an intermediate link. Pooled over all graded entities:

tp = your intermediate links that are correct
fp = your intermediate links that are wrong
fn = true intermediate links you did not produce
score = F1 = 2 tp / (2 tp + fp + fn)

Placing an entity under the ultimate parent is therefore never rewarded and never penalised as a wrong link; it only forgoes a link. A parent outside the entity's family, or the entity itself, is a wrong intermediate link. There are no other terms and no hidden weights. The score runs from 0 to 1.

What you are up against
Measured with the shipped grader on the 479 held-out families. The bootstrap standard error of the reference's score, resampling families, is 0.0222.

sample_submission.csv (every entity under the ultimate parent) - score: 0.0000
a random other member of the family as parent - score: 0.0830
name matching: the member sharing the longest run of leading name words - score: 0.1251
reference: pairwise "is j directly above i?" model, one parent per entity - score: 0.2680
Name matching is the rule a solver writes first: a sub-holding usually shares the leading words of its subsidiaries' names. It is fooled by numbered vehicles, renamed acquisitions and chains that change language at a border. The reference scores every (entity, candidate parent) pair inside a family with a gradient-boosted model over the pair's names, legal forms, jurisdictions, cities and registration dates, then takes the best non-root candidate whenever its share against the ultimate parent clears a threshold tuned on held-out training families. It scored 0.3350 on training families it had not seen and runs in about 2 minutes on a CPU. It clears name matching by 6 standard errors and leaves most of the score range above it.

Design notes
Why links to the top are not scored. In the training families most links point at the ultimate parent, so any metric that counts them rewards the star-shaped guess. Scoring only the intermediate links keeps every point of the score on structure that has to be recovered.

Which families are included. Only closed families: every member's reported direct parent is itself a member, so every link has its answer inside the file. A family also needs at least five members and at least one intermediate link. Families containing a sole proprietor, and families containing a record marked as a duplicate registration, are left out. Every member of a family can be told apart by its visible attributes, so no link has two right answers.

Allowed, and not allowed
Any model trained inside your script. Pretrained language models are allowed.
No external data and no network. Looking up any entity, relationship or register record in an outside source is outside the task.
Each held-out family is reconstructed on its own. Use its members and the training data only: no statistic pooled across test families, no vocabulary or vectorizer fitted on test names, no pseudo-labelling from your own predictions.
Validate by holding out whole families. This is a requirement on your script, not advice: a random split over entities places a family on both sides and will overstate your score.
family_id and entity_id carry no signal and row order is randomised. Hard-coded predictions keyed on either are not allowed.
Explicitly allowed: anything learnt from the training families - their names, forms, jurisdictions, registration patterns and the shapes of their trees.
Your script must be deterministic and must finish inside the compute budget: CPU only, 1 hour.
Submission
submission.csv, exactly two columns:

entity_id,parent_entity_id
E000856,E000985
E004358,E000985

One row per non-root entity in test_entities.csv (the rows of sample_submission.csv), no duplicates. Row order does not matter. The grader scores only the entities it is grading and ignores any other row, so submit the full file every time.
An empty parent_entity_id counts as placing the entity under its ultimate parent.
Extra, missing or renamed columns, duplicate entity_id values, or a missing row for an entity being graded all raise and score nothing. A parent id that is not a member of the family is simply a wrong link.
import pandas as pd

test = pd.read_csv("test_entities.csv", dtype=str, keep_default_na=False)
rows = [(e, choose_parent(e, test)) for e in test.loc[test.is_ultimate_parent == "0", "entity_id"]]
pd.DataFrame(rows, columns=["entity_id", "parent_entity_id"]).to_csv("submission.csv", index=False)

What this benchmark does not model
The links are accounting consolidation as reported by the entities, not legal share ownership, and they are as current as the source snapshot. Entities outside the source registry are absent, which is why only closed families are kept. Some links are supplied by the entity alone rather than checked against documents; they are used as reported. Legal names are reproduced as filed, in the script they were filed in. Provenance and licence are on the dataset page.