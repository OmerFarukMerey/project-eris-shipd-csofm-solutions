Habitat Dossier Normalization: turn a messy field record into a clean ecology profile
Overview
Field biologists rarely get clean data. A real occurrence record is a jumble of tag lists, half-filled free-text, dict-style notes, and geography — some of it inconsistent or mislabelled. Turning that mess into a tidy, analysis-ready profile is a daily task, and it rewards care.

In this challenge each item is one animal's messy field record (higher taxonomy, behavioural tags, diet, mating notes, and biogeography). Your job is to normalize it into a strict 7-field ecology profile. Two kinds of field:

Extraction fields — the evidence is present in the record, but messy. You must parse it and map it onto a controlled vocabulary, resolving synonyms and conflicts (e.g. a record tagged both Nocturnal and Diurnal normalizes to cathemeral).
Inference fields — the source has been held out of the record. You must reason about them from the remaining ecology and geography. Specifically, an animal's habitats and climate are never stated in the record; you infer them.
This is not a fit-one-classifier task. It is graded per field, and items span a difficulty gradient: harder records (more conflicts, broader ranges, more inference) count for more, so a thorough, well-reasoned solution scores materially higher than a quick one.

The 7 fields and their controlled vocabularies
Extraction (single value each):

activity — one of: nocturnal, diurnal, crepuscular, cathemeral, unknown
locomotion — one of: terrestrial, arboreal, aquatic, semiaquatic, fossorial, volant, unknown
social — one of: solitary, social, unknown
reproduction — one of: viviparous, oviparous, ovoviviparous, unknown
trophic_guild — one of: carnivore, herbivore, omnivore, insectivore, piscivore, unknown
Inference (a set; zero or more, semicolon-separated):

habitats — any of: Agricultural, Caves, Coastal, Forest, Freshwater, Grassland, Marine, Mountains, Rainforest, Rocky areas, Savanna, Shrubland, Wetlands
climate — any of: tropical, temperate, cold, arid, polar
Use unknown for a single field when the record does not support a confident value. Matching is case-insensitive and tolerant of common synonyms and plurals (e.g. ocean→Marine, egg-laying→oviparous).

Evaluation
Each field contributes a field score, fields are combined by weight into an item score, and items are combined by difficulty weight into the final score.

For animal i with difficulty tier t_i (an integer 1–4):

single-value field: s_f = 1 if the normalized prediction equals the gold value, else 0
multi-value field: s_f = 2 · |A ∩ B| / (|A| + |B|) (set Dice/F1 between predicted set A and gold set B; 1.0 if both empty)
item score: item_i = ( Σ_f W_f · s_f ) / ( Σ_f W_f )
final score: overall = ( Σ_i t_i · item_i ) / ( Σ_i t_i ), in [0, 1], higher is better.
Field weights W_f — the genuinely hard, held-out habitats field dominates, so the score tracks habitat-inference quality rather than the easily-parsed fields:

habitats = 20
climate = 2
activity = 1, locomotion = 1, social = 1, reproduction = 1, trophic_guild = 1
Note on the split: train and test hold out entire taxonomic orders — every Order in the test set is absent from train. You therefore cannot memorise a taxonomy→habitat lookup; the habitats and climate fields must be generalised to unseen taxa. This is deliberate and is why strong solutions land well below a perfect score.

The exact metric (this is what the grader computes):

def evaluate(gold_rows, pred_rows, tiers, W):
    SINGLE = ["activity","locomotion","social","reproduction","trophic_guild"]
    MULTI  = ["habitats","climate"]
    num = den = 0.0
    for g, p, t in zip(gold_rows, pred_rows, tiers):
        item = 0.0
        for f in SINGLE:
            item += W[f] * (1.0 if canon(f, p[f]) == canon(f, g[f]) else 0.0)
        for f in MULTI:
            A, B = canon_set(p[f]), canon_set(g[f])
            dice = 1.0 if not A and not B else (0.0 if not A or not B
                    else 2*len(A & B)/(len(A)+len(B)))
            item += W[f] * dice
        item /= sum(W.values())
        num += t * item; den += t
    return num / den

Dataset
The public/ folder contains:

train.csv — 4962 rows: id, record (the messy input), tier (difficulty 1–4), and the seven gold fields, so you can learn/validate your normalization.
test.csv — 1413 rows: id, record only (from taxonomic orders not present in train).
sample_submission.csv — 1413 rows in the exact submission format (a constant baseline).
Across the data an animal occupies about 2.58 habitats on average; many single fields are legitimately unknown for a given animal. Difficulty tiers in train are roughly 781/3156/741/284 for tiers 1/2/3/4.

Submission
A CSV with exactly these columns, one row per test id (1413 rows plus header):

id, activity, locomotion, social, reproduction, trophic_guild, habitats, climate

Single fields hold one token; habitats and climate hold a semicolon-separated set (possibly empty). A valid submission looks exactly like this (these are real test.csv ids):

id,activity,locomotion,social,reproduction,trophic_guild,habitats,climate
AD-003664,nocturnal,fossorial,solitary,viviparous,carnivore,Forest;Grassland;Rainforest;Savanna;Shrubland,temperate;tropical
AD-000671,nocturnal,arboreal,social,viviparous,herbivore,Forest;Freshwater,tropical
AD-003279,diurnal,terrestrial,solitary,viviparous,insectivore,Agricultural;Forest;Grassland;Rainforest;Savanna;Shrubland,tropical

Requirements
Provide exactly the 1413 test ids, each exactly once. A missing, duplicate, or unrecognised id makes the submission invalid.
Column names must be exactly the seven field names plus id. Extra columns are ignored; a renamed or missing required column is invalid.
Single fields must be one token from that field's vocabulary (use unknown if unsure). Values outside the vocabulary score as wrong for that field.
habitats and climate are semicolon-separated sets; order does not matter and duplicates are collapsed. An empty set is allowed.
Matching is case-insensitive and tolerant of common synonyms/plurals; unresolvable tokens are dropped, never fatal.
Allowed methods and what not to use
This is a closed-book task. You must produce every prediction using only the provided files and your own on-the-fly reasoning over them. There is no internet at solve time.

Allowed — the only inputs you may use:

public/train.csv (records + gold fields + tier) and public/test.csv (records) — including any statistics, patterns, or models you learn from these files only.
Your own general reasoning about the text in each record. Standard, offline libraries in the runtime image (pandas, scikit-learn, etc.) for modelling.
This is what you must not use and will lead to immediate disqualification of your solution — you cannot use external source verification or external-style locomotion/volancy reasoning while developing the solution.'
Prohibited — using any of these disqualifies the submission, regardless of score:

The internet / any network access at solve time — no web search, no browsing, no API calls, no runtime downloads, no fetching model or data files.
External datasets, trait databases, gazetteers, or knowledge bases about these organisms — explicitly including (but not limited to) TetrapodTraits, EltonTraits, AmphiBIO, PanTHERIA, GBIF, iNaturalist, the IUCN Red List, Wikipedia/Wikispecies, or any field guide or encyclopedia. Do not import, hard-code, paste, or otherwise inject facts from such sources (e.g. a taxon's volancy, diet, or habitat).
Re-identifying the masked species (its name and name-bearing text are removed) and then using pretrained or memorised knowledge of that species or its taxon to look up any answer field. Reasoning must generalise from the record, not recall an external fact about the specific animal or family.
Pretrained model weights that encode external biological knowledge used to supply answers (as opposed to a model you fit on train.csv).
Reading, reconstructing, or reverse-engineering the private answer key; and relying on the id or row order (ids are randomised and carry no signal).
Enforcement is harness-side: solutions run in a no-network, no-external-data environment, and any run whose visible trajectory shows web search or use of an external biological reference is disqualified even if its score is high. If you are unsure whether a source is allowed, it is not — use only train.csv and the record in front of you.

Prior work / why this is novel
This is a real, openly-licensed corpus (see the dataset documentation for the exact CC0 source), not a synthetic set and not a known benchmark. Its mechanism is deliberately different from the usual single-label trait classification: it is a graded, difficulty-weighted normalization task that combines extraction of messy in-record evidence with inference of deliberately held-out fields, scored per field so that thoroughness and reasoning both move the score. The nearest neighbours — trait databases and species-description NLP — expose the target in the input, predict a single class, or are graded by one flat metric; none pose held-out-field inference over a masked, messy record under a difficulty-weighted, per-field score.

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.