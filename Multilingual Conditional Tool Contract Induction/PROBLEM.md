Multilingual Conditional Tool Contract Induction
Overview
Production routing contracts often contain conditions that are invisible in a tool name: a request is accepted only when an argument is present, or only when an argument is absent. Each row contains six multilingual requests accepted by one unnamed tool and four rejected requests. Two rejected requests come from the same underlying behaviour but violate its hidden argument gate; two come from a nearby, distinct behaviour.

Return the raw target and peer tool tokens together with the conditional interface:

accepted requests + same-purpose gate violations + nearby negatives -> conditional tool contract

Broad tool recognition is only one component. A solution must also infer the gate argument, decide whether that argument is required or forbidden, recover the accepted interface, and separate the peer behaviour from same-tool policy violations.

Solutions must use CPU only, at most 10 CPU cores and 62 GB RAM, and finish within 90 minutes.

Dataset
train.csv: 900 labeled episodes from 30 latent tools.
test.csv: 450 unlabeled episodes from 15 entirely unseen latent tools.
sample_submission.csv: 450 schema-valid empty placeholder contracts.
Columns:

contract_id: opaque unique string identifier.
positive_examples_json: JSON array of exactly six accepted examples. Each object has string locale and string request. No intent, argument label, source identifier, or literal-span annotation is provided.
contrast_examples_json: JSON array of exactly four rejected examples in shuffled order. Two are same-purpose gate violations and two represent one nearby behaviour. Each object has string locale and string request; group membership is hidden.
argument_registry_json: JSON array of 55 permitted argument objects. Each object has string name, string description, and categorical string type equal to string. The same broad registry is used for every episode, so registry membership does not reveal the target interface.
induction_requirements: JSON object containing string rules named required_threshold, optional_threshold, routing_rule, and peer_rule.
induced_contract: train-only JSON target string.
Requests span Arabic, German, English, Spanish, French, Hindi, Indonesian, Japanese, Russian, Swahili, Turkish, and Simplified Chinese. Every episode uses ten distinct underlying semantic groups.

Target Contract
induced_contract must be one JSON object with exactly five keys:

target_tool: raw lowercase underscore-delimited tool token for the six accepted examples, such as alarm_remove.
routing_rule: object with exactly two string fields:
argument: one name from the row registry;
operator: exactly required or forbidden. required means accepted requests contain the argument and same-purpose negatives omit it; forbidden means the reverse.
required_arguments: lexicographically sorted array of unique registry names expressed in at least three accepted examples.
optional_arguments: lexicographically sorted array of unique registry names expressed in one or two accepted examples.
peer_tool: raw lowercase underscore-delimited tool token for the distinct nearby behaviour in the contrast set.
Required and optional arrays must be disjoint. The routing argument and all submitted interface arguments must come from the registry.

Example:

{"target_tool":"alarm_remove","routing_rule":{"argument":"date","operator":"required"},"required_arguments":["date"],"optional_arguments":["time"],"peer_tool":"alarm_query"}

Evaluation
target_tool and peer_tool use exact string accuracy. Required and optional arguments use exact set F1. Correct empty-set agreement scores 1 for that component; an incorrect empty/non-empty pairing scores 0. Routing score is the mean of exact gate-argument accuracy and exact operator accuracy.

The component base is worth 90 percent:

10 percent target-tool accuracy;
30 percent routing-rule score;
20 percent required-argument set F1;
20 percent optional-argument set F1;
10 percent peer-tool accuracy.
Let interface be the mean of required and optional set F1. The balance multiplier is:

0.5 + 0.5 * min(target_tool, routing, interface, peer_tool)

The row score is:

component_base * balance_multiplier + 0.10 * exact_complete_contract

exact_complete_contract is 1 only when the parsed five-field object exactly matches the canonical contract. A perfect contract scores 1.0; tool-token recognition alone cannot compensate for a failed conditional policy.

Raw tool values may contain only lowercase letters, digits, and underscores and may be at most 64 characters. A value that violates this content bound receives zero for that row; it does not abort or invalidate other submission rows. Structural failures such as malformed JSON, wrong keys, missing IDs, duplicate IDs, or unregistered argument names remain submission errors.

The final score is the mean row score clipped to [0, 1]. If one identical non-empty contract is submitted for more than 20 percent of test rows, the final score is capped at 0.10.

Submission Format
Submit a CSV with exactly these columns in this order:

contract_id,induced_contract

Every test ID must appear exactly once. JSON must be CSV-escaped normally:

contract_id,induced_contract  
ct_00dec2ba1717b9f829b7,"{""target_tool"":"""",""routing_rule"":{""argument"":"""",""operator"":""required""},""required_arguments"":[],""optional_arguments"":[],""peer_tool"":""""}"  
ct_010bb671af20a22dbbc8,"{""target_tool"":"""",""routing_rule"":{""argument"":"""",""operator"":""required""},""required_arguments"":[],""optional_arguments"":[],""peer_tool"":""""}"  

These placeholders are schema-valid but score zero.

Requirements
Use CPU only and finish within 90 minutes.
Fit learned vocabularies, thresholds, and task-specific models on train.csv only.
Aggregate all accepted and rejected examples at episode level.
Produce strict JSON with exactly the documented keys and permitted registry names.
Use a genuinely learned semantic method; deterministic logic may validate and render predictions.
Prohibited Leakage
Do not use private answers, unreleased labels, external label tables, hidden identifiers, internet services, or external-corpus record matching. Do not reconstruct released requests through search, translations, metadata, or a separately obtained copy of their source corpus. CUDA and GPU-only dependencies are prohibited.

 

Expected Output
Your script receives the public dataset directory and exact submission CSV path as two positional arguments.