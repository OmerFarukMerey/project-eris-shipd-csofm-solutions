Docstring Gap Restoration
Overview
Good documentation is one of the most useful forms of supervision for code models, but real fine-tuning corpora often contain incomplete, truncated, or partially corrupted docstrings. This challenge asks you to repair a missing span in a Python documentation sentence using the surrounding sentence and the function code as context.

For each example, you receive Python function code with the original docstring removed, plus a documentation sentence where one span has been replaced by [GAP]. Your task is to generate the missing text span.

This is a sequence-to-sequence challenge: the output is free-form text, not a class label. It is not a tabular task and not a regression task.

Dataset
The prepared challenge data contains approximately:

train.csv: up to 300,000 training examples
test.csv: up to 50,000 test examples
sample_submission.csv: example submission format
Columns
| File | Column | Type | Description |

|---|---|---|---|

| train.csv | id | string | Unique row identifier. |

| train.csv | code_context | string | Python function code with the original docstring removed. |

| train.csv | masked_docstring | string | Documentation sentence containing one [GAP] marker. |

| train.csv | target_span | string | Missing text span that should replace [GAP]. |

| test.csv | id | string | Unique row identifier. |

| test.csv | code_context | string | Python function code with the original docstring removed. |

| test.csv | masked_docstring | string | Documentation sentence containing one [GAP] marker. |

| sample_submission.csv | id | string | Test row identifier. |

| sample_submission.csv | prediction | string | Generated missing span. |

Task
For every row in test.csv, generate the text that should replace [GAP] in masked_docstring.

Submission
Submit a CSV file with exactly two columns: id and prediction.

Example:

id,prediction

docgap_000001,input image

docgap_000002,training directory

docgap_000003,returns the parsed object

Evaluation
Submissions are scored using a character n-gram F-score inspired by chrF. For each prediction/reference pair, the grader computes precision and recall over character n-grams from length 1 through 6, then computes:

F = 2  *precision*  recall / (precision + recall)

The final score is the average F-score across all test rows. Scores range from 0 to 1, where higher is better.

What You Should Use
You may use CPU-friendly sequence-to-sequence or retrieval-style methods such as:

BM25 or TF-IDF retrieval from the training set
character and word n-gram features
edit-distance or fuzzy matching over similar code examples
lightweight CPU language models if they fit the runtime limit
phrase dictionaries mined from training docstrings
function-name, argument-name, and return-statement features
reranking candidates by how well they fit the visible masked sentence
What You Should Not Use
Do not use:

GPU training or inference
external API calls
hosted proprietary model APIs
internet lookup during inference
hidden raw metadata such as repository URL, commit hash, or source path
unmasked raw docstrings from test rows
hardcoded test answers
large LLM fine-tuning during scoring
methods that exceed the 1.5 hour runtime limit
tabular-only approaches that ignore the code and masked text
Compute Limits
Solutions must run on CPU only. The scoring system has 10 CPU cores and 62 GB RAM. Each solution must finish within 1.5 hours.