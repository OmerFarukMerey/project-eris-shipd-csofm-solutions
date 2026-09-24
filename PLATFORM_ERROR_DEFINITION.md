# Platform Error Definitions

Submission-time errors reported by the Project Eris platform, their likely or confirmed cause, and the fix.

| Error | Likely/confirmed cause | Solution |
| --- | --- | --- |
| Exit code `2` | Usually `argparse` rejected platform-supplied arguments. Current case is likely but unconfirmed without stderr. | Use `parse_known_args()`, support standard `--data-dir`, `--output`, and `--threads`, then test the platform command. |
| `FileNotFoundError` | Incorrect hardcoded dataset path or missing referenced asset. | Default to `./dataset/public/`, resolve asset paths relative to it, and validate every required file before training. |
| Wrong CSV columns | A solution or CSV from another challenge was submitted. | Read `sample_submission.csv`; enforce exact column names, order, row count, and IDs before writing. |
| Missing prediction values | Empty strings became CSV null values. | Validate all predictions as non-empty before saving and reload the finished CSV with `keep_default_na=False`. |
| Generic `ValueError` | Schema, malformed JSON, tensor shape, or invalid output value. The exception name alone does not reveal which. | Add stage/row context and preserve the full traceback; reproduce using the exact final source and command. |
| Generic `RuntimeError` | Commonly CUDA/deterministic-operation conflicts, unavailable devices, OOM, or thread initialization. | Test on the target environment; avoid unsupported deterministic CUDA operations; use fixed batches and explicit device checks. |
| Deterministic execution rejected | Time-based stopping, environment-dependent fallbacks, dynamic worker counts, or unseeded operations. | Fix seeds, epochs, batches, workers, threads, model selection, and backend settings. Never change work based on elapsed time. |
| Source cannot be inspected | Oversized, encoded, compressed, generated, or overly complex source. | Keep `solution.py` plain UTF-8, directly inspectable, and below the platform's 512,000-byte limit. |
| Prompt compliance rejection | External/pretrained data, test fitting, forbidden metadata, or code the checker cannot understand safely. | Use only explicitly permitted data and methods; remove test adaptation and opaque execution paths. |
| Script succeeds locally but fails remotely | Mac and Kaggle/A10G environments differ in paths, libraries, CUDA kernels, or launch arguments. | Run a clean Linux/A10G deployment replay when possible; Mac success is only a smoke test. |
