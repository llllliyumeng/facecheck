# YTF Query-Gallery Runner: measurement and safety notes

The offline Runner uses the existing identity-disjoint Query-Gallery assignment. It selects a TAR threshold only from plaintext `threshold_selection` scores and applies that fixed threshold to both test modes.

## Missing scores and TAR/FAR

Each genuine or impostor trial remains in its trial denominator. `TAR = accepted genuine trials / all genuine trials`; `empirical FAR = accepted impostor trials / all impostor trials`. A trial without a similarity score is treated as a rejection: it contributes no true accept and no false accept. It is never replaced with a numeric score. The reports separately include total, scored, and unscored trials, unscored rates, empirical FAR over all impostor trials, and FAR resolution `1 / scored_impostor_count`.

TAR@FAR=1e-3 is `not_estimable` when fewer than 1,000 impostor pairs have actual scores, regardless of how many unscored pair rows exist. TAR remains a ratio over all genuine trials (so an unscored genuine trial is a rejection). FAR is false accepts divided by all impostor trials under the explicit reject-on-missing policy. The effective-score resolution is reported separately so missing data cannot make the score resolution appear finer.

## Query images with multiple detected faces

The Runner uses every YuNet-detected face in a Query image. Each Query-Gallery score is the maximum cosine similarity across every Query-face and Gallery-frame-face pair. A non-target Query face may contribute the maximum score. The Runner does not use the YTF annotation box to select or crop the Query face; this is an experiment limitation and must be considered when interpreting results.

## Temporary plaintext and result publication

Encrypted-restore work, including restored Gallery videos, is placed in a system temporary directory, separate from the output bundle. A restored file is removed after its features are extracted, and the temporary directory is cleaned on normal completion and handled exceptions. Temporary directory cleanup is not guaranteed after forced process termination, operating-system failure, or power loss; inspect and remove any abandoned `facecheck-ytf-work-*` directory only after confirming no Runner is active.

Result files are written to a unique staging directory next to the requested output path. The Runner validates the required result bundle before atomically publishing it. Existing output paths are never replaced; choose a new, nonexistent output directory for each run. A failure during extraction or export removes the staging bundle during ordinary exception handling. Forced process termination may leave both a private work directory and an unpublished staging directory; the final output path is only created after bundle validation, so a staging directory is not a completed experiment result.

The Runner does not currently support feature-cache persistence or checkpoint resume. An interrupted run must be repeated. Any future persistent cache should be keyed by media SHA-256, model SHA-256, feature/detection parameters, mode, and protocol version, and written atomically. Embeddings should not be reused between plaintext and encrypted-restore modes.

## Numeric scoring diagnostics and feature failures

`scoring_diagnostics.json` contains one entry for each mode, partition, and Query media. It records Query and Gallery matrix shape/dtype, input and accepted vector counts, finite/norm summaries, zero-norm and non-finite rejection counts, rejected vector indexes with media/frame/face provenance and reason, normalization results, and matrix multiplication shape/dtype/finiteness. Feature vectors themselves are never written. A Query-Gallery score is `null` when either side has no valid vectors. A non-finite matrix multiplication result or a scoring-stage exception aborts the run before candidates are ranked.

`feature_failures.csv` contains one row per detection or feature-processing failure with `mode`, `media_id`, nullable `identity_id`, `frame_index`, nullable `face_index`, `stage`, `exception_type`, and `exception_message`. Stages distinguish detector exceptions (`face_detection`), malformed detector rows (`detection_validation`), SFace/extraction errors (`feature_extraction`), and feature validation errors (`feature_validation`). Failure rows do not create cache entries; original Query/Gallery records and all protocol trial denominators remain intact. Decode failures that abort media processing are fatal run errors rather than per-frame feature failures.

The output bundle also contains `config.json`, `feature_summary.json`, `metrics.json`, `query_results.csv`, `candidate_results.csv`, `genuine_impostor_pairs.csv`, `timing.csv`, and `report.md`. The result-bundle validator checks the required filenames and CSV headers, including the diagnostic exports.

The 20-identity smoke wrapper defaults to `data/ytf/experiments/smoke_20_run_01` and refuses to overwrite it. Select a fresh destination with `FACECHECK_YTF_OUTPUT`, for example `FACECHECK_YTF_OUTPUT=data/ytf/experiments/smoke_20_run_04 bash scripts/run_ytf_smoke_20.sh`. Relative destinations are resolved from the project root. The wrapper reuses the existing permission-restricted `.env.ytf-experiment` when present; it does not print key values.

## YTF media description

Gallery MP4 files in the formal YTF manifest are reconstructed at 1 fps from the YTF frame archive. They are derived experiment inputs, not original YouTube MP4 videos. The encrypted-restore mode searches recovered plaintext after authentication and SHA-256 verification; it is not ciphertext-domain face retrieval.
