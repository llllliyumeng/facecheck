# YTF Full Run 01 Analysis

This analysis is derived from `data/ytf/experiments/ytf_full_run_01` and its
adjacent execution provenance directory. It describes the completed 1004-
identity run; it does not claim that the course-scale 10,000-identity target
was tested.

## Run Scope

- Protocol: identity-disjoint Query-Gallery; 502 identities for threshold
  selection and 502 identities for test.
- Media: 1004 Gallery videos and 1004 Query images.
- Gallery sampling: 1 fps; 191,195 sampled frames across both modes.
- Models: YuNet `face_detection_yunet_2023mar.onnx` and SFace
  `face_recognition_sface_2021dec.onnx`.
- Wall time: 5,959.2 seconds (about 99.3 minutes) on the recorded macOS
  arm64 environment.

The manifest SHA-256, identity-split SHA-256, model SHA-256 values, Python and
library versions, and source-file hashes are recorded in
`ytf_full_run_01.execution/runtime.json`.

## Retrieval Results

| Metric | Plaintext | Encrypted restore |
|---|---:|---:|
| Query count | 1004 | 1004 |
| Top-1 accuracy | 77.888% | 77.888% |
| Recall@5 | 83.964% | 83.964% |
| Detected queries | 949 | 949 |
| Scored pair rows | 474,503 | 474,503 |

The encrypted-restore scores are identical to plaintext scores for all scored
pairs (`max_absolute_similarity_delta = 0`). This verifies consistency of the
current workflow: Gallery data is encrypted, authenticated, restored and then
searched as plaintext. It does not demonstrate ciphertext-domain retrieval.

The threshold selected from the plaintext `threshold_selection` partition was
`0.4335195111`. Selection FAR was `0.00098210` with selection TAR `0.758964`.
On the held-out test partition, TAR was `0.784861` at FAR
`0.00100198`; the operating point was therefore just above the requested
`1e-3` FAR. The result is estimated because the full run has 235,556 scored
selection impostor trials, but the TAR value remains below the course target of
90%.

The measured Top-1 and TAR values do not meet the stated course targets of 95%
and 90%. A follow-up run with boundary clipping is recorded separately in
`data/ytf/experiments/ytf_full_run_02`; it reaches 77.988% Top-1 and 78.6853%
TAR, with test FAR 0.001005956. The encrypted-restore drop relative to
plaintext is 0 percentage points, but that comparison is only for the
post-restore plaintext workflow.

## Feature and Numeric Diagnostics

Each mode processed 189,582 valid features and recorded 422 feature failures.
Across both modes, `feature_failures.csv` contains 844 rows covering 38 unique
media items. All rows are `feature_validation` failures:

- 838: `face bounding box is outside the image`
- 6: `face landmarks are outside the image`

No YuNet detector exception or malformed detection row was recorded. Failed
features were skipped from the in-memory cache and remain represented as
unscored protocol trials. The run had 55 Query media with no usable feature and
4 Gallery media with no usable feature.

`scoring_diagnostics.json` reports no rejected Query or Gallery vectors, no
non-finite normalized values, and no non-finite matrix multiplication results.
The normalized Gallery vector norms were approximately 1.0. The execution log
contains no NumPy warnings, errors, or traceback entries.

## Encryption and Integrity

All 1004 Gallery videos completed encryption and authenticated restoration. The
run processed 1,299 encrypted chunks, and every restored SHA-256 matched its
source file. Temporary restored files and unpublished staging output were
cleaned after publication.

The effective Gallery encryption stage processed approximately 661 MB in 30.4
seconds, about 21.7 MB/s including the Runner's file and per-video overhead.
This is an experiment measurement, not a controlled SM4 benchmark; it is far
below the 500 MB/s course target. SM2 signing, verification, key rotation,
concurrent-terminal throughput, and true ciphertext-domain retrieval were not
measured by this run.

## Interpretation and Next Work

The storage, authentication, restore, feature-failure tracing, numeric
diagnostics, and plaintext/encrypted-restore consistency path completed
successfully. Recognition quality is the limiting result: the score
distributions overlap substantially, and 55 Query samples have no usable
feature. The next experiment should be a pre-registered diagnostic comparison
of detection/query-frame handling and feature-failure recovery, with the same
identity split and threshold protocol. Any changed detector threshold, query
selection rule, or feature aggregation rule must be recorded as a new
experiment configuration rather than applied retroactively to this result.
