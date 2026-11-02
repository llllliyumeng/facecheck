# YTF Full Run 01 Failure Audit

This audit is read from `data/ytf/experiments/ytf_full_run_01`. It does not
change the published result bundle or rerun the experiment.

## Feature-failure records

`feature_failures.csv` contains 844 rows:

| Mode | Records | Affected media | Gallery/video records | Query/image records |
|---|---:|---:|---:|---:|
| plaintext | 422 | 38 | 416 | 6 |
| encrypted_restore | 422 | 38 | 416 | 6 |
| Total | 844 | 38 unique media | 832 | 12 |

The same Gallery and Query media fail in both modes. All records are
`feature_validation` failures raised after YuNet returned a detection:

| Exception message | Records |
|---|---:|
| `face bounding box is outside the image` | 838 |
| `face landmarks are outside the image` | 6 |

There were no `face_detection` exceptions, malformed detection records, or
SFace extraction-stage exceptions in this run. Every record retains the mode,
media ID, identity ID, frame index, face index, exception type, and message.

## Unscored media

The protocol keeps all 1004 Query and 1004 Gallery records. A missing feature
does not become a zero vector and does not receive a fabricated score.

- Query: 949 media were scored, 53 had no detected face, and 2 had a detected
  face whose feature failed validation.
- Gallery: 1000 videos had at least one usable feature; 4 had no detected face
  and therefore contributed no gallery feature.
- Candidate rows preserve the unscored status: 53,212 rows are
  `query_face_not_detected`, 2,008 are `query_feature_extraction_failed`, and
  3,790 are `gallery_face_not_detected` across both modes.

The 55 Query records without a usable feature remain in the 1004-query
denominator. Consequently, the reported Top-1 and Recall@5 values are
protocol-level values, not values computed after silently removing failed
queries. Among the 949 scored Query records, the conditional Top-1 rate is
782/949 = 82.40% and the conditional Recall@5 rate is 843/949 = 88.83%; these
are diagnostic views and do not replace the primary metrics.

## Split distribution

The Query failures are distributed as 29 in `threshold_selection` and 26 in
`test`. The four Gallery videos with zero usable features are three in
`threshold_selection` and one in `test`. Threshold selection therefore still
uses only the prescribed threshold-selection partition, and the test partition
is evaluated with that fixed threshold.

## Numeric and mode consistency

`scoring_diagnostics.json` records 1004 scoring entries per mode. The full run
reported zero rejected Query vectors, zero rejected Gallery vectors, zero
non-finite normalized vectors, and zero non-finite matrix-multiplication
outputs. The normalized Gallery norm range was approximately 1.0. The runner
log contains no NumPy warning, error, or traceback.

Plaintext and encrypted-restore scored pairs are identical, with
`max_absolute_similarity_delta = 0`. This validates encryption, authenticated
restore, and post-restore plaintext search consistency. It does not establish
true ciphertext-domain retrieval.

## Follow-up experiment design

The next recognition-quality experiment should be a new output directory with
the same manifest, identity split, and threshold-selection rule. It may compare
detector threshold or query-frame handling, but each change must be recorded in
the new configuration and must not overwrite `ytf_full_run_01`.
