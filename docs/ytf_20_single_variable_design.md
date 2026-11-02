# YTF 20-Identity Single-Variable Diagnostic Design

This is a pre-registered diagnostic design based on the completed
`ytf_full_run_01` audit. It does not modify or rerun that result. No formal
experiment is started by this document.

## Fixed protocol

- Dataset: `data/ytf/smoke_20/experiment_smoke_20.json`.
- Identity list and assignment: `data/ytf/smoke_20/identity_split_smoke_20.json`.
  The 10 `threshold_selection` and 10 `test` identities must remain byte-for-
  byte unchanged across all arms.
- Gallery/query pairing, source-group disjointness, pair construction, missing
  score handling, and ranking are unchanged.
- Models: YuNet `face_detection_yunet_2023mar.onnx` and SFace
  `face_recognition_sface_2021dec.onnx`.
- Modes: plaintext and encrypted restore are paired with the same feature and
  scoring parameters. Encrypted restore still means decrypt, authenticate,
  restore, and search plaintext; it is not ciphertext-domain retrieval.
- Target FAR remains `1e-3`. The 20-identity selection partition has only 90
  impostor trials, so `TAR@FAR=1e-3` is expected to be `not_estimable`. Report
  Top-1, Recall@5, detection/feature availability, score distributions, and
  failure counts as the diagnostic outcomes. Do not compare an invented TAR
  point to the 1004-identity result.
- Every arm uses a fresh, nonexistent output directory under
  `data/ytf/experiments/`; `ytf_full_run_01` and its execution directory are
  read-only reference artifacts.

## Baseline and one-variable arms

| Arm | Variable changed | Fixed value / change | Suggested output |
|---|---|---|---|
| B0 | none | YuNet score `0.90`, NMS `0.30`, Gallery 1 fps, existing annotated Query image, all detected Query faces | `ytf_20_svc_01_baseline` |
| D1 | YuNet score threshold | score `0.85`; NMS, sampling, Query selection, models, and protocol unchanged | `ytf_20_svc_02_det085` |
| Q1 | Query frame selection | replace each manifest Query image with the first readable frame from that same annotated source video; detector and all Gallery processing unchanged | `ytf_20_svc_03_query_first` |

D1 isolates detector recall/precision sensitivity. Q1 isolates the effect of
the held-out Query frame choice. Do not combine D1 and Q1 in one run. If a
first-frame replacement is not available or cannot be mapped unambiguously to
the existing `source_group_id`, mark Q1 unavailable rather than changing the
identity set.

The existing Query image is the annotated frame selected by the YTF adapter;
the current Runner does not crop to the annotation box and retains every
detected face. That behavior is the B0 contract. For Q1, retain the same
all-detected-faces rule so only the source frame changes.

## Failure-sample annotation

Failure handling is observational and identical in every arm. Keep the
original media and pair denominators; never replace a failed feature with a
zero vector or delete the Query. Export `feature_failures.csv` with mode,
media, identity, frame index, face index, stage, exception type, and message.
Summarize counts by:

1. `face_detection` and `detection_validation`;
2. `feature_extraction` and `feature_validation`;
3. Query versus Gallery, and `threshold_selection` versus `test`.

The primary annotation labels are `query_face_not_detected`,
`query_feature_extraction_failed`, `gallery_face_not_detected`, and
`gallery_feature_extraction_failed`. A changed detector threshold may change
these counts, but the label definitions and denominator policy must not.

## Run order and acceptance checks

1. Run B0 into its new directory and verify the manifest and identity-split
   hashes match the fixed input files.
2. Run D1, then Q1, each with its own new output directory. Do not start a
   full 1004-identity experiment from these diagnostics.
3. Before comparing metrics, verify each bundle contains the required reports,
   both modes, and complete 10+10 query/gallery partition counts.
4. Compare each arm with B0 using paired Query IDs: Top-1 changes, detected
   Query count, usable feature count, failure labels, and score deltas.
5. Treat encrypted-restore equality as a consistency check only. Any mismatch
   requires stopping the comparison and auditing the bundle.

The three commands below are intentionally examples only; they are not
executed as part of this design review. They must load the existing
permission-restricted `.env.ytf-experiment` without printing key material.

```sh
FACECHECK_YTF_OUTPUT=data/ytf/experiments/ytf_20_svc_01_baseline \
  bash scripts/run_ytf_smoke_20.sh

FACECHECK_YTF_OUTPUT=data/ytf/experiments/ytf_20_svc_02_det085 \
FACECHECK_YTF_SCORE_THRESHOLD=0.85 \
  bash scripts/run_ytf_smoke_20.sh
```

Q1 requires a manifest-generation step that preserves the 20 existing
identity assignments and records the new manifest hash in its output config;
it is deliberately not generated implicitly by the Runner.
