# YTF 20-Identity Diagnostic Comparison

This comparison uses the fixed smoke manifest and identity split. It does not
change the Query-Gallery protocol and is not a formal 1004-identity result.

## Runs

| Arm | Change | Output |
|---|---|---|
| B0 | YuNet score threshold 0.90 | `data/ytf/experiments/ytf_20_svc_01_baseline` |
| D1 | YuNet score threshold 0.85 | `data/ytf/experiments/ytf_20_svc_02_det085` |

Both runs used the same manifest, identity split, models, sampling, Gallery/
Query pairing, threshold selection protocol, and plaintext/encrypted_restore
workflow. `encrypted_restore` encrypts Gallery media, verifies and restores it,
then searches restored plaintext; it is not ciphertext-domain retrieval.

## Results

| Metric | B0 plaintext | D1 plaintext | B0 encrypted_restore | D1 encrypted_restore |
|---|---:|---:|---:|---:|
| Top-1 | 0.85 (17/20) | 0.90 (18/20) | 0.85 (17/20) | 0.90 (18/20) |
| Recall@5 | 0.90 (18/20) | 1.00 (20/20) | 0.90 (18/20) | 1.00 (20/20) |
| Detected Query count | 19/20 | 20/20 | 19/20 | 20/20 |
| Scored pairs | 180/200 | 200/200 | 180/200 | 200/200 |
| Feature failure rows | 44 | 54 | 44 | 54 |

The extra D1 feature-failure rows are diagnostic validation failures; they were
not converted to zero vectors or removed from denominators. B0 and D1 each had
zero non-finite normalized vectors and zero non-finite matrix multiplication
results. Plaintext and encrypted_restore pair scores were identical within each
run, with maximum absolute difference 0.0. Gallery restore SHA-256 checks
passed for all 20 videos in both runs.

TAR@FAR=1e-3 remains `not_estimable`: the selection partition has only 90
impostor trials. These smoke results are diagnostic and do not establish the
course target metrics.

## Q1 status

Q1 would replace each Query image with the first readable frame from its same
source video. In the current smoke preparation, each of the 20 Query source
directories contains only one readable frame, which is already the manifest's
selected frame. There is therefore no independent Q1 variable available from
the current prepared inputs. Running it would duplicate B0 rather than test a
new condition, so Q1 was not run.

## Additional algorithm arms

`ytf_20_svc_04_det080` lowered the YuNet threshold to `0.80`. It produced the
same Top-1 (`0.90`) and Recall@5 (`1.00`) as D1, but increased diagnostic
feature-failure rows from 54 to 60. D1 is therefore the better detector setting
among these smoke arms.

`ytf_20_svc_05_topkmean` used the D1 detector threshold and replaced the per-
video maximum score with the mean of the three highest frame/face similarities.
It produced the same Top-1 (`0.90`) and Recall@5 (`1.00`) as D1, with exact
plaintext/encrypted_restore score consistency. The new aggregation is optional
and the historical `max` behavior remains the default; this smoke run does not
justify changing the formal experiment scoring rule.
