# YTF Gallery Frame Quality and Aggregation

The YTF Runner keeps the Query image detection, alignment, and SFace extraction
path unchanged. For each Gallery video, it starts with the configured 1 fps
samples and selects at most 12 frames at evenly spaced indices. Each sampled
frame contributes at most one face: the eligible YuNet face with the highest
confidence (larger box breaks a tie).

The defaults require detection confidence >= 0.9, face-box area >= 1.5% of the
image area, and at least 75% of the original box area to lie inside the image.
Invalid, non-finite, degenerate, or low-quality boxes are skipped. A failed
detection or SFace extraction affects that frame only. Every accepted SFace
vector is L2-normalized, the vectors are averaged, and the mean is L2-normalized
again. A Gallery with no valid frame has no feature; no substitute score/vector
is generated.

These settings are exposed by `RunnerConfig` and the `ytf_runner` CLI. The
Runner records the active settings and Gallery quality counts in the output
bundle. `--evaluation-scope threshold_selection` evaluates only identities
assigned to threshold selection and chooses the operating threshold from
plaintext selection pairs. `--evaluation-scope test` requires
`--frozen-selection-threshold`; it evaluates only test identities and does not
re-estimate the threshold. The default `full` scope retains the existing paired
full-protocol behavior.
