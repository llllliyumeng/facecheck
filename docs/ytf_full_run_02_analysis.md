# YTF Full Run 02 Analysis

This is the full 1004-identity follow-up run using the same manifest, identity
split, SFace model, YuNet threshold, scoring rule, and TAR protocol as
`ytf_full_run_01`. The change under test is safe clipping of slightly
out-of-bounds YuNet boxes and landmarks before SFace extraction.

| Metric | Plaintext | Encrypted restore |
|---|---:|---:|
| Query count | 1004 | 1004 |
| Top-1 accuracy | 77.988% (783/1004) | 77.988% (783/1004) |
| TAR at frozen threshold | 78.6853% (395/502) | 78.6853% (395/502) |
| Test FAR | 0.001005956 | 0.001005956 |
| Scored pair rows | 475,505 | 475,505 |

The threshold was selected only from the threshold-selection partition at
`0.4335195111`, with selection FAR `0.00098210`. The held-out test FAR is just
above `1e-3`, so the operating point is not marked as met. Both recognition
metrics remain below the project targets of 95% Top-1 and 90% TAR.

Relative to run 01, Top-1 increases by 0.100 percentage points and TAR by
0.199 percentage points. The run processed 191,195 sampled frames and 189,992
features per mode, with 12 feature extraction failures per mode. All 1,004
gallery videos were encrypted and authenticated-restored; all 1,299 chunks
were processed and source/restored SHA-256 values matched.

The encrypted-restore scores have a maximum absolute difference of zero from
plaintext scores. This validates the encrypted storage and restore path only;
it is not a ciphertext-domain retrieval measurement.
