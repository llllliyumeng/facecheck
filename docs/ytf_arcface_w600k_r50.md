# YTF ArcFace Candidate

## Model

- Recognition model: InsightFace `buffalo_l` / `w600k_r50.onnx` (ResNet-50 ArcFace).
- Weight source: official InsightFace v0.7 model release, distributed in the `buffalo_l` model pack.
- Weight SHA-256: `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`.
- License: InsightFace states that its supplied pretrained models are for non-commercial research purposes only. This candidate is suitable for a non-commercial course evaluation under that restriction; obtain separate permission before commercial use or redistribution.
- Runtime: OpenCV DNN ONNX, CPU backend; no InsightFace Python package is needed at inference time.
- Input: 112 x 112 BGR image aligned from YuNet's five landmarks to the standard ArcFace 112 template with a least-squares similarity transform, then converted BGR to RGB and normalized as `(pixel - 127.5) / 127.5`.
- Embedding: 512 float32 values, L2-normalized. Gallery and Query share the same detector, alignment, preprocessing, and embedding path. Similarity remains cosine similarity.

## Configuration

ArcFace is now the default for Web/API and experiment CLIs. It can also be selected explicitly:

```bash
--recognition-model arcface --arcface-model models/arcface_w600k_r50.onnx
```

Use `FACECHECK_RECOGNITION_MODEL=arcface` and optionally `FACECHECK_ARCFACE_MODEL` for Web/API. The frozen default threshold is `0.2731515063106565`, from the complete selection run below. SFace remains available through `FACECHECK_RECOGNITION_MODEL=sface`, `--recognition-model sface`, or an explicit `SFaceConfig` in Python. The `--sface-model` argument is optional and is not used for ArcFace embeddings. The default Gallery representation remains the `ytf_full_run_02` behavior: all configured 1 fps samples, all valid YuNet detections, each frame embedding retained, and pair scores aggregated by maximum. The prior quality-filter/mean-aggregation experiment remains off by default and its output is retained separately.

## Evaluation Status

The model loaded and ran locally on Apple Silicon macOS with OpenCV 4.14.0 CPU. The complete plaintext threshold-selection evaluation is recorded in `data/ytf/experiments/ytf_arcface_w600k_r50_20261101/selection_full_plaintext/`. It used all 502 existing selection identities, the unchanged formal Gallery/Query protocol, 252,004 pairs, and an in-memory per-media feature cache. The run completed in 3,504.5 seconds (feature extraction: 3,488.6 seconds); no test identities or encrypted-restore mode were accessed.

ArcFace selection results were Top-1 `433/502 = 86.25%`, Recall@5 `443/502 = 88.25%`, and TAR `86.0558%` at independently selected threshold `0.2731515063`. The corresponding SFace selection baseline is Top-1 `384/502 = 76.49%`, Recall@5 `415/502 = 82.67%`, and TAR `75.8964%`. ArcFace selection FAR was `0.0008628162` with 235,556 scored impostor trials and resolution `4.2453e-06`, meeting the `1e-3` resolution target. The selection comparison therefore meets the predeclared gate for considering a later test run; this task deliberately did not run test.

The earlier fixed 100-identity exploratory comparison remains archived in `data/ytf/experiments/ytf_models_sample_100_20261101/`. It reported SFace Top-1/Recall@5 of `79/100` and `83/100`, and ArcFace `85/100` and `86/100`; those values are exploratory and are not substitutes for the complete selection result above. The complete selection output is the authoritative ArcFace selection record, and any future test evaluation must use its frozen threshold without retuning.

The earlier smoke and interrupted-run observations remain historical context: a real-media smoke produced 139 Gallery and 1 Query normalized embeddings; a synthetic-input forward pass took approximately 20 ms. An earlier paired plaintext/encrypted-restore selection attempt was stopped before publication after its first 50 media took 244 seconds. It published no metrics and remains an interrupted attempt, separate from the completed plaintext-only run above.

See the completed selection run's `report.md` for the full SFace comparison, missing-score denominators, extraction failures, execution command and validation details. These are selection results only and do not establish held-out performance or course-target compliance. The separately authorized formal test writes to `data/ytf/experiments/ytf_arcface_test_20261101/`; only a complete execution state qualifies its metrics as formal test results. The encryption chain is unchanged.
