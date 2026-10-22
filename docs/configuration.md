# Configuration reference

Every value in `config/default.yaml` is listed here. The YAML file is the
single source of truth; this document explains what each value does and what
it must not be set to.

## How configuration is loaded

1. `config/default.yaml` is read first.
2. The file passed with `--config` is merged on top of it.
3. Environment variables are applied last, so they win over both files.
   The pattern is `YOLOX_<SECTION>__<KEY>`, for example:

```bash
set YOLOX_DETECTOR__BACKEND=haar
set YOLOX_DETECTOR__SCORE_THRESH=0.5
set YOLOX_TARGET__MODE=track_id
```

`--check-config` prints the merged configuration and exits without opening a
camera, which is the fastest way to confirm that a file was parsed correctly.

Command-line flags (`--model`, `--backend`, `--num-classes`, `--target-mode`,
`--target-track-id`, `--source`, ...) override the merged configuration for the
values they cover.

Relative paths in the configuration are resolved against the `yolox/` project
root, not against the current working directory.

## detector

| Key | Default | Meaning |
| --- | --- | --- |
| `backend` | `yolox_onnx` | `yolox_onnx` (primary) or `haar` (development fallback only). |
| `model_path` | `models/yolox_face.onnx` | ONNX weights for the `yolox_onnx` backend. |
| `labels_path` | `""` | Optional label list, one class name per line, used for the overlay text. |
| `input_size` | `[640, 640]` | Network input resolution `[width, height]`. A static input size declared by the model wins over this value. |
| `score_thresh` | `0.3` | Minimum confidence a detection must reach to be kept. |
| `nms_thresh` | `0.45` | IoU threshold of non-maximum suppression. |
| `target_classes` | `[]` | Keep only these class names. Empty means keep everything. Ignored by the `haar` backend. |
| `providers` | `[CUDAExecutionProvider, CPUExecutionProvider]` | ONNX Runtime execution providers in preference order. CPU is always appended as a fallback. |
| `input_layout` | `auto` | `auto`, `nchw` or `nhwc`. `auto` reads the layout from the model's input tensor. |
| `resize_mode` | `letterbox` | `letterbox` (YOLOX reference) or `stretch`. |
| `input_range` | `0_255` | Value range of the network input. `0_255` is the YOLOX reference convention (the head was trained on 0-255 float32 inputs), `0_1` divides by 255. |
| `color_order` | `rgb` | Channel order of the network input. `rgb` is the YOLOX reference convention, `bgr` keeps the order of the source image. |
| `output_format` | `yolox_raw` | Output layout of the ONNX export. See below. |
| `activation` | `none` | `none` or `sigmoid`. YOLOX applies the sigmoid inside the head, so exported scores are already probabilities. |
| `num_classes` | `1` | Number of classes of the model. A single-class face model has `1`. Used to validate the real output shape. |
| `strides` | `[8, 16, 32]` | Strides of the detection head, used by the grid decoding. |
| `reg_max` | `16` | `reg_max` of a distribution-focal-regression head. |
| `warmup` | `true` | Run one dummy inference at start-up so the first real frame is not slow. |
| `num_threads` | `0` | Intra-op threads for ONNX Runtime. `0` means runtime default. |

### output_format

The decoder is never guessed from the data. Run
`python -m src.inspect_model <model.onnx>` and set the value that matches the
model you actually exported.

| Value | Model output | Notes |
| --- | --- | --- |
| `yolox_raw` | `(1, N, 5+C)` | `[cx, cy, w, h, obj, cls...]` still in stride units. Needs grid + stride decoding. This is Megvii's `tools/export_onnx.py` **default**, because `--decode_in_inference` is a `store_true` flag and therefore off by default. |
| `yolox_decoded` | `(1, N, 5+C)` | Same columns, box already decoded into network-input pixels (`--decode_in_inference`). |
| `yolox_dfl` | `(1, N, 4*reg_max+1+C)` | Distribution-focal regression logits. Documented, but not yet verified against real upstream weights. |
| `xyxy` | `(1, N, 6)` | `[x1, y1, x2, y2, score, class_id]`. |

Raw decoding is `(offset + grid) * stride` for the box centre and
`exp(size) * stride` for the box size.

### input_range and color_order

These two settings are not cosmetic: a model fed the wrong input range returns
**no detections at all**, which looks like a broken decoder.

| Convention | Result on the official Megvii `yolox_s.onnx` |
| --- | --- |
| `0_255` + `rgb` | correct detections (verified) |
| `0_255` + `bgr` | correct detections, slightly noisier (verified) |
| `0_1` + `rgb`/`bgr` | **no detections** (verified) |

`python -m src.inspect_model <model.onnx> --image <some photo>` runs the model
with all four conventions and reports which one the model reacts to. Use that
output if your model returns nothing.

### The `haar` backend

`haar` is an OpenCV Haar cascade. It exists so the tracking, target-lock and
reacquisition logic can be exercised without any trained weights. It is not a
YOLOX result and must never be reported as one.

## source

| Key | Default | Meaning |
| --- | --- | --- |
| `type` | `camera` | `camera` or `video`. |
| `camera_index` | `0` | OpenCV camera index. |
| `video_path` | `""` | Video file, used when `type == video`. |
| `width` | `640` | Requested capture resolution (camera only). |
| `height` | `480` | Requested capture resolution (camera only). |
| `fps` | `30.0` | Assumed capture frame rate, used to convert frame counts to seconds. |
| `loop` | `false` | Loop the video file when it ends. |

A camera that reports a nonsensical frame rate (for example 1000 fps from a
broken driver) is rejected: the value is clamped to the plausible range
1-240 fps and a warning is logged, otherwise the time-based retention limits
would be meaningless.

## tracker

| Key | Default | Meaning |
| --- | --- | --- |
| `track_high_thresh` | `0.5` | Detections at or above this score are matched first. |
| `track_low_thresh` | `0.1` | Lower-scoring detections get a second association chance against unmatched tracks (the ByteTrack idea). |
| `match_thresh` | `0.8` | Maximum association cost. The cost blends `(1 - IoU)` and normalised centre distance, so `1.0` means "no overlap at all, far away". |
| `min_hits_to_confirm` | `3` | Hits required before a tentative track becomes confirmed. |
| `max_init_frames` | `15` | A tentative track that is not confirmed within this many frames is dropped. |
| `max_lost_time` | `90` | Maximum number of consecutive missed frames before a lost track expires. |
| `lost_retention_seconds` | `3.0` | Wall-clock retention of a lost track. The effective limit is `min(max_lost_time` frames, `lost_retention_seconds`). |
| `max_lost_tracks` | `32` | Maximum number of lost tracks kept in the pool at the same time. |
| `max_active_tracks` | `32` | Maximum number of active tracks. |
| `distance_weight` | `0.35` | Weight of the normalised centre-distance term in the association cost. `0.0` = pure IoU, `1.0` = pure distance. |
| `use_motion` | `true` | Use the Kalman velocity estimate to widen or narrow the association gate. |

## target

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | `none` | `none`, `largest`, `track_id` or `identity`. `identity` selects the enrolled person named in `name`. |
| `track_id` | `0` | Track selected at start-up when `mode == track_id`. |
| `name` | `""` | Enrolled person selected when `mode == identity`. Required in that mode; the identity layer is switched on automatically. |
| `lost_retention_seconds` | `10.0` | How long the selected target is kept after it was last observed. When the limit is reached the target expires: the ID is released, the system stops searching and **no other face is silently substituted**. |
| `require_confirmed` | `true` | Only tracks confirmed by several observations may be selected. |
| `auto_reselect` | `false` | Pick a new target automatically when the current one expires. Off by default: the system must not switch to another person on its own. |

`largest` is a geometric heuristic. It does **not** identify a person.

## reacquisition

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Master switch for the lost-track search. |
| `score_threshold` | `0.55` | Weighted evidence score required to restore a lost track after a short gap. |
| `max_distance_ratio` | `2.5` | Hard gate: the detection centre may not be further than `max_distance_ratio * (track box diagonal)` away from the predicted centre. |
| `min_size_ratio` | `0.5` | Hard gate on the size ratio (detection size / last observed size). |
| `max_size_ratio` | `2.0` | Hard gate on the size ratio. |
| `ambiguity_margin` | `0.08` | If the runner-up candidate is within this margin of the best candidate, the evidence is ambiguous and the match is rejected. |
| `distance_weight` | `0.40` | Evidence weights (they do not have to sum to 1). |
| `overlap_weight` | `0.20` | Evidence weight of the overlap term. |
| `velocity_weight` | `0.15` | Evidence weight of the velocity term. |
| `size_weight` | `0.10` | Evidence weight of the size term. |
| `confidence_weight` | `0.15` | Evidence weight of the detection confidence. |
| `time_decay_tau` | `2.0` | Time decay constant in seconds. |
| `long_gap_seconds` | `2.0` | After this many seconds without an observation the motion prediction is no longer trusted. |
| `long_gap_score_multiplier` | `1.3` | Multiplier applied to `score_threshold` for long gaps. |
| `long_gap_max_distance_ratio` | `1.5` | Tighter spatial gate for long gaps. |
| `long_gap_min_confidence` | `0.35` | Minimum detection confidence accepted for long gaps. |
| `appearance_enabled` | `false` | Optional appearance matching. Off by default. |
| `appearance_weight` | `0.0` | Weight of the appearance term. |
| `appearance_threshold` | `0.0` | Threshold of the appearance term. |

Decay semantics, because this is easy to get wrong: the decay discounts the
**positional evidence only**. Size, confidence and velocity are not decayed.
After a long gap the reference is the last observed box, which does not drift,
so no decay is applied at all and the stricter long-gap rules carry the burden.

An unvalidated face similarity score may never be the reason a track is
restored, and it may never create a match on its own.

## identity

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Master switch for the identity layer (ArcFace + gallery). |
| `embedder_path` | `models/embedder_arcface.onnx` | ArcFace ONNX weights. |
| `landmarker_path` | `models/face_landmarker.task` | MediaPipe face landmarker used for the 5-point alignment. |
| `providers` | `[CPUExecutionProvider]` | ONNX Runtime execution providers for the embedder. |
| `threshold` | `0.6` | Minimum cosine similarity for a name to be given. Same person: `0.6 - 0.99`, different people: below `0.4`. |
| `gallery_path` | `profiles/face_database.json` | The enrolled people. Plain JSON, written by `src.enroll`. |
| `gallery_interval` | `5` | Number of frames between two identity checks of the visible faces. |
| `max_embeddings` | `8` | Number of pose samples kept per track. |
| `alignment` | `auto` | `auto` (5-point landmarks when available, otherwise a plain crop), `five_point` (landmarks only) or `crop`. |
| `min_face_size` | `40` | Faces smaller than this (in pixels) are not embedded. |
| `crop_margin` | `0.15` | Relative enlargement of the crop around the detector box. |
| `reference_min_similarity` | `0.4` | Observations less similar than this to the known poses are ignored, so the reference cannot drift onto a different person. |
| `reference_interval` | `3` | Number of frames between two reference updates. |
| `max_distance_ratio` | `15.0` | Distance gate (relative to the track diagonal) that still applies when the identity confirmed the candidate. The face is the evidence then: a person who walked across the room is still that person. |
| `min_size_ratio` | `0.25` | Size ratio (detection size / last observed size) still accepted when the identity confirmed the candidate. |
| `max_size_ratio` | `4.0` | Size ratio still accepted when the identity confirmed the candidate. |

The identity never creates a track. It only gives a name to a track and
restores the ID of an **existing** lost track when the face is recognised.

## prediction

| Key | Default | Meaning |
| --- | --- | --- |
| `max_prediction_time` | `1.0` | How long a predicted position is shown and used for association after the last reliable observation. |
| `max_prediction_distance` | `0.0` | Maximum distance in pixels the prediction may travel. `0` = unlimited. |

A prediction is an estimate, never a proof that the face is still there.

## visualization

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Master switch for the overlay window. |
| `show_predictions` | `true` | Draw predicted boxes. |
| `show_trail` | `true` | Draw the motion trail of each track. |
| `trail_length` | `30` | Number of trail points kept. |
| `window_name` | `YOLOX Face Tracking` | Title of the overlay window. |
| `show_dashboard` | `true` | Draw the statistics dashboard. |

## logging

| Key | Default | Meaning |
| --- | --- | --- |
| `level` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `file` | `""` | Optional log file. Empty means log to the console only. |
| `format` | see `default.yaml` | Log line format. |

## runtime

| Key | Default | Meaning |
| --- | --- | --- |
| `detection_interval` | `1` | Run the detector every N frames. The tracker still runs on every frame, so track IDs stay stable while detection is throttled. |
| `max_frames` | `0` | Stop after N frames. `0` = unlimited. |
| `record_output` | `""` | Optional path for an annotated video of the session. |
| `save_debug_frames` | `false` | Save annotated frames to disk for debugging. |
