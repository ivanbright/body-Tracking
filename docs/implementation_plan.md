# Implementation plan - YOLOX persistent face tracking

## Goal

Track faces with a YOLOX detector and **follow one selected target** through
disappearances: keep its track ID, predict where it is, keep looking and
reacquire it only when the evidence is sufficient.

The project is self-contained: it lives in `yolox/` and depends on no other
project in this repository.

## Scope

**In scope**

* ONNX Runtime backend for a YOLOX model exported to ONNX (`yolox_onnx`).
* OpenCV Haar cascade backend (`haar`) - explicitly a **development fallback**;
  results produced with it are never reported as YOLOX results.
* Multi-object tracker (ByteTrack-style association, Kalman motion model).
* Target lock with an explicit life cycle and duration-aware reacquisition.
* Identity layer (ArcFace + a gallery of enrolled people): names for the
  enrolled faces, the target selectable by name, and the ID of a lost track
  restored when the face is recognised again.
* Overlay with observed / predicted / target rendering and a small dashboard.
* Configuration (YAML + environment overrides), logging, evaluation tooling.

**Out of scope - deliberately**

* MQTT, message brokers, tachometer/needle or any other hardware output.
* Any JSON "signal" feed for downstream consumers.
* Any modification of the other projects in this repository.

## Architecture

```
frame source -> detector -> tracker -> target lock -> renderer
                   \-> identity (ArcFace) -> gallery
                                    \-> evaluation (metrics / benchmark)
```

| Module | Responsibility |
| --- | --- |
| `src/camera/capture.py` | Camera and video sources, deterministic release. |
| `src/detection/preprocessing.py` | Letterbox/stretch, blob layout, inverse transform. |
| `src/detection/postprocessing.py` | Explicit YOLOX decoders, NMS, shape validation. |
| `src/detection/yolox_onnx.py` | ONNX Runtime backend, output-layout verification. |
| `src/detection/haar.py` | Development fallback. |
| `src/recognition/arcface.py` | The ArcFace embedder and the cosine similarity. |
| `src/recognition/alignment.py` | 5-point alignment, plain crop as the fallback. |
| `src/recognition/gallery.py` | The enrolled people, their samples and the mean template. |
| `src/recognition/identity.py` | Matching a face against the gallery. |
| `src/enroll.py` | The enrolment CLI (camera, image, video, list, remove). |
| `src/tracking/motion_prediction.py` | 8-state constant-velocity Kalman filter. |
| `src/tracking/association.py` | IoU + centre-distance cost, Hungarian assignment. |
| `src/tracking/tracker.py` | Track creation, association, retention, expiry. |
| `src/tracking/reacquisition.py` | Evidence scoring, duration-aware rules. |
| `src/tracking/target_lock.py` | Selection, protection, life cycle of the target. |
| `src/visualization/renderer.py` | Overlay and dashboard. |
| `src/evaluation/` | Metrics (IDF1, HOTA, ID switches) and the benchmark. |

## The export format must be declared, never guessed

Verified against the upstream YOLOX sources:

* `yolox/models/yolo_head.py` (evaluation branch) emits
  `cat([reg, obj.sigmoid(), cls.sigmoid()])` - the scores are **already
  probabilities** and must not be squashed again.
* `YOLOXHead.decode_outputs` decodes the box as
  `(offset + grid) * stride` and `exp(size) * stride`.
* `tools/export_onnx.py` sets `decode_in_inference` from a `store_true` flag,
  which is **off by default** - the official export returns the **raw** head
  output.

Supported layouts: `yolox_raw` (default), `yolox_decoded`, `yolox_dfl`
(documented convention, not verified against real weights), `xyxy`.
The configured layout is checked against the model's real output shape; a
mismatch is an error, never a silent misinterpretation.
`python -m src.inspect_model <model.onnx>` prints the real shape and the
compatible layouts.

## The target lock

States: `IDLE -> LOCKED -> LOST -> REACQUIRED -> LOCKED`, plus `EXPIRED` and
`CLEARED`.

Guarantees:

* The selected track keeps its **ID** while it is lost and after it is found.
* The lock never switches to another face on its own; a new face always gets a
  new ID. `auto_reselect` is off by default.
* The selected track is *protected*: it survives longer than an ordinary lost
  track (`target.lost_retention_seconds`).
* Every state change is logged and can be shown in the overlay.

Not claimed: "largest face" is a geometric heuristic, not an identity
decision.

## The identity layer

A track ID is not an identity, but the identity makes the track ID survive.

```
camera -> 5-point aligned crop -> ArcFace -> 512-D embedding -> gallery
```

* The gallery holds **several samples per person** plus the normalised mean
  template. The similarity of a face is the **maximum** over the template and
  over every sample, so a turned head still matches the sample of that angle.
* A face above `identity.threshold` (default `0.6`) gets the name of the
  enrolled person; the overlay draws that person with a **green frame**.
* `target.mode: identity` selects the track whose name matches
  `target.name`.
* When a track is lost, every candidate face is matched against the gallery. A
  candidate that is recognised gets the **original** track ID back. The
  identity never creates a track and never changes the selection; it only
  restores what already existed.
* The distance gate is relaxed when the identity confirms the candidate
  (`identity.max_distance_ratio`), because a person who walked across the room
  is still that person - the face is the evidence, not the position. The
  distance is then measured against the **last observed** box, not against a
  motion prediction that is seconds old.
* Two candidates that are both confirmed but too similar to each other are
  ambiguous: both are rejected, the system must not guess.

## Duration-aware reacquisition

| Gap | Reference | Distance gate | Required score | Confidence |
| --- | --- | --- | --- | --- |
| short (< `long_gap_seconds`) | motion prediction | `max_distance_ratio` | `score_threshold` | - |
| long (> `long_gap_seconds`) | last observed box | `long_gap_max_distance_ratio` | `score_threshold * long_gap_score_multiplier` | `long_gap_min_confidence` |

The evidence is a weighted score (distance, overlap, velocity, size,
confidence). The time decay applies to the **positional evidence only**:
there the motion prediction is the reference, and a prediction that is seconds
old is a guess. The evidence that does not depend on the prediction (size,
confidence, velocity) and the overlap - measured against a box that does not
drift - are not decayed. In the long-gap branch the reference is the last
observed box, so there is nothing to discount and the stricter rules carry
the burden.

Ambiguity guard: if a runner-up candidate is within `ambiguity_margin` of the
best one, nothing is restored - the target simply stays lost.

Appearance matching is optional, disabled by default and can only *subtract*
evidence, never create a match.

## Overlay

* Every observed face with its track ID.
* The selected target in a distinct style, with its state, the time since it
  was last seen and the ID it will keep.
* The predicted position, drawn differently (dashed) and labelled as a
  prediction - never mixed with the observed boxes.
* Dashboard: FPS, inference latency, number of visible / lost tracks,
  detections, active backend.
* Controls: click a face to select it, `t` selects the largest face, `c`
  clears the selection, `q` / `ESC` quits.

## Testing

* Unit tests use generated detections - no camera, no weights, no network.
* The ONNX integration test exports a small model with `torch.onnx.export`
  and runs it through ONNX Runtime. It proves the plumbing, not the accuracy.
* The benchmark reports measured numbers and labels the data source
  (`synthetic` or `recorded-video`). IDF1/HOTA are only reported when a
  ground-truth annotation is supplied.

## Phases

| Phase | Content | State |
| --- | --- | --- |
| A | Review of the existing code and the plan | done |
| B | Plan, configuration, explicit export formats | done |
| C | Detection layer (pre/post-processing, ONNX backend, inspection tool) | done |
| D | Tracking layer (Kalman, association, tracker, reacquisition) | done |
| E | Target lock | done |
| F | Pipeline, sources, overlay, logging | done |
| G | Documentation and configuration reference | done |
| H | Evaluation, benchmark, scripts | done |
| I | Tests and verification | done (unit level) |

## Known limitations

* **No face-trained YOLOX model is installed.** The detection quality can
  therefore not be demonstrated here; only the plumbing is verified. See
  `models/README.md`.
* A face that is fully occluded or out of view cannot be observed. The system
  can preserve the track and try again when the person returns, but it cannot
  promise that the person is still at the predicted position.
* The `yolox_dfl` layout could not be verified against real weights.
* Only the CPU execution provider is available in this environment, so the
  measured latencies are CPU latencies.
* Without an annotated video no real IDF1/HOTA number can be produced.
