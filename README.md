# YOLOX persistent face tracking

Follow **one selected face** through a camera: keep its track ID while it
disappears, predict where it is, keep searching and reacquire it only when the
evidence is sufficient.

Enrolled people are **recognised**: they get a green frame with their name, and
the system can pick the target by name instead of by clicking.

Everything lives in this directory. The project does not depend on - and never
modifies - any other project in this repository.

## Requirements

* Python 3.10+ (developed on 3.13)
* `onnxruntime`, `opencv-python`, `numpy`, `scipy`, `PyYAML`
* A YOLOX model **trained on faces**, exported to ONNX

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# 1. check that the configuration is valid
python -m src.main --check-config

# 2. inspect the model before using it (prints the real output shape and
#    the input convention the model reacts to)
python -m src.inspect_model models/yolox_face.onnx

# 3. run on the camera
python -m src.main --source camera --camera-index 0

# 4. run on a video file
python -m src.main --source video --video clip.mp4
```

Controls in the live view: **click** a face to select it, `t` selects the
largest face, `c` clears the selection, `q` / `ESC` quits.

### Enrol a person and follow them by name

```bash
# 1. enrol: the camera shows the face, SPACE captures a sample,
#    s saves the profile, q quits
python -m src.enroll --name Ivan --camera-index 1

# 2. run: every enrolled person gets a green frame with his name, the
#    selected one keeps its ID even if he walks away and comes back
python -m src.main --source camera --camera-index 1 --backend haar \
    --identity --target-mode identity --target-name Ivan
```

* `python -m src.enroll --list` shows the enrolled people.
* `python -m src.enroll --remove Ivan` removes one again.
* Enrolment also works from a file: `--image photo.jpg` or `--video clip.mp4`.
* The profiles are stored in `profiles/face_database.json` (a plain JSON file,
  no database server needed).

The enrolment captures **several samples** per person, so a turned head still
matches: the similarity is the best of all samples and of the mean template.

### The model that is installed here

`models/yolox_face.onnx` is the **official Megvii YOLOX COCO model** (85 output
columns = 80 classes). It is a *person* detector, not a face detector: it
tracks people. Its export was used to verify the whole YOLOX path against real
upstream weights - raw decoding, grid decoding, NMS and the tracker all
produce correct boxes on a reference image.

For a face model, export your own trained checkpoint and replace the file:

```bash
python tools/export_onnx.py -c exp_yolox_face.py -b 1 -o yolox_face.onnx --input-size 640 640
```

(The export script belongs to the YOLOX training repository, not to this
project.) The number of classes is read from the model itself, so a 1-class
face model works without touching the configuration.

## How it works

```
frame source -> detector -> multi-object tracker -> target lock -> overlay
                    \-> identity (ArcFace) -> gallery
```

* **Detector** - a YOLOX ONNX model executed with ONNX Runtime. The export
  layout is declared in the configuration and verified against the real output
  shape; it is never guessed. The OpenCV Haar cascade is available as a
  development fallback and is never reported as a YOLOX result.
* **Tracker** - ByteTrack-style: high-score detections are matched first, low
  score detections get a second chance, unmatched tracks are *kept* as lost
  with their ID, their last box and their motion state.
* **Target lock** - the selected track is protected, keeps its ID while it is
  lost, and is reacquired only when the geometric and probabilistic evidence
  is sufficient and unambiguous. It never switches to another face on its own.
* **Identity** - ArcFace (5-point aligned, 112x112) produces a 512-D embedding
  per face. The embedding is matched against the enrolled gallery; a match
  above the threshold (default `0.6`) gives the track a name.

A track ID is **not** an identity - but the identity makes the track ID
survive: when an enrolled person walks away and comes back, the face proves
that it is the same person and the original ID is restored.

## Configuration

`config/default.yaml` documents every setting. Values can be overridden from
the environment:

```bash
YOLOX_DETECTOR__SCORE_THRESH=0.4 YOLOX_TARGET__MODE=largest python -m src.main
```

Sections: `detector`, `source`, `tracker`, `target`, `reacquisition`,
`identity`, `prediction`, `visualization`, `logging`, `runtime`.

## Scripts

```bash
python scripts/run_camera.py --camera 0
python scripts/run_video.py --video clip.mp4
python scripts/benchmark.py --synthetic
python scripts/benchmark.py --video clip.mp4 --annotations gt.json
python scripts/inspect_model.py models/yolox_face.onnx
```

## Tests

```bash
python -m pytest
```

The unit tests generate their detections, so they need neither a camera nor
model weights.

## Documentation

* `docs/implementation_plan.md` - architecture, the verified export facts, the
  target-lock design and the known limitations.
* `docs/configuration.md` - every configuration value.
* `models/README.md` - what the model file has to look like.

## Limitations

* The installed model is the COCO person detector, so what is tracked is a
  *person*, not a face. A face-trained YOLOX export is still needed for the
  actual use case; only the plumbing is verified with the current file. Until
  then use `--backend haar`, which detects faces but is less robust.
* A fully occluded or out-of-view face cannot be observed; the track can only
  be preserved and tried again later.
* The recognition threshold (`identity.threshold`, default `0.6`) has to be
  calibrated with the real camera and the real person.
* Only the CPU execution provider is available in this environment.
