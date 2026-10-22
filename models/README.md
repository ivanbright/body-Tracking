# Models

## What is needed

A YOLOX model **trained on faces**, exported to ONNX:

```
models/yolox_face.onnx
```

No such model is installed in this repository. The unit tests therefore do not
need one, but the detection quality cannot be demonstrated until one is
installed.

## What does *not* work

* A COCO "person" model (for example `yolox.tflite`) is **not** a face
  detector. It finds bodies, not faces. The TFLite adapter was removed from
  this project for that reason.
* A model that was not exported with the YOLOX export script has an unknown
  output layout and is rejected with an actionable error.

## Checking a model

```bash
python -m src.inspect_model models/yolox_face.onnx
```

The tool prints the input tensor (name, shape, dtype, layout), the output
tensors and the value range of the output. It also lists the output layouts
whose column count matches the model.

## Choosing the configuration

| Output of the model | `detector.output_format` | `detector.num_classes` |
| --- | --- | --- |
| `(1, N, 5 + C)`, box in stride units (official export default) | `yolox_raw` | number of classes |
| `(1, N, 5 + C)`, box already in input pixels (`--decode_in_inference`) | `yolox_decoded` | number of classes |
| `(1, N, 4 * reg_max + 1 + C)` | `yolox_dfl` | number of classes |
| `(1, N, 6)` `[x1, y1, x2, y2, score, class_id]` | `xyxy` | - |

The default is `yolox_raw`, because `tools/export_onnx.py` uses
`action="store_true"` for `--decode_in_inference` and therefore leaves the
decoding to the client.

`detector.activation` stays `none`: the YOLOX head already applies the
sigmoid to the objectness and class scores, so a second sigmoid would destroy
them.

The input resolution is taken from the model itself when it is static.

## Exporting a YOLOX model

```bash
python tools/export_onnx.py \
    -c /path/to/yolox_face.py \
    -b /path/to/model_ckpt.pth \
    -o models/yolox_face.onnx
```

Add `--decode_in_inference` only if you also set
`detector.output_format: yolox_decoded`.
