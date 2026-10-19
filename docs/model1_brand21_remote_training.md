# Model 1: remote training for the 21-brand source order

The training/checkpoint class IDs intentionally follow the source `data.yaml` order:

`AXE, BEST_FOODS, CLEAR, CLOSE_UP, COMFORT, DOVE, HAZELINE, KNORR, LIFEBUOY, LUX, OMO, PONDS, PS, REXONA, SIMPLE, SUNLIGHT, SUNSILK, SURF, TRESEMME, VASELINE, VIM`

These IDs are separate from the stable application registry IDs recorded in
`mapping.json`. In particular, VIM is training ID 20 and stable registry ID 1.
Do not resume a checkpoint trained with the older registry-ordered v1 dataset.

## Transfer the prepared dataset

Datasets, runs, and weights are intentionally excluded from Git. Copy the complete
local directory below to the same repository-relative path on the server:

```text
data/brand_21brands_yolo_v2/
```

The directory must include the images, labels, `data.yaml`, `mapping.json`,
`dataset_checks.json`, `dataset_fingerprint.json`, and `dataset_report.json`.
The expected prepared-dataset fingerprint is:

```text
sha256:8ed180b7ac1def9732f2e61452c6d301a814ef32a62b17517cbd723767d0d492
```

## Set up the server

From the repository root, create an environment and install a CUDA-compatible
PyTorch build for the server first. Then install the pinned project dependencies:

```bash
python -m pip install -r requirements.txt
```

The initial `yolov8n.pt` weight is not committed. Download it into the repository
root through Ultralytics if it is not already present:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Validate, then launch manually

Preflight does not construct a model or start training:

```bash
python scripts/train_detector.py --config configs/brand_21brands_yolo.yaml --preflight
```

Confirm that preflight reports all of the following before training:

- `class_order` is `source_data_yaml`;
- `dataset_fingerprint` matches the value above;
- `source_to_training_mapping_verified` is `true`;
- the selected CUDA device is correct;
- the run directory does not already exist;
- `model_constructed` and `training_started` are `false`.

Start the manual run only after those checks pass:

```bash
python scripts/train_detector.py --config configs/brand_21brands_yolo.yaml
```

The configured output is:

```text
runs/model1_brand21/yolov8n_100ep_s42_sourceorder_v2/
```

The workflow refuses to overwrite an existing run and does not evaluate the test
split for checkpoint selection or tuning.
