# Sign Language Recognition

Real-time American Sign Language (ASL) alphabet recognition from a webcam.
The system detects one hand with MediaPipe, converts its 21 landmarks into a
normalized 63-value feature vector, classifies the vector with a PyTorch 1D
CNN, and exposes the result through a FastAPI service.

> **Current scope:** the project recognizes isolated hand signs, not continuous
> sign-language sentences. The collected dataset represents the 26 English
> alphabet letters (A-Z). The application also supports `space` and `del`
> actions when those labels are included in the loaded model.

## Features

- Real-time inference from a webcam.
- MediaPipe hand detection and 21 `(x, y, z)` landmarks.
- Wrist-relative normalization and scale normalization.
- PyTorch CNN1D classifier with CPU support and optional CUDA acceleration.
- Temporal stability checks and prediction buffering to reduce accidental
  characters.
- Confidence thresholding (`0.75` by default).
- Text accumulation with `space`, `del`, repeated-letter handling, and English
  spell correction.
- Two clients:
  - a browser UI at `/ui`;
  - an OpenCV desktop webcam client.
- FastAPI validation, interactive Swagger documentation, and health check.
- Unit tests for preprocessing, stability, prediction actions, frame
  processing, reset behavior, and API error handling.
- Optional Locust scenario for endpoint load testing.

## System flow

```text
Webcam frame
    │
    ├── OpenCV / browser capture
    │
    ├── POST /ws (base64 JPEG/PNG)
    │
    ├── MediaPipe Hands
    │       └── 21 landmarks × 3 coordinates
    │
    ├── Wrist-relative + scale normalization
    │
    ├── CNN1D inference
    │
    └── Stability / confidence filtering
            └── current text, predicted sign, FPS, confidence
```

## Requirements

- Python `>=3.10,<3.13` (Python 3.10, 3.11, or 3.12).
- A webcam for local real-time use.
- The model checkpoint and label encoder in `models/`.
- CPU works by default. For NVIDIA GPU inference, install the PyTorch build
  matching the local CUDA version from the
  [official PyTorch selector](https://pytorch.org/get-started/locally/).

Runtime dependencies are pinned in [`requirements.txt`](requirements.txt).
There are no separate development or load-testing requirements files in this
repository; install extra tools such as `pytest`, `black`, `flake8`, and
`locust` separately when you need them.

## Installation

```bash
git clone https://github.com/mu-3mar/sign-language-recognition.git
cd sign-language-recognition

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the tools used by the test and load-test commands when needed:

```bash
python -m pip install pytest black flake8 locust
```

## Model files

The application loads these files by default:

```text
models/cnn_asl_model_full.pth
models/label_encoder_classes.npy
```

The checkpoint contains the CNN weights and model metadata. The NumPy file
contains the class labels in the same index order used during training. Keep
the checkpoint and label encoder from the same training run; mixing them can
produce incorrect predictions or a model-loading error.

## Running the application

### FastAPI server

```bash
python app.py
```

The server listens on `http://localhost:5000`.

- API root: `GET http://localhost:5000/`
- Health check: `GET http://localhost:5000/health`
- Swagger UI: `http://localhost:5000/docs`
- Browser interface: `http://localhost:5000/ui`

### Browser interface

Open `http://localhost:5000/ui` after starting the server and allow camera
access. The interface captures frames, sends them to the API, and displays the
current text, predicted sign, confidence, FPS, stability, and action feedback.
Use the reset control to clear the accumulated text.

### OpenCV webcam client

In another terminal:

```bash
python client.py
```

The client uses `http://localhost:5000` by default. To connect to another
machine:

```bash
SLR_API_URL=http://192.168.1.20:5000 python client.py
```

Keyboard controls:

- `q`: quit.
- `r`: reset the text and temporal state.

## API reference

### `GET /`

Returns:

```json
{"message": "Sign Language Recognition API is running"}
```

### `GET /health`

Returns `{"status": "ok"}` when the process is running.

### `POST /ws`

Request body:

```json
{"image": "<base64-encoded JPEG or PNG>"}
```

Successful responses contain:

```json
{
  "status": "success",
  "is_stable": true,
  "current_text": "Hello",
  "action_feedback": "Added: A",
  "fps": 24.5,
  "predicted_letter": "A",
  "confidence": 0.91
}
```

`confidence` is the model softmax confidence for the current prediction, not
an independently measured accuracy score. Invalid base64 or undecodable image
data returns HTTP `400`; unexpected server failures return HTTP `500`.

### `POST /reset`

Clears the accumulated text, prediction buffer, landmark history, and timing
state:

```json
{"status": "success", "message": "Processor reset"}
```

## Training

### Dataset

The original dataset was manually collected for this project and is not
included because of data-privacy considerations. See
[`data/data.md`](data/data.md) for the dataset description and reference
dataset.

The training CSV must be saved as `data/landmarks.csv` and contain:

- 63 landmark columns: 21 landmarks × 3 coordinates;
- one `label` column containing the sign class.

### Training pipeline

The training script:

1. reads `data/landmarks.csv`;
2. normalizes every sample relative to the wrist;
3. creates an augmented subset equal to 20% of the original samples using
   noise, scaling, and small in-plane rotation;
4. encodes labels with `LabelEncoder`;
5. splits the resulting data into 80% training and 20% validation sets using a
   stratified split with `random_state=42`;
6. trains a CNN1D for 4 epochs with Adam (`learning_rate=0.001`, batch size
   `32`);
7. saves the checkpoint and label encoder under `models/`.

Run it with:

```bash
python -m pip install pandas scikit-learn matplotlib seaborn
python training/train_cnn1d.py
```

The script prints `Train Acc` and `Val Acc` after every epoch. The repository
does not currently store a reproducible metrics report, dataset row count, or
an independent held-out test score, so no single final accuracy percentage is
claimed here. To report a new accuracy, save the full training output together
with the dataset version, hardware, and checkpoint used.

## What was tested

The test suite covers:

- `SignProcessor` model and label initialization;
- landmark normalization and wrist-origin behavior;
- stable versus unstable landmark histories;
- letter insertion and `del` behavior;
- frame processing output shape and default state;
- processor reset;
- `GET /`;
- valid `POST /ws` image handling;
- invalid base64 rejection with HTTP `400`;
- `POST /reset`.

Run the unit tests:

```bash
pytest -q
```

For formatting and lint checks:

```bash
black --check .
flake8 slr scripts training tests
```

### Load testing

The Locust scenario in [`tests/locustfile.py`](tests/locustfile.py) exercises
`POST /ws` four times as often as `POST /reset`, using a generated dummy PNG.
It defines light, medium, and heavy user profiles.

Start the API first, then run:

```bash
locust -f tests/locustfile.py --host http://localhost:5000
```

The scenario prints total requests, failures, average response time, and
maximum response time when the run finishes. These values depend on the
machine, model, number of users, spawn rate, and test duration; the checked-in
screenshots are evidence of an earlier run, not a universal performance
guarantee.

## Project structure

```text
.
├── app.py                         # Backward-compatible API entry point
├── client.py                      # Backward-compatible webcam entry point
├── slr/
│   ├── api.py                     # FastAPI app and routes
│   ├── config.py                  # Repository paths and defaults
│   ├── sign_processor.py          # MediaPipe, CNN, smoothing, and text state
│   └── static/                    # Browser UI, CSS, JavaScript, and logo
├── scripts/
│   └── client.py                  # OpenCV webcam client implementation
├── training/
│   └── train_cnn1d.py             # Data preparation and training pipeline
├── models/
│   ├── cnn_asl_model_full.pth     # CNN checkpoint
│   └── label_encoder_classes.npy  # Label-to-index mapping
├── data/
│   └── data.md                    # Dataset documentation
├── tests/
│   ├── test_sign_language_recognition.py
│   └── locustfile.py
├── requirements.txt
└── pyproject.toml
```

## Troubleshooting

- **Model loading fails:** verify that both files under `models/` exist and
  were produced for the same architecture and label mapping.
- **Predictions are unstable:** use good lighting, keep one hand visible, hold
  the sign steady, and ensure the hand is inside the camera frame.
- **Camera cannot open:** close applications using the camera and check
  operating-system camera permissions.
- **Slow inference:** use a CUDA-enabled PyTorch build, lower the camera
  resolution, or run the API and client on the same machine.
- **Client cannot connect:** start the API first and verify
  `GET /health`; when using another host, set `SLR_API_URL`.
- **Low confidence:** the runtime ignores predictions below the configured
  `0.75` threshold. Confidence is affected by lighting, hand pose, camera
  angle, and similarity between classes.

## Limitations and future improvements

- The current model handles isolated static signs rather than dynamic
  multi-frame signs or full sentence grammar.
- Accuracy is not yet reported with a saved, reproducible evaluation artifact.
- The dataset is private, which limits external reproduction and comparison.
- A production deployment should restrict CORS origins, add authentication and
  request limits, and use a production process manager.
- Future evaluation should include a subject-independent test split, a
  confusion matrix, per-class precision/recall/F1, and latency measurements
  under a documented hardware setup.

## License and data notice

Check the repository license before redistributing the project. The privately
collected training data is not included and must not be redistributed without
the data owners' permission.
