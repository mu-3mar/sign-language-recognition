# Sign Language Recognition

Real-time American Sign Language (ASL) recognition from a webcam. The project
extracts 21 hand landmarks with MediaPipe, classifies the normalized landmarks
with a PyTorch 1D CNN, and exposes the result through a FastAPI service.

## What this project provides

- FastAPI inference API for base64-encoded camera frames.
- OpenCV webcam client with live text, confidence, FPS, and stability feedback.
- Temporal smoothing and simple `space`/`del` actions.
- Training script for a landmark-based CNN.
- Unit tests and an optional Locust load-test scenario.

## Requirements

- Python 3.10, 3.11, or 3.12.
- A webcam for the local client.
- The model checkpoint and label encoder in `models/`.
- CPU works by default. For NVIDIA GPU acceleration, install the matching
  PyTorch build from the official PyTorch selector instead of the CPU wheels.

## Installation

```bash
git clone https://github.com/mu-3mar/sign-language-recognition.git
cd sign-language-recognition

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Install optional dependency groups only when needed:

```bash
pip install -r requirements-training.txt
pip install -r requirements-load-testing.txt
```

## Model files

Place these files in `models/`:

```text
models/cnn_asl_model_full.pth
models/label_encoder_classes.npy
```

They are runtime artifacts rather than source code. Do not commit private,
unverified, or oversized checkpoints to a public repository.

## Run the application

Start the API:

```bash
python app.py
```

The service listens on `http://localhost:5000`. Interactive API documentation
is available at `http://localhost:5000/docs`, and the health check is
`GET /health`.

Open the browser interface at `http://localhost:5000/ui`. It uses the supplied
animated logo, requests webcam permission, sends captured frames to `POST /ws`,
and clears the live message through `POST /reset`. The JSON response at `/`
is intentionally preserved for existing API clients.

In another terminal, start the webcam client:

```bash
python client.py
```

The client sends frames to `http://localhost:5000` by default. To use another
server, set `SLR_API_URL` (or copy `.env.example` to `.env` and export it):

```bash
SLR_API_URL=http://192.168.1.20:5000 python client.py
```

Keyboard controls in the client:

- `q`: quit.
- `r`: reset the accumulated text.

## API

### `GET /`

Returns a service status message.

### `GET /health`

Returns `{"status": "ok"}` when the API process is running.

### `POST /ws`

Request body:

```json
{"image": "<base64-encoded JPEG or PNG>"}
```

The response contains the current text, predicted sign, confidence, FPS, and
stability state. Invalid base64 or image data returns HTTP 400.

### `POST /reset`

Clears the accumulated text and temporal prediction state.

## Training

Put a CSV containing 63 landmark columns plus a `label` column at
`data/landmarks.csv`, then install the training dependencies:

```bash
pip install -r requirements-training.txt
python training/train_cnn1d.py
```

The trained checkpoint and label encoder are written to `models/`.

## Testing and quality checks

```bash
pytest -q
black --check .
flake8 slr scripts training tests
```

For load testing, install `requirements-load-testing.txt` and run:

```bash
locust -f tests/locustfile.py --host http://localhost:5000
```

The Locust scenario targets `/ws` and `/reset`. Set the host explicitly; do
not use a hard-coded public IP.

## Project layout

```text
.
├── app.py                         # Backward-compatible API entry point
├── client.py                      # Backward-compatible webcam entry point
├── slr/
│   ├── api.py                     # FastAPI app and routes
│   ├── config.py                  # Repository paths and defaults
│   ├── sign_processor.py          # MediaPipe, CNN, and text state
│   └── static/                    # Browser UI and supplied animated logo
├── scripts/
│   └── client.py                  # Webcam client implementation
├── training/
│   └── train_cnn1d.py             # Model training pipeline
├── models/                        # Checkpoints and label encoder
├── data/                          # Local training data (ignored)
├── tests/
│   ├── test_sign_language_recognition.py
│   └── locustfile.py
├── requirements*.txt              # Runtime and optional dependency sets
├── pyproject.toml                 # Python project and tool configuration
└── .env.example                   # Local environment template
```

## Troubleshooting

- **Model loading fails:** verify both files exist under `models/` and that the
  checkpoint architecture matches `CNN1D`.
- **Camera cannot open:** close other camera applications and check OS camera
  permissions.
- **Slow inference:** run on a GPU-enabled PyTorch installation or reduce the
  webcam resolution before sending frames.
- **Client cannot connect:** start the API first and verify `GET /health`.
