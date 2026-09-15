# Sign Language Recognition (FastAPI + MediaPipe + PyTorch)

Real-time sign language recognition via webcam, exposed as a FastAPI service and a simple camera client.

## Features

- Real-time hand landmark detection (MediaPipe) + classification (PyTorch)
- FastAPI endpoint that accepts a base64-encoded frame
- Client script that streams webcam frames to the API
- Basic text accumulation + delete/space handling + spell correction

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

- Start the API:

```bash
python app.py
```

- Start the webcam client:

```bash
python client.py
```

## Models (not committed)

Model artifacts are **ignored by git**. Put these files in `models/`:

- `models/cnn_asl_model_full.pth`
- `models/label_encoder_classes.npy`

## Project structure

```text
.
├── app.py                      # Entry-point wrapper (runs FastAPI)
├── client.py                   # Entry-point wrapper (runs webcam client)
├── slr/                        # Core package
│   ├── __init__.py
│   ├── api.py                  # FastAPI app + routes
│   └── sign_processor.py       # SignProcessor + CNN1D
├── scripts/
│   └── client.py               # Webcam client implementation
├── models/                     # Model artifacts (gitignored, kept via .gitkeep)
├── training/
│   └── train_cnn1d.py           # Training script
├── tests/
│   ├── locustfile.py
│   ├── test_sign_language_recognition.py
│   └── artifacts/              # Test output images (gitignored, kept via .gitkeep)
├── requirements.txt
└── README.md
```

