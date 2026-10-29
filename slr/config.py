"""Application configuration and repository paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "cnn_asl_model_full.pth"
DEFAULT_LABELS_PATH = MODELS_DIR / "label_encoder_classes.npy"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000

