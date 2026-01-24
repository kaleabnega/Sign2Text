from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ----------------------------
# ----- Config Constants -----
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
LABEL_MAP_PATH = CACHE_DIR / "label_map.json"
MODEL_WEIGHTS_PATH = CACHE_DIR / "arsl_tcn.pt"
SEQUENCE_LENGTH = 32  # Must match training configuration
NUM_HANDS = 2
NUM_KEYPOINTS = 21
LANDMARK_DIMS = 3
FEATURE_DIM = NUM_HANDS * NUM_KEYPOINTS * LANDMARK_DIMS  # 126
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequencePayload(BaseModel):
    """Payload representing a sequence of flattened landmark frames."""

    sequence: List[List[float]] = Field(
        ..., description="List of frames; each frame is a flattened landmark vector (len=126)."
    )


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    scores: List[float]


def load_label_map() -> Tuple[dict[str, int], dict[int, str]]:
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"Label map not found at {LABEL_MAP_PATH}. "
            "Run generate_all_caches() before starting the API."
        )
    label_map = json.loads(LABEL_MAP_PATH.read_text())
    reverse_map = {idx: label for label, idx in label_map.items()}
    return label_map, reverse_map


class TemporalConvNet(torch.nn.Module):
    """Same architecture used in the training notebook (2 temporal blocks)."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        hidden_channels = (128, 256)
        kernel_size = 3
        dropout = 0.3
        layers: List[torch.nn.Module] = []
        in_channels = feature_dim
        for out_channels in hidden_channels:
            layers.append(
                torch.nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(torch.nn.BatchNorm1d(out_channels))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
            in_channels = out_channels
        self.encoder = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, features, time)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=-1)
        logits = self.classifier(pooled)
        return logits


def load_model(reverse_label_map: dict[int, str]) -> TemporalConvNet:
    num_classes = len(reverse_label_map)
    model = TemporalConvNet(FEATURE_DIM, num_classes).to(DEVICE)
    if not MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_WEIGHTS_PATH}. "
            "Train the notebook and save weights as arsl_tcn.pt."
        )
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def validate_sequence(sequence: List[List[float]]) -> np.ndarray:
    if len(sequence) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Sequence must contain {SEQUENCE_LENGTH} frames; received {len(sequence)}.",
        )
    array = np.array(sequence, dtype=np.float32)
    if array.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
        raise HTTPException(
            status_code=422,
            detail=f"Each frame must be length {FEATURE_DIM}; received shape {array.shape}.",
        )
    return array


app = FastAPI(
    title="Sign2Text ArSL API",
    description="Accepts landmark sequences and returns predicted Arabic sign words.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABEL_MAP, REVERSE_LABEL_MAP = load_label_map()
MODEL = load_model(REVERSE_LABEL_MAP)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SequencePayload) -> PredictionResponse:
    sequence = validate_sequence(payload.sequence)
    tensor = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_label = REVERSE_LABEL_MAP.get(pred_idx, "UNKNOWN")
    confidence = float(probs[pred_idx])
    return PredictionResponse(
        prediction=pred_label,
        confidence=confidence,
        scores=probs.tolist(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "arsl-word-level-detection.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
