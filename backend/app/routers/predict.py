"""The two addresses the app uses: GET /health and POST /predict."""
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.services import flow  # also puts app/services on the import path for the next line
from features import decode_image

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MAX_BYTES = 10 * 1024 * 1024  # 10 MB
SIGNATURES = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")  # first bytes of every JPG and PNG file

router = APIRouter()
_state = {}


def load_models():
    """Load the four models (about 100 MB) once, when the server starts."""
    _state["models"] = flow.load_models(MODELS_DIR)
    version_file = MODELS_DIR / "model_version.txt"
    _state["version"] = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else "unknown"


class PredictResult(BaseModel):
    """What the app shows; see ml/src/flow.py. Save it to Supabase only when outcome is
    healthy, severity_only or pest (database/README.md)."""
    outcome: Literal["healthy", "severity_only", "pest", "not_corn", "unclear"]
    pest: Optional[str] = None                  # fall_armyworm, common_cutworm, armyworm, corn_planthopper, corn_aphid
    pest_confidence: Optional[float] = None
    severity: Optional[Literal["Healthy", "Low", "Moderate", "High", "Unclear"]] = None
    damage_percent: Optional[float] = None
    insect_visible: Optional[bool] = None
    insect_confidence: Optional[float] = None
    healthy: Optional[bool] = None
    health_confidence: Optional[float] = None
    corn_confidence: float
    model_version: str


@router.get("/health")
def health():
    """Is the server running, and which models are loaded?"""
    return {"status": "ok", "models_loaded": sorted(_state.get("models", {})),
            "model_version": _state.get("version")}


@router.post("/predict", response_model=PredictResult)
async def predict(photo: UploadFile = File(..., description="JPG or PNG photo of a corn plant, at most 10 MB")):
    """Run the app's full decision on one photo: corn check -> insect detector ->
    healthy-vs-damaged -> severity -> pest classifier."""
    data = await photo.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "The photo is empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The photo is larger than 10 MB.")
    if not data.startswith(SIGNATURES):
        raise HTTPException(415, "Only JPG and PNG photos are accepted.")
    try:
        rgb = decode_image(data)
    except ValueError:
        raise HTTPException(400, "The photo could not be read. It may be damaged.")
    result = await run_in_threadpool(flow.assess_photo, rgb, _state["models"])
    return {**result, "model_version": _state["version"]}
