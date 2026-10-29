"""FastAPI application for sign language recognition."""

import base64
import binascii

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from slr.config import PROJECT_ROOT
from slr.sign_processor import SignProcessor

app = FastAPI(
    title="Sign Language Recognition API",
    description="Real-time ASL recognition from hand landmarks.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/ui/assets", StaticFiles(directory=PROJECT_ROOT / "slr" / "static"), name="ui-assets")

processor = SignProcessor()


class ImageData(BaseModel):
    """Request body for an encoded camera frame."""

    image: str


@app.get("/")
async def root():
    return {"message": "Sign Language Recognition API is running"}


@app.get("/ui", include_in_schema=False)
async def ui():
    return FileResponse(PROJECT_ROOT / "slr" / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ws")
async def process_image(data: ImageData):
    try:
        image_data = base64.b64decode(data.image, validate=True)
        frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        return processor.process_frame(frame)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Server error: {exc}") from exc


@app.post("/reset")
async def reset_processor():
    try:
        return processor.reset()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset error: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
