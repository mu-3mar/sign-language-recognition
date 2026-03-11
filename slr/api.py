# FastAPI app for sign language recognition
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from pydantic import BaseModel
from slr.sign_processor import SignProcessor
import binascii

# Initialize FastAPI app
app = FastAPI(title="Sign Language Recognition API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sign language processor
processor = SignProcessor()

class ImageData(BaseModel):
    image: str

@app.get("/")
async def root():
    return {"message": "Sign Language Recognition API is running"}

@app.post("/ws")
async def process_image(data: ImageData):
    try:
        # Convert base64 to image
        image_data = base64.b64decode(data.image, validate=True)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Process frame and return results
        response = processor.process_frame(frame)
        return response

    except binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/reset")
async def reset_processor():
    try:
        response = processor.reset()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)