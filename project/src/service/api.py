import os
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from src.inference.predictor import LandscapePredictor


app = FastAPI(
    title="Landscape Classification API",
    description="Сервис классификации ландшафтов по изображениям",
    version="1.0.0"
)

# Загрузка модели ПРИ ИМПОРТЕ модуля (не через startup event)
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model_best.pth")
predictor = None

if os.path.exists(MODEL_PATH):
    try:
        predictor = LandscapePredictor(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
else:
    print(f"⚠️ Model not found at {MODEL_PATH}")


@app.get("/health")
async def health_check():
    """Health-check эндпоинт."""
    return {
        "status": "ok",
        "model_loaded": predictor is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Классифицировать загруженное изображение.
    
    Возвращает предсказанный класс и вероятности по всем классам.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Проверка формата
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        result = predictor.predict(image)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Батчевая классификация (до 10 изображений)."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    try:
        images = []
        for file in files:
            contents = await file.read()
            image = Image.open(BytesIO(contents)).convert("RGB")
            images.append(image)
        
        results = predictor.predict_batch(images)
        return JSONResponse(content={"predictions": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)