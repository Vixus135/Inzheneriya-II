import os
import sys
from io import BytesIO

# Добавляем корень проекта в PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Указываем путь к модели ДО импорта app
os.environ["MODEL_PATH"] = os.path.join(project_root, "artifacts", "model_best.pth")

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.service.api import app


client = TestClient(app)


def _make_test_image(size=(150, 150), color="red"):
    img = Image.new("RGB", size, color=color)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_predict_valid_image():
    img = _make_test_image(size=(150, 150), color="blue")
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    
    # Модель должна быть загружена
    assert response.status_code == 200
    data = response.json()
    
    assert "predicted_class" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1
    
    expected_classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    for cls in expected_classes:
        assert cls in data["probabilities"]
        assert isinstance(data["probabilities"][cls], float)
    
    total = sum(data["probabilities"].values())
    assert 0.99 <= total <= 1.01


def test_predict_wrong_file_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", BytesIO(b"not an image"), "text/plain")}
    )
    
    assert response.status_code == 400
    assert "must be an image" in response.json()["detail"]


def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422


def test_batch_predict_two_images():
    img1 = _make_test_image(size=(150, 150), color="green")
    img2 = _make_test_image(size=(150, 150), color="yellow")
    
    response = client.post(
        "/predict_batch",
        files=[
            ("files", ("img1.jpg", img1, "image/jpeg")),
            ("files", ("img2.jpg", img2, "image/jpeg"))
        ]
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    
    for pred in data["predictions"]:
        assert "predicted_class" in pred
        assert "confidence" in pred
        assert "probabilities" in pred


def test_batch_predict_too_many():
    files = []
    for i in range(12):
        img = _make_test_image(size=(64, 64), color="red")
        files.append(("files", (f"img{i}.jpg", img, "image/jpeg")))
    
    response = client.post("/predict_batch", files=files)
    assert response.status_code == 400
    assert "Maximum 10" in response.json()["detail"]


def test_predict_response_types():
    img = _make_test_image(size=(150, 150), color="purple")
    
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data["predicted_class"], str)
    assert isinstance(data["confidence"], (int, float))
    assert isinstance(data["probabilities"], dict)
    
    valid_classes = {"buildings", "forest", "glacier", "mountain", "sea", "street"}
    assert data["predicted_class"] in valid_classes