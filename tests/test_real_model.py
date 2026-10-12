import pytest
from fastapi.testclient import TestClient

from app.inference import SentimentPredictor
from app.main import PredictionResponse, create_app

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def predictor() -> SentimentPredictor:

    return SentimentPredictor.load()


def test_provided_model_loads(predictor: SentimentPredictor) -> None:
    assert predictor.encoder is not None
    assert predictor.classifier is not None


@pytest.mark.parametrize(
    "text",
    [
        "I love this lecture!",
        "The lecture was terrible.",
        "The lecture starts at nine.",
    ],
)
def test_real_inference(predictor: SentimentPredictor, text: str) -> None:
    assert predictor.predict(text) in {"negative", "neutral", "positive"}


def test_real_api(predictor: SentimentPredictor) -> None:
    with TestClient(create_app(lambda: predictor)) as client:
        response = client.post("/predict", json={"text": "A great lecture!"})
    assert response.status_code == 200
    PredictionResponse.model_validate(response.json())
