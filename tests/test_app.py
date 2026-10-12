from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import PredictionResponse, create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    predictor = Mock()
    predictor.predict.return_value = "positive"
    with TestClient(create_app(lambda: predictor)) as test_client:
        yield test_client


def test_prediction_is_valid_json(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "A great lecture!"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"prediction": "positive"}
    PredictionResponse.model_validate(response.json())


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"text": ""},
        {"text": " \n\t "},
        {"text": 123},
        {"text": None},
        {"text": []},
        {"text": "Hello", "extra": True},
    ],
)
def test_invalid_input_returns_json_error(client: TestClient, body: dict) -> None:
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["msg"]
    assert error["loc"][0] == "body"


def test_malformed_json(client: TestClient) -> None:
    response = client.post(
        "/predict", content="{", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"]


@pytest.mark.parametrize("label", ["negative", "neutral", "positive"])
def test_response_accepts_valid_sentiments(label: str) -> None:
    assert PredictionResponse.model_validate({"prediction": label}).prediction == label


@pytest.mark.parametrize(
    "body",
    [
        {"prediction": "wrong"},
        {"prediction": 2},
        {},
        {"prediction": "positive", "extra": 1},
    ],
)
def test_response_rejects_invalid_data(body: dict) -> None:
    with pytest.raises(ValidationError):
        PredictionResponse.model_validate(body)


def test_model_is_loaded_once() -> None:
    predictor = Mock()
    predictor.predict.return_value = "neutral"
    loader = Mock(return_value=predictor)
    with TestClient(create_app(loader)) as client:
        assert client.post("/predict", json={"text": "First"}).status_code == 200
        assert client.post("/predict", json={"text": "Second"}).status_code == 200
    loader.assert_called_once()


def test_prediction_errors_return_json() -> None:
    predictor = Mock()
    predictor.predict.side_effect = RuntimeError("private model error")
    with TestClient(create_app(lambda: predictor)) as client:
        response = client.post("/predict", json={"text": "Hello"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed."}
