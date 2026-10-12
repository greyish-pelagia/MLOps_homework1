import logging
from typing import Annotated

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.inference import Sentiment

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[
        str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)
    ]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction: Sentiment


app = FastAPI(title="Sentiment Analysis")


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest, request: Request) -> PredictionResponse:

    sentiment: Sentiment = "positive"
    return PredictionResponse(prediction=sentiment)
