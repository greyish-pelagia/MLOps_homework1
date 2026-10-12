import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.inference import Sentiment, SentimentPredictor

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[
        str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)
    ]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction: Sentiment


def create_app(
    load_predictor: Callable[[], SentimentPredictor] = SentimentPredictor.load,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:

        application.state.predictor = load_predictor()
        yield
        del application.state.predictor

    application = FastAPI(title="Sentiment Analysis", lifespan=lifespan)

    @application.post("/predict", response_model=PredictionResponse)
    def predict(body: PredictionRequest, request: Request) -> PredictionResponse:

        try:
            sentiment = request.app.state.predictor.predict(body.text)
            return PredictionResponse(prediction=sentiment)
        except Exception as error:
            logger.exception("Sentiment prediction failed")
            raise HTTPException(status_code=500, detail="Prediction failed.") from error

    return application


app = create_app()
