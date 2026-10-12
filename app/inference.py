from pathlib import Path
from typing import Any, Literal

import joblib
from sentence_transformers import SentenceTransformer

Sentiment = Literal["negative", "neutral", "positive"]
LABELS: dict[int, Sentiment] = {0: "negative", 1: "neutral", 2: "positive"}
MODEL_DIRECTORY = Path(__file__).resolve().parent.parent / "models"


def load_classifier(path: Path) -> Any:
    """Load a trusted classifier using the serializer that created it."""
    if not path.is_file():
        raise FileNotFoundError(f"Classifier file not found: {path}")

    return joblib.load(path)


class SentimentPredictor:
    def __init__(self, encoder: Any, classifier: Any) -> None:
        self.encoder = encoder
        self.classifier = classifier

    @classmethod
    def load(cls) -> "SentimentPredictor":

        encoder_path = MODEL_DIRECTORY / "sentence_transformer.model"
        classifier_path = MODEL_DIRECTORY / "classifier.joblib"

        if not encoder_path.is_dir():
            raise FileNotFoundError(
                f"Sentence Transformer directory not found: {encoder_path}"
            )
        classifier = load_classifier(classifier_path)

        encoder = SentenceTransformer(
            str(encoder_path), device="mps", local_files_only=True
        )
        return cls(encoder, classifier)

    def predict(self, text: str) -> Sentiment:
        """Predict the sentiment of a given text."""
        embeddings = self.encoder.encode(
            [text], convert_to_numpy=True, show_progress_bar=False
        )
        class_id = int(self.classifier.predict(embeddings)[0])
        if class_id not in LABELS:
            raise ValueError(f"Unknown sentiment class: {class_id}")
        return LABELS[class_id]
