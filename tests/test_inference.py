from pathlib import Path
from unittest.mock import Mock

import joblib
import numpy as np
import pytest

from app.inference import SentimentPredictor, load_classifier


@pytest.mark.parametrize(
    "class_id, expected", [(0, "negative"), (1, "neutral"), (2, "positive")]
)
def test_embeddings_are_used_for_prediction(class_id: int, expected: str) -> None:
    embedding = np.array([[0.1, 0.2, 0.3]])
    encoder = Mock()
    encoder.encode.return_value = embedding
    classifier = Mock()
    classifier.predict.return_value = np.array([class_id])
    predictor = SentimentPredictor(encoder, classifier)
    assert predictor.predict("A sample sentence.") == expected
    encoder.encode.assert_called_once_with(
        ["A sample sentence."], convert_to_numpy=True, show_progress_bar=False
    )
    assert np.array_equal(classifier.predict.call_args.args[0], embedding)


def test_unknown_class_is_rejected() -> None:
    encoder = Mock()
    classifier = Mock()
    classifier.predict.return_value = [99]
    with pytest.raises(ValueError, match="Unknown sentiment class"):
        SentimentPredictor(encoder, classifier).predict("Hello")


def test_classifier_deserialization(tmp_path: Path) -> None:

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression().fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    path = tmp_path / "classifier.joblib"
    joblib.dump(classifier, path)

    restored = load_classifier(path)
    assert restored.predict([[0], [3]]).tolist() == [0, 1]


def test_missing_classifier_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Classifier file not found"):
        load_classifier(tmp_path / "missing.joblib")
