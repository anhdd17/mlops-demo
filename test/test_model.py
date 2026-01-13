import os
import joblib


def test_model_artifact_exists():
    assert os.path.exists("artifacts/model.joblib")


def test_model_can_be_loaded():
    model = joblib.load("artifacts/model.joblib")
    assert model is not None
