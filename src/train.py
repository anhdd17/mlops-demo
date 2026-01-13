from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import joblib
import os


def train():
    print("Loading dataset...")
    X, y = load_diabetes(return_X_y=True, as_frame=True)

    print("Training model...")
    model = LinearRegression()
    model.fit(X, y)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")

    print("Model saved to artifacts/model.joblib")


if __name__ == "__main__":
    train()
