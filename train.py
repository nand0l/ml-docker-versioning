import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    load_dotenv()

    test_size = float(os.getenv("TEST_SIZE", "0.2"))
    random_state = int(os.getenv("RANDOM_STATE", "42"))
    max_iter = int(os.getenv("MODEL_MAX_ITER", os.getenv("MAX_ITER", "300")))
    model_c = float(os.getenv("MODEL_C", "1.0"))

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=random_state,
        stratify=data.target,
    )

    model = LogisticRegression(max_iter=max_iter, C=model_c)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    output_path = Path("model") / "model.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
