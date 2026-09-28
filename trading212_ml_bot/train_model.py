from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from config import MODEL_PATH, WATCHLIST
from features import FEATURE_COLUMNS, build_features
from market_data import download_history


def training_frame(symbol: str) -> pd.DataFrame:
    raw = download_history(symbol, period="5y")
    data = build_features(raw)

    # Predict whether the NEXT daily close is higher than today's close.
    data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    # The final row has no known future outcome and must not be training data.
    data = data.iloc[:-1]
    return data.dropna(subset=FEATURE_COLUMNS + ["target"])


def main():
    frames = [training_frame(symbol) for _, symbol in WATCHLIST]
    dataset = pd.concat(frames).sort_index()

    # Time-aware split: never randomly mix future observations into training.
    split = int(len(dataset) * 0.80)
    train = dataset.iloc[:split]
    test = dataset.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(train[FEATURE_COLUMNS], train["target"])

    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(classification_report(test["target"], predictions, digits=3))
    print(f"Out-of-sample ROC AUC: {roc_auc_score(test['target'], probabilities):.3f}")

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
