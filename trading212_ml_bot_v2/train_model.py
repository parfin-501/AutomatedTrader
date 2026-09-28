from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from config import HISTORICAL_NEWS_CSV, MODEL_PATH, TARGET_DAYS, TARGET_RETURN, WATCHLIST
from features import FEATURE_COLUMNS, TECHNICAL_FEATURE_COLUMNS, build_features
from market_data import download_history
from news import NEWS_FEATURE_COLUMNS, aggregate_news, load_historical_news


def training_frame(symbol: str, historical_news: pd.DataFrame) -> pd.DataFrame:
    raw = download_history(symbol, period="5y")
    data = build_features(raw)
    future_return = data["Close"].shift(-TARGET_DAYS) / data["Close"] - 1
    data["target"] = (future_return > TARGET_RETURN).astype(int)

    symbol_news = historical_news[historical_news["symbol"] == symbol]
    for idx in data.index:
        as_of = pd.Timestamp(idx)
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        news_features = aggregate_news(symbol_news, as_of=as_of)
        for column, value in news_features.items():
            data.loc[idx, column] = value

    data = data.iloc[:-TARGET_DAYS]
    return data.dropna(subset=FEATURE_COLUMNS + ["target"])


def main():
    if not Path(HISTORICAL_NEWS_CSV).exists():
        raise FileNotFoundError(
            f"{HISTORICAL_NEWS_CSV} is required for leakage-safe V2 training. "
            "See data/historical_news.example.csv."
        )

    historical_news = load_historical_news(HISTORICAL_NEWS_CSV)
    frames = [training_frame(symbol, historical_news) for _, symbol in WATCHLIST]
    dataset = pd.concat(frames).sort_index()

    split = int(len(dataset) * 0.80)
    train, test = dataset.iloc[:split], dataset.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])

    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    print(classification_report(test["target"], predictions, digits=3))
    print(f"Out-of-sample ROC AUC: {roc_auc_score(test['target'], probabilities):.3f}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("\nFeature importance:\n", importances.to_string())

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
