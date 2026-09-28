from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re

import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


EVENT_PATTERNS = {
    "product_launch": [r"\blaunch(?:es|ed|ing)?\b", r"\bunveil(?:s|ed|ing)?\b", r"\bnew product\b", r"\brelease(?:s|d)?\b"],
    "software_update": [r"\bsoftware update\b", r"\bupdate(?:s|d)?\b", r"\bversion \d", r"\bupgrade(?:s|d)?\b"],
    "earnings": [r"\bearnings\b", r"\brevenue\b", r"\bquarter(?:ly)? results\b", r"\bguidance\b"],
    "partnership": [r"\bpartnership\b", r"\bpartner(?:s|ed)? with\b", r"\bcollaboration\b"],
    "acquisition": [r"\bacquir(?:e|es|ed|ing)\b", r"\bacquisition\b", r"\bmerger\b"],
    "regulatory": [r"\bregulat(?:or|ory|ion)\b", r"\bantitrust\b", r"\bapproval\b", r"\bfine\b"],
    "recall_or_delay": [r"\brecall(?:s|ed)?\b", r"\bdelay(?:s|ed)?\b", r"\bpostpone(?:s|d)?\b", r"\bdefect\b"],
    "management": [r"\bCEO\b", r"\bCFO\b", r"\bchief executive\b", r"\bresign(?:s|ed)?\b", r"\bappoint(?:s|ed)?\b"],
}

NEWS_FEATURE_COLUMNS = [
    "news_sentiment_24h",
    "news_sentiment_7d",
    "news_count_24h",
    "news_count_7d",
    "news_sentiment_change",
    *[f"event_{name}_7d" for name in EVENT_PATTERNS],
]

_analyzer = SentimentIntensityAnalyzer()


def _published_at(item: dict) -> datetime | None:
    ts = item.get("providerPublishTime")
    if ts:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    content = item.get("content", {}) or {}
    date = content.get("pubDate") or content.get("displayTime")
    if date:
        try:
            return pd.to_datetime(date, utc=True).to_pydatetime()
        except Exception:
            return None
    return None


def _headline(item: dict) -> str:
    content = item.get("content", {}) or {}
    return str(item.get("title") or content.get("title") or "")


def _summary(item: dict) -> str:
    content = item.get("content", {}) or {}
    return str(item.get("summary") or content.get("summary") or "")


def classify_events(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        name: int(any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns))
        for name, patterns in EVENT_PATTERNS.items()
    }


def fetch_current_news(symbol: str) -> pd.DataFrame:
    items = yf.Ticker(symbol).news or []
    rows = []
    for item in items:
        published = _published_at(item)
        if not published:
            continue
        text = f"{_headline(item)}. {_summary(item)}".strip()
        if not text:
            continue
        row = {
            "published_at": published,
            "text": text,
            "sentiment": _analyzer.polarity_scores(text)["compound"],
        }
        row.update(classify_events(text))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["published_at", "text", "sentiment", *EVENT_PATTERNS])
    return pd.DataFrame(rows).sort_values("published_at")


def aggregate_news(news: pd.DataFrame, as_of=None) -> dict[str, float]:
    result = {name: 0.0 for name in NEWS_FEATURE_COLUMNS}
    if news.empty:
        return result

    as_of = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")

    frame = news.copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    frame = frame[frame["published_at"] <= as_of]

    last_24h = frame[frame["published_at"] > as_of - pd.Timedelta(hours=24)]
    last_7d = frame[frame["published_at"] > as_of - pd.Timedelta(days=7)]

    result["news_count_24h"] = float(len(last_24h))
    result["news_count_7d"] = float(len(last_7d))
    result["news_sentiment_24h"] = float(last_24h["sentiment"].mean()) if len(last_24h) else 0.0
    result["news_sentiment_7d"] = float(last_7d["sentiment"].mean()) if len(last_7d) else 0.0
    result["news_sentiment_change"] = result["news_sentiment_24h"] - result["news_sentiment_7d"]

    for event in EVENT_PATTERNS:
        result[f"event_{event}_7d"] = float(last_7d[event].sum()) if event in last_7d else 0.0
    return result


def load_historical_news(path: str) -> pd.DataFrame:
    """CSV columns: symbol,published_at,text. Sentiment/events are derived here."""
    frame = pd.read_csv(path)
    required = {"symbol", "published_at", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Historical news CSV is missing columns: {sorted(missing)}")
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    frame["sentiment"] = frame["text"].fillna("").map(lambda x: _analyzer.polarity_scores(str(x))["compound"])
    event_rows = frame["text"].fillna("").map(lambda x: classify_events(str(x)))
    for event in EVENT_PATTERNS:
        frame[event] = event_rows.map(lambda d: d[event])
    return frame
