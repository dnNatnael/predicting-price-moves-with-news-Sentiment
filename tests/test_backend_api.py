"""
Unit and integration tests for FastAPI backend routes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "AAPL" in data["available_tickers"]


def test_overview_endpoint(client):
    response = client.get("/api/overview?ticker=AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["total_news_articles"] > 0
    assert data["total_trading_days"] > 0
    assert "positive" in data["sentiment_distribution"]


def test_sentiment_benchmark_endpoint(client):
    response = client.get("/api/sentiment-benchmark?ticker=AAPL")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 4
    approaches = [d["approach"] for d in data]
    assert "Ensemble Composite" in approaches
    assert "Financial VADER" in approaches


def test_news_feed_endpoint(client):
    response = client.get("/api/news-feed?ticker=AAPL&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "headline" in data[0]
        assert "fin_vader" in data[0]
        assert "ensemble" in data[0]


def test_prices_endpoint(client):
    response = client.get("/api/prices?ticker=AAPL&limit=30")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 30
    assert "close" in data[0]
    assert "open" in data[0]
    assert "volume" in data[0]


def test_correlations_endpoint(client):
    response = client.get("/api/correlations?ticker=AAPL")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_models_endpoint(client):
    response = client.get("/api/models?ticker=AAPL")
    assert response.status_code == 200
    data = response.json()
    assert "ticker" in data
    assert "binary_models" in data
    assert "multiclass_models" in data
    assert "top_features" in data


def test_validation_endpoint(client):
    response = client.get("/api/validation?ticker=AAPL")
    assert response.status_code == 200
    data = response.json()
    assert "ticker" in data
    assert "metrics" in data
    assert "equity_curve" in data
    assert "cv_folds" in data


def test_predict_endpoint(client):
    payload = {
        "headline": "Company reports record Q4 earnings beat and raises revenue guidance",
        "ticker": "AAPL",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["sentiment_label"] in ["Bullish", "Bearish", "Neutral"]
    assert data["predicted_next_day_direction"] in ["UP", "DOWN"]
    assert 0.0 <= data["up_probability"] <= 1.0
    assert 0.0 <= data["down_probability"] <= 1.0
    assert "explanation" in data


def test_upload_price_endpoint(client):
    csv_content = (
        "Date,Open,High,Low,Close,Volume\n"
        "2023-01-03,100.0,105.0,99.0,104.0,1000000\n"
        "2023-01-04,104.0,108.0,103.0,107.0,1200000\n"
        "2023-01-05,107.0,109.0,105.0,106.0,900000\n"
        "2023-01-06,106.0,111.0,105.5,110.0,1500000\n"
        "2023-01-09,110.0,114.0,109.0,113.0,1400000\n"
    )
    files = {"file": ("test_stock.csv", csv_content, "text/csv")}
    data = {"ticker": "TEST"}

    response = client.post("/api/upload-price", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["ticker"] == "TEST"
    assert res_data["total_bars"] == 5
    assert res_data["latest_close"] == 113.0
    assert "daily_return" in res_data["indicators_computed"]


def test_upload_news_endpoint(client):
    csv_content = (
        "headline,date,publisher\n"
        "Company beats Q4 revenue views and hikes quarterly dividend,2023-01-04 10:00:00,Bloomberg\n"
        "Shares plunge following federal investigation and fraud allegations,2023-01-05 14:00:00,Reuters\n"
        "Annual shareholder meeting scheduled for next month,2023-01-06 09:00:00,PR Newswire\n"
    )
    files = {"file": ("test_news.csv", csv_content, "text/csv")}
    data = {"ticker": "TEST"}

    response = client.post("/api/upload-news", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert res_data["total_articles"] == 3
    assert res_data["sentiment_breakdown"]["positive"]["count"] >= 1
    assert res_data["sentiment_breakdown"]["negative"]["count"] >= 1
    assert "top_bullish_sample" in res_data


