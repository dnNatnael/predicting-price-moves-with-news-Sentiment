/**
 * API Service for Financial News Sentiment & Alpha Dashboard
 */

const API_BASE = '/api';

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Failed to fetch API health');
  return res.json();
}

export async function fetchOverview(ticker = 'AAPL') {
  const res = await fetch(`${API_BASE}/overview?ticker=${ticker}`);
  if (!res.ok) throw new Error(`Failed to fetch overview for ${ticker}`);
  return res.json();
}

export async function fetchSentimentBenchmark(ticker = 'AAPL') {
  const res = await fetch(`${API_BASE}/sentiment-benchmark?ticker=${ticker}`);
  if (!res.ok) throw new Error('Failed to fetch sentiment benchmark');
  return res.json();
}

export async function fetchNewsFeed(ticker = 'AAPL', limit = 40) {
  const res = await fetch(`${API_BASE}/news-feed?ticker=${ticker}&limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch news feed');
  return res.json();
}

export async function fetchPrices(ticker = 'AAPL', limit = 252) {
  const res = await fetch(`${API_BASE}/prices?ticker=${ticker}&limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to fetch prices for ${ticker}`);
  return res.json();
}

export async function fetchCorrelations(ticker = 'AAPL') {
  const res = await fetch(`${API_BASE}/correlations?ticker=${ticker}`);
  if (!res.ok) throw new Error(`Failed to fetch correlations for ${ticker}`);
  return res.json();
}

export async function fetchModels(ticker = 'AAPL') {
  const res = await fetch(`${API_BASE}/models?ticker=${ticker}`);
  if (!res.ok) throw new Error(`Failed to fetch models for ${ticker}`);
  return res.json();
}

export async function fetchValidation(ticker = 'AAPL') {
  const res = await fetch(`${API_BASE}/validation?ticker=${ticker}`);
  if (!res.ok) throw new Error(`Failed to fetch validation for ${ticker}`);
  return res.json();
}

export async function predictHeadline(headline, ticker = 'AAPL') {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ headline, ticker }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Prediction request failed');
  }
  return res.json();
}

export async function uploadPriceFile(file, ticker) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('ticker', ticker);

  const res = await fetch(`${API_BASE}/upload-price`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Price upload failed');
  }
  return res.json();
}

export async function uploadNewsFile(file, ticker = '') {
  const formData = new FormData();
  formData.append('file', file);
  if (ticker) {
    formData.append('ticker', ticker);
  }

  const res = await fetch(`${API_BASE}/upload-news`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'News upload failed');
  }
  return res.json();
}

