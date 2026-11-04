"""
FastAPI application entry point.
Launches REST API server for the financial news sentiment & stock price prediction dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dashboard_backend")

app = FastAPI(
    title="Financial News Sentiment & Price Prediction API",
    description="Backend API powering the institutional interactive dashboard for financial NLP and stock prediction.",
    version="1.0.0",
)

# Enable CORS for frontend local development (Vite typically runs on 5173 or 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_304_cache_middleware(request, call_next):
    """
    Disable HTTP 304 Not Modified caching on development server.
    Forces every static asset and API request to return 200 OK with fresh data.
    """
    # Strip conditional request headers from client so static files always return 200 OK
    if "headers" in request.scope:
        request.scope["headers"] = [
            (name, val)
            for name, val in request.scope["headers"]
            if name.lower() not in (b"if-none-match", b"if-modified-since")
        ]

    response = await call_next(request)

    # Instruct browsers never to cache stale assets
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Register API routes
app.include_router(router)

# Mount frontend production build if available
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    logger.info(f"Mounted static frontend from {frontend_dist}")


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)

