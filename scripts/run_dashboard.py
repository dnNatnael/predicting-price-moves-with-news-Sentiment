"""
Interactive Dashboard Launcher
Starts the FastAPI backend server (which also serves the compiled React frontend).

Usage:
    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the FastAPI + React Financial News Sentiment Dashboard."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to serve on (default: 8000).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload on backend code changes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("FINANCIAL NEWS SENTIMENT & ALPHA DASHBOARD (FASTAPI + REACT)")
    print(f"Server URL:     http://{args.host}:{args.port}")
    print(f"API Docs:       http://{args.host}:{args.port}/docs")
    print(f"Interactive UI: http://{args.host}:{args.port}/")
    print("=" * 80)
    print("Press Ctrl+C to stop the server.\n")

    from backend.main import app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
