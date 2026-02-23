#!/usr/bin/env bash
set -e

uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &
streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=8501
