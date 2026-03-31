#!/usr/bin/env bash

set -e

uv pip install -r requirements.txt
uv run python scripts/ingest_documents.py
