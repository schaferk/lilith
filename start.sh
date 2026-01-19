#!/bin/bash

# Start backend in background
# We run uvicorn on port 8000
echo "Starting Backend..."
cd /app/backend
# Use uvicorn directly or via python -m
# Make sure we are in the root of the copy so imports work
# We copied '.' to /app/backend, so 'web.api.main' is at /app/backend/web/api/main.py
# PYTHONPATH should include /app/backend
export PYTHONPATH=$PYTHONPATH:/app/backend
# Don't use reload in prod
uvicorn web.api.main:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds for backend
sleep 5

# Start frontend
echo "Starting Frontend..."
cd /app/frontend
# Pass -p 7860 specifically because HF expects it
npm start -- -p 7860
