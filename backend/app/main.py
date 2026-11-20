"""Corn Pest Assessment backend.

Receives a photo from the app, runs the four trained models through the same decision as the
whole-app test (ml/src/flow.py, copied into app/services), and returns the result. It never
touches the database: the app saves results to Supabase itself (database/README.md).

Run from the backend folder:
    .venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Try it in a browser at http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    predict.load_models()
    yield


app = FastAPI(title="Corn Pest Assessment API", version="1.0",
              description="Photo of a corn plant in, result out: healthy, severity only, or pest + severity. "
                          "No treatment advice.",
              lifespan=lifespan)
# Lets the Flutter app also run in a web browser during development.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(predict.router)
