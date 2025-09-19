from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from core.config import get_settings # Experimental
from routes.playground import router as playground_router
from routes.test import router as test_router
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Get CORS origins from environment variables
def get_cors_origins():
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    if cors_origins == "*":
        return ["*"]
    return [origin.strip() for origin in cors_origins.split(",")]

origins = get_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hello from vyse",
            "allowed_origins": origins
      }

app.include_router(playground_router, prefix="/playground", tags=["playground"])
app.include_router(test_router, prefix="/test", tags=["test"])