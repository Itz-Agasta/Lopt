from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from core.config import get_settings # Experimental
from routes.playground import router as playground_router
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

allowed_origin = os.getenv("FRONTEND_ORIGIN_URL")
if allowed_origin:
    origins =  allowed_origin   # --Production
else:
    origins = [                 # --Development
        "http://localhost",
        "http://localhost:5173"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hello from vyse",
            "allowed_origins": origins
      }

app.include_router(playground_router, prefix="/playground", tags=["playground"])