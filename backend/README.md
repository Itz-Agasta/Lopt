# Lopt Backend

The **Lopt backend** is a FastAPI-powered REST API service designed to detect deepfakes in both **images** and **videos**. It leverages two state-of-the-art models:

- **Virtus** – for deepfake **image** detection.
- **Scarlet** – for deepfake **video** detection.

This repository is part of the [Lopt](https://github.com/Itz-Agasta/Lopt) monorepo.  
The frontend (React-based) can be found in the [`frontend/`](https://github.com/Itz-Agasta/Lopt/tree/main/frontend) folder.

---

## Project Structure

```bash
backend/
├── app/
│   ├── Datasets/             # Test datasets
│   ├── routes/               # Core routes logic
│   ├── models/               # ML model inference scripts
│   ├── main.py               # FastAPI app entry point
│   └── main.py
├── Dockerfile
├── .dockerignore
├── .env
├── pyproject.toml            # UV dependency manager config
└── README.md                 # You are here!
```

---

## Features

- Deepfake detection for both **images** and **videos**
- Modular FastAPI backend structure
- Model inference using pre-trained models hosted on Hugging Face
- Optimized for `fast REST API` inference
- CORS enabled for frontend integration
- Swagger UI for easy API testing

---

## Tech Stack

- **FastAPI** – Web framework for APIs
- **PyTorch** – Model inference
- **Transformers** – Video model via `facebook/timesformer`
- **Pillow** – Image handling
- **MoviePy / decord** – Video preprocessing
- **Pydantic v2** – Settings and request validation
- **Poetry** – Dependency management

---

## Installation (Local)

```bash
git clone https://github.com/Itz-Agasta/Lopt.git
cd Lopt/backend

# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
```

---

## Running the API (Local)

```bash
uv run fastapi dev
```

Then visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Running with Docker

You can also pull and run the backend API using Docker from Docker Hub:

```bash
docker pull agasta/lopt:latest
docker run -d -p 8000:8000 agasta/lopt
```

After starting the container, the API will be available at:
http://localhost:8000/docs

> DockerHub Repository: https://hub.docker.com/repository/docker/agasta/lopt/

Make sure to set any required environment variables if needed by using `-e flags` or mounting a `.env` file.

## 🔎 API Endpoints

| Method | Endpoint      | Description                              |
| ------ | ------------- | ---------------------------------------- |
| POST   | `/playground` | Upload an image/video to detect fakeness |
| POST   | `/try`        | Trial API endpoint                       |

---

## Models

The models are load & cached automatically from Hugging Face:

- **Virtus (Image)**: [`Itz-Agasta/Virtus`](https://huggingface.co/agasta/Virtus)
- **Scarlet (Video)**: [`Itz-Agasta/Scarlet`](https://huggingface.co/agasta/Scarlet)

Ensure you have internet access on first run for model download.

---
