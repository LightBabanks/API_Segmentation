from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.inference import model_manager
from app.schemas import HealthResponse, InferResponse, VoteRequest, VoteResponse
from app.votes import compute_vote_stats, load_votes, register_vote


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.load_models()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(app=settings.app_name)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(app=settings.app_name)


@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)) -> InferResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image vide.")

    result = model_manager.predict_from_bytes(image_bytes)
    return InferResponse(**result)


@app.get("/votes", response_model=VoteResponse)
def get_votes() -> VoteResponse:
    votes = load_votes(settings.votes_file)
    return VoteResponse(**compute_vote_stats(votes))


@app.post("/vote", response_model=VoteResponse)
def vote(payload: VoteRequest) -> VoteResponse:
    model_name = payload.model.strip().lower()
    if model_name not in {"unet", "felix"}:
        raise HTTPException(status_code=400, detail="Le modèle doit être 'unet' ou 'felix'.")

    votes = register_vote(settings.votes_file, model_name)
    return VoteResponse(**compute_vote_stats(votes))
