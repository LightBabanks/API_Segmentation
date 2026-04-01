from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str


class InferResponse(BaseModel):
    unet_score: float = Field(..., ge=0.0, le=1.0)
    felix_score: float = Field(..., ge=0.0, le=1.0)
    unet_mask_base64: str
    felix_mask_base64: str
    latency_ms: float


class VoteRequest(BaseModel):
    model: str


class VoteResponse(BaseModel):
    unet_votes: int
    felix_votes: int
    total_votes: int
    unet_percentage: float
    felix_percentage: float
