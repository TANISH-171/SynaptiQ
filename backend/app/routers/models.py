from fastapi import APIRouter
from ..schemas.base import ApiResponse

router = APIRouter(prefix="/models", tags=["models"])

@router.get("/leaderboard", response_model=ApiResponse)
def leaderboard():
    return ApiResponse(ok=True, message="Model leaderboard coming soon")
