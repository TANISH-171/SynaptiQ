from fastapi import APIRouter
from ..schemas.base import APIResponse

router = APIRouter(prefix="/models", tags=["models"])

@router.get("/leaderboard", response_model=APIResponse)
def leaderboard():
    return APIResponse(ok=True, message="Model leaderboard coming soon")
