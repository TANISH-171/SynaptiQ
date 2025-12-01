from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("", summary="Liveness probe")
def health():
    return {"status": "ok"}
