from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging_config import setup_logging

# Routers
from .routers.health import router as health_router
from .routers.ingest import router as ingest_router
from .routers.nlq import router as nlq_router
from .routers.models import router as models_router

# ---------------------------------------------------------
# Logging & App init
# ---------------------------------------------------------
setup_logging()

api = FastAPI(title=settings.project_name)

# ---------------------------------------------------------
# CORS middleware (no ellipsis!)
# ---------------------------------------------------------
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # or use settings.cors_origins if you have it
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Routers
# ---------------------------------------------------------

# Health: expose /health (tests call GET /health)
api.include_router(health_router,prefix="/health", tags=["health"])

# Ingest / NLQ / Models as versioned groups
api.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
api.include_router(nlq_router, prefix="/nlq", tags=["nlq"])
api.include_router(models_router, prefix="/models", tags=["models"])


@api.get("/")
def root():
    return {
        "status": "ok",
        "project": settings.project_name,
    }


# This is what pytest imports: from backend.app.main import app
app = api
