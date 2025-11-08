from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging_config import setup_logging
from .routers import health, ingest, nlq, models

setup_logging()
app = FastAPI(title=settings.project_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(nlq.router)
app.include_router(models.router)

@app.get("/", tags=["root"])
def root():
    return {"app": settings.project_name, "env": settings.env}
