# backend/app/schemas/base.py
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class APIResponse(BaseModel):
    """Base envelope for success responses (extend if needed)."""
    ok: bool = True
    message: Optional[str] = None
