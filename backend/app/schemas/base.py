from pydantic import BaseModel

class ApiResponse(BaseModel):
    ok: bool = True
    message: str = "success"
