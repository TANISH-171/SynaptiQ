from pydantic import BaseModel

class DatasetMeta(BaseModel):
    dataset_id: str
    name: str
    rows: int | None = None
    cols: int | None = None
    storage_path: str
