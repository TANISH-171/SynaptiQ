from fastapi import APIRouter, UploadFile, File, HTTPException
from uuid import uuid4
from pathlib import Path
import pandas as pd

from ..services import registry
from ..schemas.dataset import DatasetMeta

router = APIRouter(prefix="/ingest", tags=["ingest"])
DATA_DIR = Path(".data/datasets")
DATA_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload", response_model=DatasetMeta)
async def upload_dataset(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in [".csv", ".xlsx", ".xls", ".json", ".parquet"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    dataset_id = str(uuid4())[:8]
    save_path = DATA_DIR / f"{dataset_id}{ext}"
    content = await file.read()
    save_path.write_bytes(content)

    # Get quick shape
    try:
        if ext == ".csv":
            df = pd.read_csv(save_path, nrows=1000)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(save_path, nrows=1000)
        elif ext == ".json":
            df = pd.read_json(save_path)
        elif ext == ".parquet":
            df = pd.read_parquet(save_path)
        rows, cols = df.shape
    except Exception:
        rows, cols = None, None

    meta = DatasetMeta(
        dataset_id=dataset_id,
        name=file.filename,
        rows=rows,
        cols=cols,
        storage_path=str(save_path),
    ).model_dump()
    registry.register_dataset(dataset_id, meta)
    return DatasetMeta(**meta)

@router.get("/list")
def list_uploaded():
    return registry.list_datasets()
