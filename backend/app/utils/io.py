import pandas as pd
from pathlib import Path

def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in [".csv"]:
        return pd.read_csv(p)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    if p.suffix.lower() in [".json"]:
        return pd.read_json(p, lines=False)
    if p.suffix.lower() in [".parquet"]:
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type: {p.suffix}")
