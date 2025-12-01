from fastapi import APIRouter
from ..schemas.taskplan import TaskPlan, TaskStep

router = APIRouter(tags=["nlq"])

@router.post("/route", response_model=TaskPlan)
def route_query(query: str):
    """
    Very simple placeholder:
    - If user mentions "forecast", return a forecast TaskPlan
    - Otherwise return ingest/profile
    """
    q = (query or "").lower()
    if "forecast" in q:
        steps = [
            TaskStep(action="select_dataset", params={"dataset_id": "LAST"}),
            TaskStep(action="forecast", params={"target": "y", "horizon": 12, "model": "prophet"})
        ]
    else:
        steps = [
            TaskStep(action="profile", params={"dataset_id": "LAST"})
        ]
    return TaskPlan(query=query, steps=steps)
