from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/design", tags=["design"])


class RTLSource(BaseModel):
    source: str


@router.post("/parse")
def parse_design(payload: RTLSource) -> dict[str, object]:
    return {"source": payload.source, "status": "received"}
