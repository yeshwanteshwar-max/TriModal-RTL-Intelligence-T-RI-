from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/multimodal", tags=["multimodal"])


class MultimodalInput(BaseModel):
    rtl: str = ""
    vcd: str = ""
    netlist: str = ""


@router.post("/analyze")
def analyze(payload: MultimodalInput) -> dict[str, object]:
    return {"rtl": bool(payload.rtl), "vcd": bool(payload.vcd), "netlist": bool(payload.netlist)}
