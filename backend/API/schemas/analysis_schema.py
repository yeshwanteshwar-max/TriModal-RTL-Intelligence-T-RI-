from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    rtl_path: str
    vcd_path: str | None = None
    netlist_path: str | None = None
