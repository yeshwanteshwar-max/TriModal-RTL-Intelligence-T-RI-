from pydantic import BaseModel


class OptimizationRequest(BaseModel):
    design_path: str
    objective: str = "area"
