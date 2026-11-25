from pydantic import BaseModel


class DesignRequest(BaseModel):
    name: str
    source_path: str
