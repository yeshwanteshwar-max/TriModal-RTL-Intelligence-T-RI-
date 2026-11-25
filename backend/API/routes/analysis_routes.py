from fastapi import APIRouter

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/status")
def status() -> dict[str, str]:
    return {"status": "ready"}
