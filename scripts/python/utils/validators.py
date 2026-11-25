from pathlib import Path


def require_file(path: str) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    return result
