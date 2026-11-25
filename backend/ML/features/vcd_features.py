from pathlib import Path


def extract_vcd_features(path: str) -> dict[str, int]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {"lines": len(lines), "transitions": sum(line[:1] in "01" for line in lines)}
