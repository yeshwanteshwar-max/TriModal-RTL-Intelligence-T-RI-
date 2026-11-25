from pathlib import Path


def calculate_activity(vcd_path: str) -> dict[str, float]:
    text = Path(vcd_path).read_text(encoding="utf-8")
    transitions = sum(line.startswith(("0", "1", "x", "X", "z", "Z")) for line in text.splitlines())
    lines = max(1, len(text.splitlines()))
    return {"transitions": transitions, "density": transitions / lines}
