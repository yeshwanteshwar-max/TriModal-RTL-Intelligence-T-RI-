from pathlib import Path
from .pyosys_runner import run_yosys


def synthesize(source: str, output_dir: str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_yosys(source, str(output / "netlist.json"))
    return output / "netlist.json"
