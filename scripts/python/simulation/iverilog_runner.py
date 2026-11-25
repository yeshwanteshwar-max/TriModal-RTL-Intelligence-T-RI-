import shutil
import subprocess
from pathlib import Path


def compile_and_run(
    source: str,
    testbench: str,
    output: str,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess:
    if not shutil.which("iverilog"):
        raise RuntimeError("Icarus Verilog is not installed")
    subprocess.run(["iverilog", "-o", output, source, testbench], check=True, cwd=cwd)
    return subprocess.run(["vvp", output], check=True, cwd=cwd)
