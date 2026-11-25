import shutil
import subprocess


def run_vvp(compiled: str) -> subprocess.CompletedProcess:
    "complied:str ->location"
    if not shutil.which("vvp"):
        raise RuntimeError("VVP is not installed")
    return subprocess.run(["vvp", compiled], check=True)
