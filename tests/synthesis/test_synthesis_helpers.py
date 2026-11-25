from scripts.python.utils.validators import require_file
from pathlib import Path


def test_require_file_rejects_missing_file(tmp_path):
    try:
        require_file(str(tmp_path / "missing.v"))
    except FileNotFoundError:
        return
    raise AssertionError("missing file should raise")


def test_generated_netlist_artifact_layout():
    expected = [
        Path("netlist/raw/counter.rtlil"),
        Path("netlist/json/counter.json"),
        Path("netlist/verilog/counter_netlist.v"),
        Path("netlist/graphs/counter.dot"),
        Path("netlist/reports/counter_report.txt"),
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    assert not missing, f"Missing netlist artifacts: {missing}"
    images = list(Path("netlist/img").glob("counter_*.png"))
    assert images, "Missing numbered netlist image"
