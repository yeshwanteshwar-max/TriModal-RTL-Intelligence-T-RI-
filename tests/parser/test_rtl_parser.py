import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.python.parsers.rtl_parser import parse_rtl


def test_parse_counter_module():
    source = Path("design/rtl/counter.v").read_text()
    modules = parse_rtl(source)
    assert modules
    assert modules[0].name == "counter"
