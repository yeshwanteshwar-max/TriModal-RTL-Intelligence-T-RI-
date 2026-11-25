from pathlib import Path


def test_counter_source_exists():
    assert Path("design/rtl/counter.v").is_file()
