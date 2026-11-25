#!/usr/bin/env python3
"""Read Verilog/SystemVerilog source files."""

from pathlib import Path
import argparse
import re
import sys

EXTENSIONS = {".v", ".vh", ".sv", ".svh"}
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)


def iter_verilog_files(paths):
    seen = set()
    for item in paths:
        path = Path(item)
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in EXTENSIONS:
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield candidate


def read_verilog_file(path: Path, strip_comments: bool = False) -> str:
    source = path.read_text(encoding="utf-8")
    if strip_comments:
        source = BLOCK_COMMENT.sub("", source)
        source = LINE_COMMENT.sub("", source)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description="Read Verilog source files")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--strip-comments", action="store_true")
    args = parser.parse_args()

    found = False
    for path in iter_verilog_files(args.paths):
        found = True
        try:
            print(f"===== {path} =====")
            print(read_verilog_file(path, args.strip_comments), end="")
        except (OSError, UnicodeDecodeError) as error:
            print(f"Error reading {path}: {error}", file=sys.stderr)
    if not found:
        print("No Verilog files found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
