#!/usr/bin/env python3
"""Read Verilog/SystemVerilog source files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from collections.abc import Iterable, Iterator


VERILOG_EXTENSIONS = {".v", ".vh", ".sv", ".svh"}
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)


def iter_verilog_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield Verilog source files from files or recursively from directories."""
    seen: set[Path] = set()

    for path in paths:
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in VERILOG_EXTENSIONS:
                continue

            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def read_verilog_file(path: Path, strip_comments: bool = False) -> str:
    """Read a Verilog file and optionally remove block and line comments."""
    source = path.read_text(encoding="utf-8")
    if strip_comments:
        source = BLOCK_COMMENT.sub("", source)
        source = LINE_COMMENT.sub("", source)
    return source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read Verilog and SystemVerilog source files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Source files or directories to read",
    )
    parser.add_argument(
        "--strip-comments",
        action="store_true",
        help="Remove // and /* */ comments from displayed source",
    )
    parser.add_argument(
        "--no-headers",
        action="store_true",
        help="Display source only, without file-name headers",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    found = False

    for path in iter_verilog_files(args.paths):
        found = True
        try:
            source = read_verilog_file(path, args.strip_comments)
        except (OSError, UnicodeDecodeError) as error:
            print(f"Error reading {path}: {error}", file=sys.stderr)
            continue

        if not args.no_headers:
            print(f"===== {path} =====")
        print(source, end="" if source.endswith("\n") else "\n")

    if not found:
        print("No Verilog files found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
