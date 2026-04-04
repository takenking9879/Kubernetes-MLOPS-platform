#!/usr/bin/env python3
"""Strip everything up to and including the second ESC (\x1b) on each line.

Examples:
  python3 scripts/strip_after_second_esc.py logs.txt > cleaned.txt
  python3 scripts/strip_after_second_esc.py -i logs.txt    # edit in-place
  cat logs.txt | python3 scripts/strip_after_second_esc.py
"""
from __future__ import annotations
import argparse
import sys
import tempfile
from pathlib import Path

ESC = "\x1b"

def process_line(line: str) -> str:
    # Find first ESC
    first = line.find(ESC)
    if first == -1:
        return line
    # Find second ESC after the first
    second = line.find(ESC, first + 1)
    if second == -1:
        return line
    # Return content after the second ESC (exclude the ESC itself)
    return line[second + 1:]

def process_stream(inp, out):
    for line in inp:
        out.write(process_line(line))

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Strip up to and including second ESC from each line")
    p.add_argument("input", nargs="?", help="Input file (defaults to stdin)")
    p.add_argument("-i", "--inplace", action="store_true", help="Edit file in-place")
    args = p.parse_args(argv)

    if args.inplace and not args.input:
        p.error("--inplace requires an input filename")

    if args.input:
        path = Path(args.input)
        if args.inplace:
            with path.open("r", encoding="utf-8", errors="replace") as inf:
                with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
                    process_stream(inf, tf)
            Path(tf.name).replace(path)
            return 0
        else:
            with path.open("r", encoding="utf-8", errors="replace") as inf:
                process_stream(inf, sys.stdout)
            return 0
    else:
        process_stream(sys.stdin, sys.stdout)
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
