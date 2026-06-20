#!/usr/bin/env python3
"""Street Cleaning solver CLI.

Reads instance files from ./input, writes submissions to ./output using the
SAME file name, then self-scores each one.

Usage:
    python main.py                 # solve every file in input/
    python main.py input/example.txt   # solve a single file
    python main.py --score-only output/example.txt input/example.txt
"""
from __future__ import annotations

import os
import sys
import time

from src.parser import parse_instance
from src.solver1 import solve
from src.writer import write_solution
from src.scorer import score_submission

INPUT_DIR = "input"
OUTPUT_DIR = "output"


def run_one(in_path: str) -> None:
    name = os.path.basename(in_path)
    out_path = os.path.join(OUTPUT_DIR, name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    inst = parse_instance(in_path)
    sol = solve(inst)
    write_solution(sol, out_path)
    dt = time.time() - t0

    report = score_submission(inst, out_path)
    print(f"=== {name}  (N={inst.n} M={inst.m} C={inst.c} alpha={inst.alpha}) "
          f"solved in {dt:.2f}s ===")
    print(report.summary())
    print(f"  -> wrote {out_path}\n")


def resolve_input_path(arg: str) -> str:
    """Accept either a path or a bare input/ filename."""
    if os.path.exists(arg):
        return arg
    candidate = os.path.join(INPUT_DIR, arg)
    if os.path.exists(candidate):
        return candidate
    return arg


def main(argv):
    args = [a for a in argv if not a.startswith("--")]

    if args:
        targets = [resolve_input_path(a) for a in args]
    else:
        if not os.path.isdir(INPUT_DIR):
            print(f"no input/ directory and no files given")
            return 1
        targets = sorted(
            os.path.join(INPUT_DIR, f)
            for f in os.listdir(INPUT_DIR)
            if not f.startswith(".")
        )
        if not targets:
            print("input/ is empty")
            return 1

    for path in targets:
        run_one(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
