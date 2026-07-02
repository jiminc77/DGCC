"""Simulator comparison CLI stub.

Purpose: compare MuJoCo and DLO-Lab pilot scenarios and produce the M1 report.
Implemented in M1.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulator comparison stub")
    parser.add_argument("--seed", type=int, default=0, help="deterministic random seed")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    return parser


def main() -> None:
    build_parser().parse_args()
    raise NotImplementedError("compare_sims is implemented in P0-M1")


if __name__ == "__main__":
    main()
