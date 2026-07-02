"""G2 gate measurement CLI stub.

Purpose: measure dual-goal correlation evidence for the human G2 gate.
Implemented in M5.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G2 gate measurement stub")
    parser.add_argument("--seed", type=int, default=0, help="deterministic random seed")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    return parser


def main() -> None:
    build_parser().parse_args()
    raise NotImplementedError("gate_g2 is implemented in P0-M5")


if __name__ == "__main__":
    main()
