"""G1 stiffness pilot CLI stub.

Purpose: measure stiffness/friction effect-size evidence for the human G1 gate.
Implemented in M6.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G1 stiffness pilot stub")
    parser.add_argument("--seed", type=int, default=0, help="deterministic random seed")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    return parser


def main() -> None:
    build_parser().parse_args()
    raise NotImplementedError("gate_g1 is implemented in P0-M6")


if __name__ == "__main__":
    main()
