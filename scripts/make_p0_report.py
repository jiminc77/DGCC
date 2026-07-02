"""P0 final report CLI stub.

Purpose: assemble the final P0 report after all human-gated decisions.
Implemented in M7.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P0 final report stub")
    parser.add_argument("--seed", type=int, default=0, help="deterministic random seed")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    return parser


def main() -> None:
    build_parser().parse_args()
    raise NotImplementedError("make_p0_report is implemented in P0-M7")


if __name__ == "__main__":
    main()
