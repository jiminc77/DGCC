"""DLO-Lab smoke-test CLI stub.

Purpose: run the DLO-Lab bring-up smoke scenario. Implemented in M1.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DLO-Lab smoke-test stub")
    parser.add_argument("--seed", type=int, default=0, help="deterministic random seed")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    return parser


def main() -> None:
    build_parser().parse_args()
    raise NotImplementedError("smoke_dlolab is implemented in P0-M1")


if __name__ == "__main__":
    main()
