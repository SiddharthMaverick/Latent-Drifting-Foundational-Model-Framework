"""Unified CLI entry point."""

from __future__ import annotations

import argparse
import runpy


_COMMANDS = {
    "cache-latents": "latent_drifting.cli.cache_latents",
    "train-mae": "latent_drifting.cli.train_mae",
    "train-generator": "latent_drifting.cli.train_generator",
    "sample": "latent_drifting.cli.sample",
    "evaluate": "latent_drifting.cli.evaluate",
    "compute-real-stats": "latent_drifting.cli.compute_real_stats",
    "export": "latent_drifting.cli.export",
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="latent-drifting")
    parser.add_argument("command", choices=tuple(_COMMANDS))
    args, remaining = parser.parse_known_args()
    import sys

    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module(_COMMANDS[args.command], run_name="__main__")


if __name__ == "__main__":
    main()
