"""
Command-line interface for sotrplib.
"""

from argparse import ArgumentParser
from pathlib import Path

from sotrplib.config.config import Settings


def main():
    ap = ArgumentParser(
        prog="sotrp",
        usage="Use sotrplib to run a sample pipeline from a configuration file",
        description="The SO Time-Resolved Pipeline runner",
    )

    ap.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help="Path to the configuration file",
    )

    args = ap.parse_args()
    config = Settings.from_file(args.config)

    pipeline = config.to_basic()
    pipeline.run()
