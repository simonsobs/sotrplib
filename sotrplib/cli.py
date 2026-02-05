"""
Command-line interface for sotrplib.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import structlog

from sotrplib.config.config import Settings

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)


def parse_args() -> ArgumentParser:
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
    return args


def main():
    args = parse_args()
    config = Settings.from_file(args.config)
    pipeline = config.to_runner()
    pipeline.run()
