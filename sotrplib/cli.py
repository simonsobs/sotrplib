"""
Command-line interface for sotrplib.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import structlog

from sotrplib.config.config import Settings
from sotrplib.config.maps import MapGeneratorConfig

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

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(config.log_level),
    )
    log = structlog.get_logger()

    if isinstance(config.maps, MapGeneratorConfig):
        maps = config.maps.to_generator(log=log)
    elif isinstance(config.maps, list):
        maps = [m.to_map(log=log) for m in config.maps]
    else:
        maps = None

    if maps is not None:
        pipeline.run(maps)
    else:
        log.warning("No maps provided.")
