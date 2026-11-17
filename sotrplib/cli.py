"""
Command-line interface for sotrplib.
"""

import logging
import pickle
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
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

    ap.add_argument(
        "--mode",
        choices=["run", "analyze", "match"],
        default="run",
        help=(
            "'run' (default): build this config's maps and run the full "
            "pipeline, including map matching and outputs. 'analyze': build "
            "this config's maps, run everything up to (not including) map "
            "matching, and pickle the resulting MapResult list to "
            "--result-out, for a later 'match' step to consume -- e.g. one "
            "grid job per map. 'match': skip map building (this config's "
            "'maps' is ignored), load the MapResult lists pickled by one or "
            "more 'analyze' runs from --result-in, then match and write "
            "outputs as 'run' would -- e.g. one grid job per observation, "
            "after all of its per-band 'analyze' jobs finish."
        ),
    )
    ap.add_argument(
        "--result-out",
        type=Path,
        help="analyze mode: path to pickle this run's MapResult list to.",
    )
    ap.add_argument(
        "--result-in",
        type=Path,
        nargs="+",
        help="match mode: one or more MapResult pickles from 'analyze' runs.",
    )

    args = ap.parse_args()
    if args.mode == "analyze" and args.result_out is None:
        ap.error("--mode analyze requires --result-out")
    if args.mode == "match" and not args.result_in:
        ap.error("--mode match requires --result-in")
    return args


def main():
    args = parse_args()
    config = Settings.from_file(args.config)

    pipeline = config.to_runner()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(config.log_level),
    )
    log = structlog.get_logger()

    if config.seed is not None:
        # Simulations use both the random and numpy.random generators.
        random.seed(config.seed)
        np.random.seed(config.seed)
        log.info("sotrp.random_seed_set", seed=config.seed)

    if args.mode == "match":
        map_results = []
        for result_path in args.result_in:
            with open(result_path, "rb") as handle:
                map_results.extend(pickle.load(handle))
        log.info(
            "sotrp.match.loaded_results",
            n_results=len(map_results),
            n_inputs=len(args.result_in),
        )
        pipeline.match_and_output(map_results)
        return

    maps = None
    if isinstance(config.maps, MapGeneratorConfig):
        maps = config.maps.to_generator(log=log)
    elif isinstance(config.maps, list):
        maps = [m.to_map(log=log) for m in config.maps]

    if maps is None:
        log.warning("No maps provided.")
        return

    if args.mode == "analyze":
        map_results = pipeline.analyze(maps)
        args.result_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_out, "wb") as handle:
            pickle.dump(map_results, handle)
        log.info(
            "sotrp.analyze.wrote_results",
            n_results=len(map_results),
            result_out=str(args.result_out),
        )
    else:
        pipeline.run(maps)
