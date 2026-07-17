"""
Command-line interface for sotrp-coadd: streaming, per-map-preprocessed
depth-1 map coadding. See sotrplib.maps.streaming_coadd for why this is a
separate tool from sotrp's own (coadd-then-preprocess) pipeline.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import structlog
from mapcat.helper import settings as mapcat_settings

from sotrplib.config.coadd import CoaddSettings
from sotrplib.maps.database import register_coadd, set_processing_end
from sotrplib.maps.streaming_coadd import stream_coadd

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)


def _check_registration_paths(config: CoaddSettings) -> None:
    """
    mapcat stores every path (depth-1 maps and coadds alike) relative to
    MAPCAT_DEPTH_ONE_PARENT. Check that up front, before doing any of the
    (expensive, per-map-preprocessed) coadding work, rather than failing
    only once register_coadd() tries to make paths relative at the very end.
    """
    if config.mapcat_registration is None or not config.mapcat_registration.enabled:
        return

    depth_one_parent = Path(mapcat_settings.depth_one_parent).resolve()
    for output in config.map_outputs:
        directory = output.directory.resolve()
        try:
            directory.relative_to(depth_one_parent)
        except ValueError:
            raise ValueError(
                f"map_outputs directory {directory} is not under "
                f"MAPCAT_DEPTH_ONE_PARENT ({depth_one_parent}). mapcat stores all "
                "paths relative to it, so mapcat_registration requires output "
                "directories to live underneath it -- either move the output "
                "directory there or set mapcat_registration.enabled to false."
            ) from None


def parse_args() -> ArgumentParser:
    ap = ArgumentParser(
        prog="sotrp-coadd",
        usage="Stream-coadd depth-1 maps, preprocessing each one before merging",
        description="Per-map-preprocessed, memory-bounded depth-1 map coadder",
    )

    ap.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        help="Path to the configuration file",
    )

    return ap.parse_args()


def main():
    args = parse_args()
    config = CoaddSettings.from_file(args.config)

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(config.log_level),
    )
    log = structlog.get_logger()

    _check_registration_paths(config)

    dependencies = config.to_dependencies()
    reader = dependencies["maps"]

    # reader.map_ids accumulates every map_id read (and marked "processing")
    # as soon as the reader is first iterated, regardless of where in the
    # run below we might fail -- so on any exception it names exactly the
    # maps this run touched but never got into a successfully registered
    # coadd, and they're marked "failed" rather than left dangling as
    # "processing" (which would otherwise make the reader silently skip
    # them on the next attempt, within stale_processing_time).
    try:
        coadd, map_ids = stream_coadd(
            maps=reader,
            preprocessors=dependencies["preprocessors"],
            coadder=dependencies["coadder"],
            log=log,
        )

        if coadd is None:
            log.warning("sotrp_coadd.no_maps_found")
            return

        # rho/kappa only exist before finalize() (which deletes them once
        # flux/snr are derived), so write outputs in two passes -- matching
        # the pattern the old standalone coadd_maps.py script already used.
        # Fields not yet available on a given pass (e.g. flux/snr on the
        # first pass) are skipped with a harmless log line by
        # MapOutputSerializer.
        output_paths: dict[str, Path] = {}
        for output in dependencies["map_outputs"]:
            output_paths.update(output.output(input_map=coadd))

        coadd.finalize()

        for output in dependencies["map_outputs"]:
            output_paths.update(output.output(input_map=coadd))

        if (
            config.mapcat_registration is not None
            and config.mapcat_registration.enabled
        ):
            coadd_id = register_coadd(
                coadd=coadd,
                map_ids=map_ids,
                coadd_name=config.mapcat_registration.coadd_name,
                coadd_type=config.mapcat_registration.coadd_type,
                output_paths=output_paths,
            )
            log.info(
                "sotrp_coadd.registered",
                coadd_id=coadd_id,
                n_maps=len(map_ids),
            )
            # A coadd's id doesn't exist until register_coadd() has already
            # finished, so unlike maps (which get set_processing_start()
            # before we know if they'll succeed), a coadd only ever gets
            # this one terminal status row, written once the outcome is
            # known.
            set_processing_end(coadd_id=coadd_id, status="completed")
    except Exception:
        log.error(
            "sotrp_coadd.failed",
            n_maps_read=len(reader.map_ids),
            map_ids=reader.map_ids,
        )
        for map_id in reader.map_ids:
            set_processing_end(map_id, status="failed")
        raise
    else:
        for map_id in map_ids:
            set_processing_end(map_id, status="completed")


if __name__ == "__main__":
    main()
