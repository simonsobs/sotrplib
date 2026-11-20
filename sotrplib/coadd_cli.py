"""
Command-line interface for sotrp-coadd. See docs/coadding/make_coadds.md.
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import structlog
from mapcat.helper import settings as mapcat_settings

from sotrplib.config.coadd import CoaddSettings
from sotrplib.maps.database import (
    register_coadd,
    set_processing_end,
    set_processing_start,
)
from sotrplib.maps.map_coadding import stream_coadd

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)


def _check_registration_paths(config: CoaddSettings) -> None:
    """
    Make sure that each output directory is in MAPCAT_DEPTH_ONE_COADD_PARENT.
    Do this check before the coadd work starts, not at registration.
    """
    if config.mapcat_registration is None or not config.mapcat_registration.enabled:
        return

    coadd_parent = Path(mapcat_settings.depth_one_coadd_parent).resolve()
    for output in config.map_outputs:
        directory = output.directory.resolve()
        try:
            directory.relative_to(coadd_parent)
        except ValueError:
            raise ValueError(
                f"map_outputs directory {directory} is not under "
                f"MAPCAT_DEPTH_ONE_COADD_PARENT ({coadd_parent}). mapcat stores "
                "coadd paths relative to it, so mapcat_registration requires "
                "output directories to live underneath it -- set "
                "MAPCAT_DEPTH_ONE_COADD_PARENT to (a parent of) the output "
                "directory, or set mapcat_registration.enabled to false."
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
    # Off by default (see CoaddMapCatDatabaseConfig). The coadd links from
    # register_coadd() record the maps in the coadd.
    track_maps = reader.track_processing

    try:
        coadd, map_ids, failed_map_ids = stream_coadd(
            maps=reader,
            preprocessors=dependencies["preprocessors"],
            coadder=dependencies["coadder"],
            log=log,
        )

        if failed_map_ids:
            log.warning(
                "sotrp_coadd.maps_excluded",
                n_excluded=len(failed_map_ids),
                n_merged=len(map_ids),
                excluded_map_ids=failed_map_ids,
            )

        if coadd is None:
            if failed_map_ids:
                raise RuntimeError(
                    f"All {len(failed_map_ids)} input maps failed; no coadd built."
                )
            log.warning("sotrp_coadd.no_maps_found")
            return

        # finalize() removes rho and kappa. Write them before finalize().
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
            # The coadd gets its coadd_id only from register_coadd().
            # set_processing_end() needs a row, so create the row first.
            set_processing_start(coadd_id, map_type="coadd")
            set_processing_end(coadd_id, map_type="coadd", status="completed")
    except Exception:
        log.error(
            "sotrp_coadd.failed",
            n_maps_read=len(reader.map_ids),
            map_ids=reader.map_ids,
        )
        if track_maps:
            for map_id in reader.map_ids:
                set_processing_end(map_id, status="failed")
        raise
    else:
        if track_maps:
            for map_id in map_ids:
                set_processing_end(map_id, status="completed")
            for map_id in failed_map_ids:
                set_processing_end(map_id, status="failed")


if __name__ == "__main__":
    main()
