"""
Register SPT depth-1 map files (from list_grid_maps.py's cache) into a
mapcat database.

For each file, this opens its Observation frame (spt3g_software's G3File,
stopping at the first frame; the Map frame, far larger, is never read) to
get the real start/stop time and subfield name, then writes a
DepthOneMapTable row. Map files are fetched to scratch with
sotrplib.maps.gfal.fetch_to_scratch and, by default, kept there afterward,
so a later sotrp run over the same observations does not gfal-copy them a
second time.

    python scripts/mock_spt_run/ingest_spt_maps_to_mapcat.py \\
        --maps-cache cache/ra0hdec-52.25_maps.json \\
        --grid-root davs://osg-gridftp.grid.uchicago.edu:2880/sptgrid/data/onlinemaps/ra0hdec-52.25 \\
        --scratch-dir scratch \\
        --obsids 308782645 308791373

Run against a mapcat set up by MAPCAT_DATABASE_NAME (sqlite) and
MAPCAT_DEPTH_ONE_PARENT (mapcat requires this be set, but it is not used to
resolve map_path when --grid-root is given); run `mapcatmigrate` first to
create the schema.
"""

import argparse
import json
from datetime import timezone
from pathlib import Path

from astropy.time import Time

from sotrplib.maps import gfal


def observation_frame_times(
    map_filename: str, scratch_dir: Path, keep: bool = True
) -> tuple[Time, Time, str]:
    """
    Return (start_time, stop_time, subfield) read from a map file's
    Observation frame. Stops at the first frame: the Map frame, which holds
    the actual (far larger) map data, is never read.
    """
    from spt3g import core

    with gfal.fetch_to_scratch(map_filename, scratch_dir, keep=keep) as local_g3:
        for frame in core.G3File(str(local_g3)):
            if frame.type == core.G3FrameType.Observation:
                start = Time(
                    frame["ObservationStart"].time / core.G3Units.s, format="unix"
                )
                stop = Time(
                    frame["ObservationStop"].time / core.G3Units.s, format="unix"
                )
                return start, stop, frame["SourceName"]
    raise ValueError(f"No Observation frame found in {map_filename}")


def build_row(
    obsid: str,
    band: str,
    filename: str,
    grid_root: str | None,
    scratch_dir: Path,
    keep: bool = True,
):
    """Build one DepthOneMapTable row for a cached map file. Returns
    (row, subfield)."""
    from mapcat.database import DepthOneMapTable

    source = gfal.grid_url(grid_root, filename) if grid_root else filename
    start, stop, subfield = observation_frame_times(source, scratch_dir, keep=keep)
    ctime = start + (stop - start) / 2
    row = DepthOneMapTable(
        map_name=f"spt_{obsid}_{band}",
        map_path=filename,
        tube_slot="all",  # SPT3G has no multi-tube concept, unlike SO/ACT.
        frequency=band,
        ctime=ctime.to_datetime(timezone=timezone.utc),
        start_time=start.to_datetime(timezone=timezone.utc),
        stop_time=stop.to_datetime(timezone=timezone.utc),
    )
    return row, subfield


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--maps-cache", type=Path, required=True)
    parser.add_argument(
        "--grid-root",
        default=None,
        help="Grid door root that map_path is relative to (see sotrplib.maps.gfal). "
        "If omitted, files are read from MAPCAT_DEPTH_ONE_PARENT directly.",
    )
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument(
        "--obsids",
        nargs="+",
        default=None,
        help="Only these obsids (default: every observation in the cache).",
    )
    parser.add_argument(
        "--no-keep-scratch",
        action="store_true",
        help="Delete each map file from scratch right after reading its Observation "
        "frame, instead of leaving it for a later sotrp run to reuse.",
    )
    args = parser.parse_args(argv)

    by_obsid = json.loads(args.maps_cache.read_text())
    obsids = args.obsids or sorted(by_obsid, key=int)

    from mapcat.helper import settings as mapcat_settings

    n_written = 0
    with mapcat_settings.session() as session:
        for obsid in obsids:
            bands = by_obsid.get(obsid)
            if not bands:
                print(f"skip {obsid}: not in {args.maps_cache}")
                continue
            for band, filename in bands.items():
                row, subfield = build_row(
                    obsid,
                    band,
                    filename,
                    args.grid_root,
                    args.scratch_dir,
                    keep=not args.no_keep_scratch,
                )
                session.add(row)
                n_written += 1
                print(f"{obsid} {band} ({subfield}): {row.start_time} -> {row.stop_time}")
        session.commit()
    print(f"wrote {n_written} depth_one_maps rows")


if __name__ == "__main__":
    main()
