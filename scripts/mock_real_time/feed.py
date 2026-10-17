"""
Make the live mapcat of a mock real-time run, and copy the depth-1 maps of
one observation into it from the source mapcat.

The live mapcat has the same schema as the source mapcat (the script copies
the CREATE statements and the alembic version). It starts with no maps. For
each observation, feed_window() copies the rows of the maps that start in
the window of the observation: depth_one_maps, and the TOD rows and links of
these maps. It does not
copy the processing status or the pointing residuals: the pipeline makes
these.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# Tables that the live mapcat gets from the source, for each new map.
# (table, column that holds the map_id)
MAP_TABLES = [("depth_one_maps", "map_id")]
TOD_LINK_TABLE = "link_tod_to_depth_one_map"
TOD_TABLE = "tod_depth_one"


def db_time(t: datetime) -> str:
    """The text format of the mapcat DateTime columns (SQLite, UTC)."""
    return t.strftime("%Y-%m-%d %H:%M:%S")


def create_live_mapcat(source: Path, live: Path) -> None:
    """
    Make an empty mapcat at `live` with the schema of `source`. Does nothing
    if `live` exists.
    """
    if live.exists():
        return
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        statements = [
            sql
            for (sql,) in src.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
                "AND name NOT LIKE 'sqlite_%' ORDER BY type = 'index', name"
            )
        ]
        versions = src.execute("SELECT version_num FROM alembic_version").fetchall()
    finally:
        src.close()

    dst = sqlite3.connect(live)
    try:
        for sql in statements:
            dst.execute(sql)
        dst.executemany("INSERT INTO alembic_version VALUES (?)", versions)
        dst.commit()
    finally:
        dst.close()


def _columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


def maps_in_window(
    source: Path,
    start: datetime,
    end: datetime,
    bands: list[str] | None = None,
    tubes: list[str] | None = None,
    exclude: set[str] | None = None,
) -> list[tuple]:
    """
    Return (map_id, map_name) of the source maps that start in [start, end),
    in the bands and tubes, and not in `exclude` (map base names, the part
    of the map name after the last "/").
    """
    con = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT map_id, map_name, frequency, tube_slot FROM depth_one_maps "
            "WHERE start_time >= ? AND start_time < ? ORDER BY start_time",
            (db_time(start), db_time(end)),
        ).fetchall()
    finally:
        con.close()
    exclude = exclude or set()
    return [
        (map_id, name)
        for map_id, name, band, tube in rows
        if (bands is None or band in bands)
        and (tubes is None or tube in tubes)
        and name.split("/")[-1] not in exclude
    ]


def list_observations(
    source: Path,
    start: datetime,
    end: datetime,
    bands: list[str] | None = None,
    tubes: list[str] | None = None,
    exclude: set[str] | None = None,
) -> list[datetime]:
    """
    Return the start times of the observations in [start, end), in time
    order. All the depth-1 maps of one observation (all the tubes and the
    bands) have the same start time. An observation is in the list only if
    it has a map in the bands and tubes that is not in `exclude`.
    """
    con = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT map_name, frequency, tube_slot, start_time FROM depth_one_maps "
            "WHERE start_time >= ? AND start_time < ? ORDER BY start_time",
            (db_time(start), db_time(end)),
        ).fetchall()
    finally:
        con.close()
    exclude = exclude or set()
    times = sorted(
        {
            t
            for name, band, tube, t in rows
            if (bands is None or band in bands)
            and (tubes is None or tube in tubes)
            and name.split("/")[-1] not in exclude
        }
    )
    # The mapcat times are UTC, with no time zone.
    return [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times]


def feed_window(
    source: Path,
    live: Path,
    start: datetime,
    end: datetime,
    bands: list[str] | None = None,
    tubes: list[str] | None = None,
    exclude: set[str] | None = None,
) -> list[str]:
    """
    Copy the source maps that start in [start, end) to the live mapcat.
    Maps that are already in the live mapcat are not copied again. Returns
    the names of the new maps.
    """
    maps = maps_in_window(source, start, end, bands, tubes, exclude)
    if not maps:
        return []

    con = sqlite3.connect(f"file:{live}", uri=True)
    try:
        con.execute("ATTACH DATABASE ? AS src", (f"file:{source}?mode=ro",))
        existing = {r[0] for r in con.execute("SELECT map_id FROM depth_one_maps")}
        new = [(map_id, name) for map_id, name in maps if map_id not in existing]
        if not new:
            return []
        ids = [map_id for map_id, _ in new]
        marks = ",".join("?" * len(ids))

        for table, key in MAP_TABLES:
            cols = ",".join(_columns(con, table))
            con.execute(
                f"INSERT INTO main.{table} ({cols}) SELECT {cols} FROM src.{table} "
                f"WHERE {key} IN ({marks})",
                ids,
            )

        tod_ids = [
            r[0]
            for r in con.execute(
                f"SELECT DISTINCT tod_id FROM src.{TOD_LINK_TABLE} "
                f"WHERE map_id IN ({marks})",
                ids,
            )
        ]
        if tod_ids:
            tmarks = ",".join("?" * len(tod_ids))
            cols = ",".join(_columns(con, TOD_TABLE))
            con.execute(
                f"INSERT OR IGNORE INTO main.{TOD_TABLE} ({cols}) SELECT {cols} "
                f"FROM src.{TOD_TABLE} WHERE tod_id IN ({tmarks})",
                tod_ids,
            )
            cols = ",".join(_columns(con, TOD_LINK_TABLE))
            con.execute(
                f"INSERT OR IGNORE INTO main.{TOD_LINK_TABLE} ({cols}) SELECT {cols} "
                f"FROM src.{TOD_LINK_TABLE} WHERE map_id IN ({marks})",
                ids,
            )
        con.commit()
    finally:
        con.close()
    return [name for _, name in new]


def reset_unfinished_status(live: Path, start: datetime, end: datetime) -> int:
    """
    Remove the processing status rows that are not "completed" for the live
    depth-1 maps that start in [start, end). Use this before a retry of the
    daily run of a day: if the source outputs fail, the runner leaves some
    maps "processing", and the reader skips them for some hours.
    Returns the number of rows removed.
    """
    con = sqlite3.connect(live, timeout=60)
    try:
        cur = con.execute(
            "DELETE FROM time_domain_processing WHERE processing_status != 'completed' "
            "AND map_id IN (SELECT map_id FROM depth_one_maps "
            "WHERE start_time >= ? AND start_time < ?)",
            (db_time(start), db_time(end)),
        )
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def update_sky_coverage() -> int:
    """
    Fill the sky coverage of the live maps that have no coverage rows (the
    mapcat `updatesky` command). Uses MAPCAT_DATABASE_NAME and
    MAPCAT_DEPTH_ONE_PARENT, which must be set before mapcat is imported.
    Returns the number of new coverage rows.
    """
    from mapcat.database.sky_coverage import SkyCoverageTable
    from mapcat.helper import settings
    from mapcat.toolkit.update_sky_coverage import core
    from sqlmodel import func, select

    def count() -> int:
        with settings.session() as session:
            return session.execute(
                select(func.count()).select_from(SkyCoverageTable)
            ).scalar()

    before = count()
    core(session=settings.session)
    return count() - before


def read_exclude_list(path: Path | None) -> set[str]:
    """
    Return the base names (depth1_<ctime>_<tube>_<band>) of the maps to leave
    out. The CSV has a header. It has a map_name column, or band, array and
    ctime columns (for example post_fix_shifted_maps.csv).
    """
    import csv

    if path is None:
        return set()
    names = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("map_name"):
                names.add(row["map_name"].strip().split("/")[-1])
            elif row.get("ctime") and row.get("array") and row.get("band"):
                names.add(f"depth1_{row['ctime']}_{row['array']}_{row['band']}")
    return names
