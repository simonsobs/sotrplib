"""
Build socat entries from SPT's point-source list (winter_combined.txt).

The main list has four kinds of entries, tagged by a "class" column: "ps"
(the source-finder catalog) and "thumb" (bright source thumbnails masked
before source-finding) are real positive-flux point sources; "dec"
(decrements: negative CMB dips, not flux sources) and "edge" (field-edge
artifacts, interpolated over in the map run) are not, and are excluded by
default.

Flux is read at 90GHz (the "s90raw(mJy)" column). A source above
--pointing-min-mjy is flagged "pointing" (bright/compact enough for
pointing-model fits, as SO's pointing_provider.min_flux does at 300 mJy); a
source above --monitored-min-mjy is flagged "monitored" (a forced-photometry
target). A source can be both.

    python scripts/mock_spt_run/ingest_spt_point_sources.py \\
        --input /home/axf295/2026/spt3g_software/sources/main_lists/winter_combined.txt

Writes to the socat database named by MAPCAT-style socat env vars
(socat_client_client_type=db, socat_model_database_name=...); run
`socat-migrate` against that database first to create its schema.
"""

import argparse
from pathlib import Path

from astropy import units as u
from astropy.coordinates import ICRS

POINT_SOURCE_CLASSES = ("ps", "thumb")


def load_rows(path: Path) -> list[dict]:
    """
    Parse winter_combined.txt: tab-separated, commented with '#', with one
    quoted header row (not '#'-prefixed, column names in "quotes") that
    this skips along with the comments.
    """
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 10:
                continue
            cls = parts[9].strip('"')
            if cls == "class":  # the header row
                continue
            rows.append(
                {
                    "index": int(parts[0]),
                    "ra_deg": float(parts[1]),
                    "dec_deg": float(parts[2]),
                    "s90raw_mjy": float(parts[6]),
                    "class": cls,
                }
            )
    return rows


def select_sources(
    rows: list[dict],
    monitored_min_mjy: float,
    pointing_min_mjy: float,
    classes: tuple[str, ...] = POINT_SOURCE_CLASSES,
) -> list[dict]:
    """
    Keep rows in `classes` with 90GHz flux above monitored_min_mjy (the
    inclusion floor), each flagged "monitored" (always true for a selected
    row) and "pointing" (flux above pointing_min_mjy).
    """
    selected = []
    for row in rows:
        if row["class"] not in classes:
            continue
        flux = row["s90raw_mjy"]
        if flux <= monitored_min_mjy:
            continue
        selected.append(
            {
                **row,
                "name": f"SPT-{row['index']}",
                "monitored": True,
                "pointing": flux > pointing_min_mjy,
            }
        )
    return selected


def ingest(sources: list[dict], client) -> int:
    for source in sources:
        client.create_source(
            position=ICRS(ra=source["ra_deg"] * u.deg, dec=source["dec_deg"] * u.deg),
            name=source["name"],
            flux=source["s90raw_mjy"] * u.mJy,
            flags={"monitored": source["monitored"], "pointing": source["pointing"]},
        )
    return len(sources)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--monitored-min-mjy", type=float, default=20.0)
    parser.add_argument("--pointing-min-mjy", type=float, default=300.0)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(POINT_SOURCE_CLASSES),
        help='winter_combined.txt classes to ingest (default: "ps" "thumb"; '
        '"dec" and "edge" are excluded by default).',
    )
    args = parser.parse_args(argv)

    rows = load_rows(args.input)
    sources = select_sources(
        rows, args.monitored_min_mjy, args.pointing_min_mjy, tuple(args.classes)
    )
    n_pointing = sum(s["pointing"] for s in sources)
    print(
        f"{len(rows)} rows read, {len(sources)} sources selected "
        f"({n_pointing} flagged pointing) from classes {args.classes}"
    )

    from socat.client.settings import SOCatClientSettings

    client = SOCatClientSettings().client
    ingest(sources, client)
    print(f"ingested {len(sources)} sources into socat")


if __name__ == "__main__":
    main()
