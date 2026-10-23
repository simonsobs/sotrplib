"""
List the SPT depth-1 map files of one subfield and group them by
observation, writing the result to a JSON cache.

Listing /sptgrid directly is slow (NFS, many files) and can congest
bandwidth other SPT users need, so this should be run once per subfield and
then not run again: everything downstream (mapcat ingest, run-config
building) reads the JSON cache this writes, not the directory.

    # one real directory listing (slow: do this sparingly)
    python scripts/mock_spt_run/list_grid_maps.py \\
        --source-dir /sptgrid/data/onlinemaps/ra0hdec-52.25 \\
        --out cache/ra0hdec-52.25_maps.json

    # re-parse a listing already captured to a text file (no filesystem
    # access at all -- use this to rebuild the cache from, e.g., a saved
    # `ls` transcript, instead of listing the directory again)
    python scripts/mock_spt_run/list_grid_maps.py \\
        --from-listing cache/ra0hdec-52.25_ls.txt \\
        --out cache/ra0hdec-52.25_maps.json

Each file is named "<obsid>_<band>GHz_tonly.g3.gz" (observed on
/sptgrid/data/onlinemaps/<subfield>/). The cache groups them by obsid:
{"<obsid>": {"<band>GHz": "<filename>", ...}, ...}.
"""

import argparse
import json
import re
from pathlib import Path

FILENAME_RE = re.compile(r"^(?P<obsid>\d+)_(?P<band>\d+GHz)_tonly\.g3\.gz$")


def parse_filenames(names: list[str]) -> dict[str, dict[str, str]]:
    """Group map filenames by obsid: {obsid: {band: filename}}. Names that
    don't match the expected pattern are skipped."""
    by_obsid: dict[str, dict[str, str]] = {}
    for name in names:
        m = FILENAME_RE.match(name)
        if not m:
            continue
        by_obsid.setdefault(m["obsid"], {})[m["band"]] = name
    return by_obsid


def list_source_dir(source_dir: Path) -> list[str]:
    return [p.name for p in source_dir.iterdir() if p.is_file()]


def list_from_listing(path: Path) -> list[str]:
    """Read filenames from a plain-text listing: one name per line (for
    example, `ls` output already captured to a file)."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--source-dir", type=Path, help="List this directory (slow: do this sparingly)."
    )
    source.add_argument(
        "--from-listing",
        type=Path,
        help="Re-parse filenames from a text file instead of listing a directory.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    names = (
        list_source_dir(args.source_dir)
        if args.source_dir
        else list_from_listing(args.from_listing)
    )
    by_obsid = parse_filenames(names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(by_obsid, indent=1, sort_keys=True))
    n_bands = sum(len(bands) for bands in by_obsid.values())
    print(f"{len(names)} files listed, {len(by_obsid)} observations, {n_bands} map files")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
