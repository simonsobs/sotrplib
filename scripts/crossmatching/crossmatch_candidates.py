import argparse

import pandas as pd
from astropy import units as u

from sotrplib.sifter.transient_crossmatch import TransientCrossmatcher
from sotrplib.source_catalog.external_catalogs import (
    GaiaCatalog,
    TableExternalCatalog,
)
from sotrplib.sources.sources import MeasuredSource

"""
Crossmatch a list of transient candidates against external catalogs
(Million Quasars + Gaia bright stars) and report the ranked counterparts.

CSV must have 'ra' and 'dec' columns in degrees; an optional 'id' column
names each candidate (row number otherwise). Negative RAs are wrapped.

Gaia can be a local extract (--gaia-file, made by fetch_gaia_bright_stars.py,
preferred since compute nodes have no internet) or the online archive
(--gaia-online). Candidates whose best match_score falls below --min-score
are reported as unmatched: likely noise, satellites, or something new.
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--candidates-csv",
    required=True,
    help="Candidate list CSV. Must have ra, dec columns in deg; optional id column.",
)
parser.add_argument(
    "--milliquas-path",
    default=None,
    help="Million Quasars catalog file (HEASARC pipe-delimited text or FITS).",
)
parser.add_argument(
    "--gaia-file",
    default=None,
    help="Local Gaia extract from fetch_gaia_bright_stars.py.",
)
parser.add_argument(
    "--gaia-online",
    action="store_true",
    help="Query the Gaia archive per candidate instead of using a local file.",
)
parser.add_argument(
    "--gaia-max-mag",
    type=float,
    default=15.0,
    help="Faint limit in Gaia G magnitude (only applied with --gaia-online; "
    "local extracts are already magnitude-limited).",
)
parser.add_argument(
    "--match-radius",
    type=float,
    default=2.0,
    help="Match radius in arcmin.",
)
parser.add_argument(
    "--min-score",
    type=float,
    default=0.0,
    help="Drop matches with match_score below this.",
)
parser.add_argument(
    "--all-matches",
    action="store_true",
    help="Write every match per candidate instead of only the best one.",
)
parser.add_argument(
    "--output",
    default="crossmatch_results.csv",
    help="Output CSV of ranked crossmatches.",
)
args = parser.parse_args()

catalogs = []
if args.milliquas_path is not None:
    catalogs.append(TableExternalCatalog.from_millionquasar_file(args.milliquas_path))
if args.gaia_file is not None:
    catalogs.append(
        TableExternalCatalog.from_table_file(
            args.gaia_file,
            ra_column="ra",
            dec_column="dec",
            name_column="designation",
            mag_column="phot_g_mean_mag",
            source_type="star",
            catalog_name="gaia_bright_stars",
        )
    )
if args.gaia_online:
    catalogs.append(GaiaCatalog(max_mag=args.gaia_max_mag))
if not catalogs:
    parser.error(
        "No catalogs configured; give --milliquas-path, --gaia-file, or --gaia-online."
    )

candidates_table = pd.read_csv(args.candidates_csv)
candidates = [
    MeasuredSource(
        ra=(float(row.ra) % 360.0) * u.deg,
        dec=float(row.dec) * u.deg,
        source_id=str(row.id) if "id" in candidates_table.columns else str(i),
        measurement_type="blind",
    )
    for i, row in enumerate(candidates_table.itertuples(index=False))
]

crossmatcher = TransientCrossmatcher(
    catalogs=catalogs,
    match_radius=args.match_radius * u.arcmin,
    min_score=args.min_score,
)
crossmatcher.crossmatch(candidates)

rows = []
for candidate in candidates:
    matches = candidate.crossmatches or []
    if not args.all_matches:
        matches = matches[:1]
    if not matches:
        rows.append(
            {
                "candidate_id": candidate.source_id,
                "ra": candidate.ra.to_value(u.deg),
                "dec": candidate.dec.to_value(u.deg),
                "match_rank": None,
                "match_id": None,
                "catalog": None,
                "match_type": None,
                "separation_arcmin": None,
                "mag": None,
                "redshift": None,
                "chance_probability": None,
                "match_score": None,
            }
        )
        continue
    for rank, match in enumerate(matches):
        rows.append(
            {
                "candidate_id": candidate.source_id,
                "ra": candidate.ra.to_value(u.deg),
                "dec": candidate.dec.to_value(u.deg),
                "match_rank": rank,
                "match_id": match.source_id,
                "catalog": match.catalog_name,
                "match_type": match.source_type,
                "separation_arcmin": match.angular_separation.to_value(u.arcmin),
                "mag": match.mag,
                "redshift": match.redshift,
                "chance_probability": match.chance_probability,
                "match_score": match.match_score,
            }
        )

results = pd.DataFrame(rows)
results.to_csv(args.output, index=False)

n_matched = results[results.match_rank.notna()].candidate_id.nunique()
print(f"{n_matched}/{len(candidates)} candidates matched; wrote {args.output}")
best = results[results.match_rank == 0]
if len(best) > 0:
    print("\nMatched candidates (best match each):")
    print(
        best[
            [
                "candidate_id",
                "match_id",
                "catalog",
                "match_type",
                "separation_arcmin",
                "mag",
                "chance_probability",
                "match_score",
            ]
        ].to_string(index=False)
    )
unmatched = results[results.match_rank.isna()]
if len(unmatched) > 0:
    print(f"\nUnmatched candidates ({len(unmatched)}):")
    print(unmatched[["candidate_id", "ra", "dec"]].to_string(index=False))
