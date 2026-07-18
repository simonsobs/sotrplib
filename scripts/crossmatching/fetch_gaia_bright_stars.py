import argparse

import numpy as np
from astroquery.gaia import Gaia

"""
Download a bright-star extract from the Gaia archive for use as a local,
file-backed crossmatch catalog (compute nodes have no internet access,
so the crossmatching suite prefers file-backed catalogs).

mm-band stellar flares detectable by CMB survey instruments come from
nearby active stars, which are bright in Gaia; a G <= 15 cut keeps the
extract small while retaining essentially all plausible stellar
counterparts.

The output table (ECSV/FITS, anything astropy can write) can be loaded
with TableExternalCatalog.from_table_file / TableCatalogConfig using
    ra_column=ra, dec_column=dec, name_column=designation,
    mag_column=phot_g_mean_mag, source_type=star
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    default="gaia_bright_stars.ecsv",
    help="Output table path; format inferred from extension (.ecsv, .fits, ...).",
)
parser.add_argument(
    "--max-mag",
    type=float,
    default=15.0,
    help="Faint limit in Gaia G magnitude.",
)
parser.add_argument(
    "--ra-min",
    type=float,
    default=None,
    help="Minimum RA in deg (may be negative for ranges crossing RA=0). Omit for all-sky.",
)
parser.add_argument(
    "--ra-max",
    type=float,
    default=None,
    help="Maximum RA in deg. Omit for all-sky.",
)
parser.add_argument(
    "--dec-min",
    type=float,
    default=None,
    help="Minimum Dec in deg. Omit for all-sky.",
)
parser.add_argument(
    "--dec-max",
    type=float,
    default=None,
    help="Maximum Dec in deg. Omit for all-sky.",
)
parser.add_argument(
    "--require-parallax",
    action="store_true",
    help="Keep only sources with a significant parallax (parallax_over_error > 3), "
    "i.e. confirmed galactic stars.",
)
args = parser.parse_args()

conditions = [f"phot_g_mean_mag <= {args.max_mag}"]
if args.ra_min is not None and args.ra_max is not None:
    ra_min = args.ra_min % 360.0
    ra_max = args.ra_max % 360.0
    if ra_min <= ra_max:
        conditions.append(f"ra BETWEEN {ra_min} AND {ra_max}")
    else:
        ## range crosses RA = 0
        conditions.append(f"(ra >= {ra_min} OR ra <= {ra_max})")
if args.dec_min is not None:
    conditions.append(f"dec >= {args.dec_min}")
if args.dec_max is not None:
    conditions.append(f"dec <= {args.dec_max}")
if args.require_parallax:
    conditions.append("parallax_over_error > 3")

query = f"""
SELECT designation, ra, dec, phot_g_mean_mag, parallax, parallax_over_error,
       pmra, pmdec
FROM gaiadr3.gaia_source
WHERE {" AND ".join(conditions)}
"""
print(query)

job = Gaia.launch_job_async(query)
table = job.get_results()
print(f"Retrieved {len(table)} sources")
if len(table) > 0:
    mags = np.asarray(table["phot_g_mean_mag"], dtype=float)
    print(f"G magnitude range: {np.nanmin(mags):.2f} to {np.nanmax(mags):.2f}")
table.write(args.output, overwrite=True)
print(f"Wrote {args.output}")
