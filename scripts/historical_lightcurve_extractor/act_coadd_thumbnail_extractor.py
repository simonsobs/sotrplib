import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from pixell import enmap
from pixell import utils as pixell_utils
from tqdm import tqdm

from sotrplib.filters.filters import matched_filter_depth1_map
from sotrplib.maps.maps import get_thumbnail
from sotrplib.utils.utils import get_frequency, get_fwhm, radec_to_str_name

"""
Can run this script to extract thumbnails of matched-filtered coadd maps around point sources listed in a CSV file.
CSV must have 'ra' and 'dec' columns in degrees.
Additional columns of the CSV can be stored as metadata in the output HDF5 files by using the --additional-columns flag.

"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ps-csv",
    required=True,
    help="Point source catalog CSV file. Must have ra, dec columns.",
)
parser.add_argument(
    "--additional-columns",
    default=[],
    type=str,
    nargs="+",
    help="Additional columns to read from the ps-csv.",
)
parser.add_argument("--odir", default="./", help="Output directory for thumbnails.")
parser.add_argument(
    "--coadd-dir",
    default="/scratch/gpfs/SIMONSOBS/users/amfoster/act/ACT_coadd_maps/",
    help="Directory containing coadd map files.",
)
parser.add_argument(
    "--thumbnail-radius",
    action="store",
    type=float,
    default=0.2,
    help="Thumbnail half-width, in deg.",
)
args = parser.parse_args()


# Get our input map files
mapfiles = sorted(glob(args.coadd_dir + "*map.fits"))
print(mapfiles)
nfile = len(mapfiles)


## read in ps-csv
additional_info = {}
ps_df = pd.read_csv(args.ps_csv)
ps_ra = ps_df["ra"].values
ps_dec = ps_df["dec"].values
poss = np.array(
    [
        [d * pixell_utils.degree for d in ps_dec],
        [r * pixell_utils.degree for r in ps_ra],
    ]
)
## read in additional columns
for col in args.additional_columns:
    additional_info[col] = ps_df[col].values


for fi in range(nfile):
    mapfile = mapfiles[fi]
    ivarfile = "_ivar.fits".join(mapfile.split("_map.fits"))
    map_name = os.path.basename(mapfile).split("_map.fits")[0]
    freq = map_name.split("_")[-1]

    # Check if any sources are inside our geometry
    shape, wcs = enmap.read_map_geometry(mapfile)
    pixs = enmap.sky2pix(shape, wcs, poss)
    inside = np.where(np.all((pixs.T >= 0) & (pixs.T < shape[-2:]), -1))[0]
    if len(inside) == 0:
        print("No sources in %s, skipping." % map_name)
        continue

    print("Processing %s with %4d srcs" % (map_name, len(inside)))

    ra = np.asarray(ps_ra)[inside]
    dec = np.asarray(ps_dec)[inside]

    for i in tqdm(range(len(ra))):
        thumbnail_maps = []
        thumbnail_map_info = []
        ## make box 2x radius to allow better noise estimation
        box = (
            np.array(
                [
                    [
                        dec[i] - 2 * args.thumbnail_radius,
                        ra[i] - 2 * args.thumbnail_radius,
                    ],
                    [
                        dec[i] + 2 * args.thumbnail_radius,
                        ra[i] + 2 * args.thumbnail_radius,
                    ],
                ]
            )
            * pixell_utils.degree
        )

        m = enmap.read_map(mapfile, sel=0, box=box) * 1e-6  # convert to uK
        ivar = enmap.read_map(ivarfile, sel=0, box=box) * 1e12  # convert to uK^-2

        if np.all(ivar == 0) or np.isnan(ivar).all():
            print(f"Source at RA: {ra[i]}, DEC: {dec[i]} has no coverage, skipping.")
            continue

        r, k = matched_filter_depth1_map(
            m,
            ivar,
            band_center=get_frequency(freq),
            beam_fwhm=get_fwhm(freq),
        )

        ## get smaller thumb to remove edge effects.
        rho_thumb = get_thumbnail(r, ra[i], dec[i], args.thumbnail_radius)
        kappa_thumb = get_thumbnail(k, ra[i], dec[i], args.thumbnail_radius)

        del r, k, m, ivar
        source_name = radec_to_str_name(ra[i], dec[i])
        ## save rho and kappa thumbnails with metadata specifying which one
        thumbnail_maps.append(rho_thumb)
        thumbnail_map_info.append(
            {
                "ra": ra[i],
                "dec": dec[i],
                "name": source_name,
                "arr": map_name.split("_AA")[0],
                "freq": freq,
                "maptype": "rho",
                "Id": f"{source_name}_{map_name.split('_AA')[0]}_{freq}_rho",
                **{
                    col: additional_info[col][inside][i]
                    for col in args.additional_columns
                },
            }
        )
        thumbnail_maps.append(kappa_thumb)
        thumbnail_map_info.append(
            {
                "ra": ra[i],
                "dec": dec[i],
                "name": source_name,
                "arr": map_name.split("_AA")[0],
                "freq": freq,
                "maptype": "kappa",
                "Id": f"{source_name}_{map_name.split('_AA')[0]}_{freq}_kappa",
                **{
                    col: additional_info[col][inside][i]
                    for col in args.additional_columns
                },
            }
        )

        for tm in range(len(thumbnail_maps)):
            enmap.write_hdf(
                args.odir
                + "/"
                + source_name.split(" ")[-1]
                + "_"
                + map_name
                + "_thumbnail.hdf5",
                thumbnail_maps[tm],
                address=thumbnail_map_info[tm]["Id"],
                extra=thumbnail_map_info[tm],
            )
