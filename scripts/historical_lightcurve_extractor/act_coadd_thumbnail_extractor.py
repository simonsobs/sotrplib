import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
from pixell import enmap
from pixell import utils as pixell_utils

from sotrplib.filters.filters import matched_filter_depth1_map
from sotrplib.maps.maps import get_thumbnail
from sotrplib.utils.utils import get_frequency, get_fwhm, radec_to_str_name

parser = argparse.ArgumentParser()
parser.add_argument("--ps-csv", default=None, help="Point source catalog CSV file.")
parser.add_argument(
    "--additional-columns",
    default=[],
    type=str,
    nargs="+",
    help="Additional columns to read from the ps-csv.",
)
parser.add_argument("--odir", default="./")
parser.add_argument("--scratch-dir", default="./tmp/")
parser.add_argument(
    "--coadd-dir",
    default="/scratch/gpfs/SIMONSOBS/users/amfoster/act/ACT_coadd_maps/",
)
parser.add_argument(
    "--save-thumbnails", action="store_true", help="Cut out and save thumbnails."
)
parser.add_argument(
    "--thumbnail-radius",
    action="store",
    type=float,
    default=0.2,
    help="Thumbnail width, in deg.",
)
parser.add_argument("-o", "--output-thumbnail-fname", default="tmp_thumbnails.hdf5")

parser.add_argument("-T", "--tol", type=float, default=1e-4)
parser.add_argument("-S", "--fitlim", type=float, default=0)

args = parser.parse_args()


sotrp_root = Path(__file__).resolve().parents[2]
sys.path.append(str(sotrp_root))

# Get our input map files
mapfiles = sorted(glob(args.coadd_dir + "*map.fits"))
print(mapfiles)
nfile = len(mapfiles)


## read in ps-csv if provided
additional_info = {}
if args.ps_csv is not None:
    import pandas as pd

    ps_df = pd.read_csv(args.ps_csv)
    ps_ra = ps_df["ra"].values
    ps_dec = ps_df["dec"].values
    poss = np.array(
        [
            [d * pixell_utils.degree for d in ps_dec],
            [r * pixell_utils.degree for r in ps_ra],
        ]
    )
    for col in args.additional_columns:
        additional_info[col] = ps_df[col].values


thumbnail_maps = []
thumbnail_map_info = []
for fi in range(nfile):
    mapfile = mapfiles[fi]
    ivarfile = "_ivar.fits".join(mapfile.split("_map.fits"))
    name = os.path.basename(mapfile).split("_map.fits")[0]
    freq = name.split("_")[-1]

    # Check if any sources are inside our geometry
    shape, wcs = enmap.read_map_geometry(mapfile)
    pixs = enmap.sky2pix(shape, wcs, poss)
    inside = np.where(np.all((pixs.T >= 0) & (pixs.T < shape[-2:]), -1))[0]
    if len(inside) == 0:
        print("No sources in %s, skipping." % name)
        continue

    print("Processing %s with %4d srcs" % (name, len(inside)))

    ra = np.asarray(ps_ra)[inside]
    dec = np.asarray(ps_dec)[inside]

    rho_thumbs = []
    kappa_thumbs = []
    m = enmap.read_map(mapfile, sel=0)
    ivar = enmap.read_map(ivarfile, sel=0)
    for i in range(len(ra)):
        m_thumb = get_thumbnail(m, ra[i], dec[i], 2 * args.thumbnail_radius)
        ivar_thumb = get_thumbnail(ivar, ra[i], dec[i], 2 * args.thumbnail_radius)
        r, k = matched_filter_depth1_map(
            m_thumb,
            ivar_thumb,
            band_center=get_frequency(freq),
            beam_fwhm=get_fwhm(freq),
        )
        rho_thumbs.append(get_thumbnail(r, ra[i], dec[i], args.thumbnail_radius))
        kappa_thumbs.append(get_thumbnail(k, ra[i], dec[i], args.thumbnail_radius))

    del m, ivar

    for i in range(len(ra)):
        source_name = radec_to_str_name(ra[i], dec[i])
        ## save rho and kappa thumbnails with metadata specifying which one
        if args.save_thumbnails:
            thumbnail_maps.append(rho_thumbs[i])
            thumbnail_map_info.append(
                {
                    "ra": ra[i],
                    "dec": dec[i],
                    "name": source_name,
                    "arr": name.split("_AA")[0],
                    "freq": freq,
                    "maptype": "rho",
                    "Id": f"{source_name}_{name.split('_AA')[0]}_{freq}_rho",
                    **{col: additional_info[col][i] for col in args.additional_columns},
                }
            )
            thumbnail_maps.append(kappa_thumbs[i])
            thumbnail_map_info.append(
                {
                    "ra": ra[i],
                    "dec": dec[i],
                    "name": source_name,
                    "arr": name.split("_AA")[0],
                    "freq": freq,
                    "maptype": "kappa",
                    "Id": f"{source_name}_{name.split('_AA')[0]}_{freq}_kappa",
                    **{col: additional_info[col][i] for col in args.additional_columns},
                }
            )

if args.save_thumbnails:
    for i in range(len(thumbnail_maps)):
        enmap.write_hdf(
            args.odir + "/" + args.output_thumbnail_fname,
            thumbnail_maps[i],
            address=thumbnail_map_info[i]["Id"],
            extra=thumbnail_map_info[i],
        )
