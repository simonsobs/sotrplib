## initially from Sigurd Naess, downloaded from https://phy-act1.princeton.edu/~snaess/actpol/lightcurves/20220624/depth1_lightcurves.py

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
from pixell import bunch, enmap
from pixell import utils as pixell_utils

from sotrplib.maps.maps import get_thumbnail, get_time_safe
from sotrplib.utils.utils import fit_poss, radec_to_str_name

parser = argparse.ArgumentParser()
parser.add_argument("--ra", default=[], type=float, nargs="+")
parser.add_argument("--dec", default=[], type=float, nargs="+")

parser.add_argument("--rho-maps", nargs="+", default=[])
parser.add_argument("--odir", default="./lightcurves/")
parser.add_argument("--scratch-dir", default="./lightcurves/tmp/")
parser.add_argument("-o", "--output-lightcurve-fname", default="tmp_lightcurves.txt")
parser.add_argument(
    "--coadd-dir",
    default="/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/coadds/",
)
parser.add_argument("--subtract-coadd", action="store_true")
parser.add_argument(
    "--save-thumbnails", action="store_true", help="Cut out and save thumbnails."
)
parser.add_argument(
    "--thumbnail-radius",
    action="store",
    type=float,
    default=0.5,
    help="Thumbnail width, in deg.",
)
parser.add_argument("--output-thumbnail-fname", default="tmp_thumbnails.hdf5")

parser.add_argument("-T", "--tol", type=float, default=1e-4)
parser.add_argument("-S", "--fitlim", type=float, default=0)

args = parser.parse_args()


sotrp_root = Path(__file__).resolve().parents[2]
sys.path.append(str(sotrp_root))

# Get our input map files
rhofiles = sum([sorted(glob(fname)) for fname in args.rho_maps], [])
nfile = len(rhofiles)

## get ra,dec into format used by pixell sky2pix
poss = np.array(
    [
        [d * pixell_utils.degree for d in args.dec],
        [r * pixell_utils.degree for r in args.ra],
    ]
)

lines = []
thumbnail_maps = []
thumbnail_map_info = []
for fi in range(nfile):
    rhofile = rhofiles[fi]
    kappafile = "kappa".join(rhofile.split("rho"))
    timefile = "time".join(rhofile.split("rho"))
    infofile = rhofile.split("rho")[0] + "info.hdf"
    name = os.path.basename(rhofile).split("_rho.fits")[0]
    ## assumes files named depth1_[ctime]_pa[x]_f[freq]_[suffix]
    ttag, arr, ftag = name.split("_")[1:4]

    if args.subtract_coadd:
        coadd = args.coadd_dir + f"act_daynight_{ftag}_map.fits"
    else:
        coadd = None

    # Check if any sources are inside our geometry
    shape, wcs = enmap.read_map_geometry(rhofile)
    pixs = enmap.sky2pix(shape, wcs, poss)
    inside = np.where(np.all((pixs.T >= 0) & (pixs.T < shape[-2:]), -1))[0]
    if len(inside) == 0:
        print("No sources in %s, skipping." % name)
        continue

    print("Processing %s with %4d srcs" % (name, len(inside)))

    ra = np.asarray(args.ra)[inside]
    dec = np.asarray(args.dec)[inside]
    kappa_map = enmap.read_map(kappafile)
    # reference kappa value. Will be used to determine if individual kappa values are too low
    ref = np.max(kappa_map)
    kappa = kappa_map.at(poss[:, inside])
    if ref == 0:
        ref = 1

    rho_map = enmap.read_map(rhofile)
    if args.fitlim > 0:
        pos_fit, sn_fit = fit_poss(
            rho_map.preflat[0], kappa_map.preflat[0], poss[:, inside], tol=args.tol
        )
        good = sn_fit >= args.fitlim
        poss[:, inside[good]] = pos_fit[:, good]

    rho = rho_map.at(poss[:, inside])

    kappa_thumbs = []
    rho_thumbs = []
    if args.save_thumbnails:
        for i in range(len(ra)):
            kappa_thumbs.append(
                get_thumbnail(kappa_map, ra[i], dec[i], args.thumbnail_radius)
            )
            rho_thumbs.append(
                get_thumbnail(rho_map, ra[i], dec[i], args.thumbnail_radius)
            )

    del kappa_map, rho_map

    time_map = enmap.read_map(timefile)
    info = bunch.read(infofile)
    with pixell_utils.nowarn():
        t = get_time_safe(time_map, poss[:, inside]) + info.t
    del time_map, info

    good = np.where(kappa[0] > ref * args.tol)[0]
    rho, kappa, t = rho[:, good], kappa[:, good], t[good]

    coadd_thumbs = []
    if coadd:
        coadd_flux_map = enmap.read_map(coadd, sel=0)
        coadd_flux = coadd_flux_map.at(poss[:, inside])
        if args.save_thumbnails:
            for i in range(len(ra)):
                coadd_thumbs.append(
                    get_thumbnail(coadd_flux_map, ra[i], dec[i], args.thumbnail_radius)
                )
                ## assume depth1 map noise negligible in coadd subtraction
                rho_thumbs[i] -= coadd_thumbs[i] * kappa_thumbs[i]
        del coadd_flux_map
    else:
        coadd_flux = 0.0

    flux = rho / kappa - coadd_flux
    dflux = kappa**-0.5

    ## intensity snr only
    snr = flux[0] / dflux[0]
    if len(snr) == 0:
        continue

    for i, gi in enumerate(good):
        source_name = radec_to_str_name(ra[gi], dec[gi])
        line = "%10.0f, %s, %5f, %.5f, %3s, %4s, %8.2f," % (
            t[i],
            source_name,
            ra[gi],
            dec[gi],
            arr,
            ftag,
            snr[i],
        )
        for f, df in zip(flux[:, i], dflux[:, i]):
            line += " %8.1f, %6.1f," % (f, df)
        line += " %s\n" % ttag
        lines.append(line)
        ## save rho and kappa thumbnails with metadata specifying which one
        if args.save_thumbnails:
            thumbnail_maps.append(rho_thumbs[gi])
            thumbnail_map_info.append(
                {
                    "ra": ra[gi],
                    "dec": dec[gi],
                    "name": source_name,
                    "t": t[i],
                    "arr": arr,
                    "freq": ftag,
                    "maptime": ttag,
                    "maptype": "rho",
                    "Id": f"{source_name}_{ttag}_{arr}_{ftag}_rho",
                }
            )
            thumbnail_maps.append(kappa_thumbs[gi])
            thumbnail_map_info.append(
                {
                    "ra": ra[gi],
                    "dec": dec[gi],
                    "name": source_name,
                    "t": t[i],
                    "arr": arr,
                    "freq": ftag,
                    "maptime": ttag,
                    "maptype": "kappa",
                    "Id": f"{source_name}_{ttag}_{arr}_{ftag}_kappa",
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

lc_file_header = "#Obs Time, SourceID, RA(deg), DEC(deg), Array, Frequency, SNR, I Flux (mJy), I Flux Unc (mJy), Q Flux (mJy), Q Flux Unc (mJy), U Flux (mJy), U Flux Unc (mJy), MapID\n"
ofile = "%s/%s" % (args.odir, args.output_lightcurve_fname)
times = [float(line.split(",")[0]) for line in lines]
order = np.argsort(times)
with open(ofile, "w") as f:
    f.write(lc_file_header)
    for oi in order:
        f.write(lines[oi])
