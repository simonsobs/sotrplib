"""
Emily Biermann
6/30/23

Script to test transient pipeline driver function

Updated Oct 2024, by Allen Foster
Testing for use as SO Time domain pipeline test.

"""

import argparse

import numpy as np
from astropy.table import Table

from sotrplib.maps.maps import load_maps
from sotrplib.maps.masks import get_masked_map
from sotrplib.maps.tiles import get_medrat, get_tmap_tiles
from sotrplib.sources.sources import SourceCandidate
from sotrplib.utils.plot import plot_source_thumbnail
from sotrplib.utils.utils import (
    crossmatch_mask,
    get_sourceflux_threshold,
    radec_to_str_name,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--maps",
    help="input maps... the map.fits ones.",
    default=[
        "/scratch/gpfs/snaess/actpol/maps/depth1/release/15064/depth1_1506466826_pa4_f150_map.fits"
    ],
    nargs="+",
)
parser.add_argument(
    "-g",
    "--gridsize",
    help="Flatfield tile gridsize (deg)",
    default=1.0,  # Deg
)
parser.add_argument(
    "--edge-cut",
    help="Npixels to cut from map edges",
    default=10,  # integer
)
parser.add_argument(
    "--solarsystem",
    help="path asteroid ephem directories",
    default="/scratch/gpfs/snaess/actpol/ephemerides/objects",
)
parser.add_argument(
    "--plot-output",
    help="path to output maps and code for plots",
    default="/scratch/gpfs/amfoster/scratch/",
)
parser.add_argument(
    "--source-catalog",
    help="path to source catalog",
    default="/home/eb8912/transients/depth1/ACT_depth1-transient-pipeline/data/inputs/PS_S19_f090_2pass_optimalCatalog.fits",
)
parser.add_argument(
    "--galaxy-mask",
    help="path to galaxy mask",
    default="/home/eb8912/transients/depth1/ACT_depth1-transient-pipeline/data/inputs/mask_for_sources2019_plus_dust.fits",
)
parser.add_argument("-v", "--verbosity", help="Level of logging", default=1)


def main():
    args = parser.parse_args()

    ## will need to be replaced by querying the TDP source catalog
    sourcecat = Table.read(args.source_catalog)

    ## will we mask the galaxy?
    galmask = None  # enmap.read_map(args.galaxy_mask)

    for m in args.maps:
        mapdata = load_maps(m)
        if not mapdata:
            print(f"No data found in {m}...")
            continue

        ## get the masked flux, snr, and uncertainty maps
        flux = get_masked_map(
            mapdata.flux(),
            mapdata.mask_map_edge(args.edge_cut),
            galmask,
            mapdata.mask_planets(),
        )
        snr = get_masked_map(
            mapdata.snr(),
            mapdata.mask_map_edge(args.edge_cut),
            galmask,
            mapdata.mask_planets(),
        )
        ## don't need to mask this because we don't use it for source finding.
        dflux = mapdata.dflux()

        ## Tile the map into gridsize x gridsize  grids and calculate the median
        ## snr; "flatfield" the map using that median snr
        med_ratio = get_medrat(
            snr,
            get_tmap_tiles(
                mapdata.time_map,
                args.gridsize,
                snr,
                id=f"{mapdata.map_ctime}_{mapdata.wafer_name}_{mapdata.freq}",
            ),
        )
        flux *= med_ratio
        dflux *= med_ratio  ## this isn't in the ACT pipeline...
        snr *= med_ratio

        # initial point source detection
        ra_can, dec_can = mapdata.extract_sources(
            snr_threshold=5.0,
            verbosity=args.verbosity,
        )

        print("Number of initial candidates: ", len(ra_can))
        if len(ra_can) == 0:
            continue

        ## known source masking -- this will be the job of the sifter.
        # separate sources
        if sourcecat:
            crossmatch_radius = 2  ## arcmin , radius for considering something a crossmatch. This will need editing.
            sources = sourcecat[
                sourcecat["fluxJy"] > (get_sourceflux_threshold(mapdata.freq) / 1000.0)
            ]
            catalog_match = crossmatch_mask(
                np.array([dec_can, ra_can]).T,
                np.array([sources["decDeg"], sources["RADeg"]]).T,
                crossmatch_radius,
            )
        else:
            catalog_match = np.array(len(ra_can) * [False])

        if args.verbosity > 0:
            print(
                "Number of transient candidates after catalog matching: ",
                len(ra_can) - np.sum(catalog_match),
                "\nand\n",
                np.sum(catalog_match),
                " catalog matches.",
            )

        ## The act pipeline then does a re-calculation of the matched filters for each source
        ## I will skip that here.
        # matched filter and renormalization
        # ra_new, dec_new, mf_sub, renorm_sub = perform_matched_filter(
        #    data, ra_can, dec_can, catalog_match, sourcemask, detection_threshold
        #
        # if verbosity > 0:
        #    print("number of candidates after matched filter: ", np.sum(mf_sub))
        #    print("number of candidates after renormalization: ", np.sum(renorm_sub))
        #

        source_candidates = []
        for c in zip(dec_can, ra_can):
            print(c)
            source_string_name = radec_to_str_name(c[1], c[0])
            print(source_string_name)
            source_candidates.append(
                SourceCandidate(
                    ra=c[1] % 360,
                    dec=c[0],
                    flux=flux.at(np.deg2rad(c), order=0, mask_nan=True),
                    dflux=dflux.at(np.deg2rad(c), order=0, mask_nan=True),
                    snr=snr.at(np.deg2rad(c), order=0, mask_nan=True),
                    ctime=mapdata.time_map.at(np.deg2rad(c), order=0, mask_nan=True),
                    sourceID=source_string_name,
                    match_filtered=False,
                    renormalized=False,
                    catalog_crossmatch=crossmatch_mask(
                        np.asarray(c).T,
                        np.array([sources["decDeg"], sources["RADeg"]]).T,
                        crossmatch_radius,
                    ),
                )
            )

        for sc in source_candidates:
            plot_source_thumbnail(
                mapdata,
                sc.ra,
                sc.dec,
                source_name=sc.sourceID,
                plot_dir=args.plot_output,
                colorbar_range=1000,
            )
