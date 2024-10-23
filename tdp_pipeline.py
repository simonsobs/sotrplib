"""
Emily Biermann
6/30/23

Script to test transient pipeline driver function

Updated Oct 2024, by Allen Foster 
Testing for use as SO Time domain pipeline test.

"""
import os.path as op
import numpy as np

from astropy.table import Table

from sotdplib.maps.maps import load_maps
from sotdplib.maps.tiles import get_medrat, get_tmap_tiles
from sotdplib.maps.masks import get_masked_map
from sotdplib.utils.utils import get_sourceflux_threshold, crossmatch_mask, radec_to_str_name
from sotdplib.utils.plot import plot_source_thumbnail

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--maps",
    help="input maps... the map.fits ones.",
    default=["/scratch/gpfs/snaess/actpol/maps/depth1/release/15064/depth1_1506466826_pa4_f150_map.fits"],
    nargs='+',
)
parser.add_argument("-g", 
                    "--gridsize", 
                    help="Flatfield tile gridsize (deg)", 
                    default=1.0 # Deg
                    )
parser.add_argument("--edge-cut", 
                    help="Npixels to cut from map edges", 
                    default=10 # integer
                    )
parser.add_argument(
    "--solarsystem",
    help="path asteroid ephem directories",
    default="/scratch/gpfs/snaess/actpol/ephemerides/objects",
)
parser.add_argument(
    "--plot_output",
    help="path to output maps and code for plots",
    default="/scratch/gpfs/amfoster/so/test_transient_pipeline/",
)
parser.add_argument("-v", 
                    "--verbosity", 
                    help="Level of logging", 
                    default=1
                    )

args = parser.parse_args()


## Don't know where to put this.
from pydantic import BaseModel

class SourceCandidate(BaseModel):
    ra: float
    dec: float
    flux: float
    dflux: float
    snr: float
    ctime: float
    sourceID: str
    match_filtered: bool
    renormalized: bool
    catalog_crossmatch: bool


## hardcoded stuff bc I'm lazy.
## will be replace by querying the source catalog
sourcecat_fname = "/home/eb8912/transients/depth1/ACT_depth1-transient-pipeline/data/inputs/PS_S19_f090_2pass_optimalCatalog.fits"
galmask_fname = "/home/eb8912/transients/depth1/ACT_depth1-transient-pipeline/data/inputs/mask_for_sources2019_plus_dust.fits"
sourcecat = Table.read(sourcecat_fname)
galmask = None#enmap.read_map(galmask_fname)


for m in args.maps:
    data = load_maps(m)
    if not data:
        print(f'No data found in {m}...')
        continue

    ## get the masked flux, snr, and uncertainty maps
    flux = get_masked_map(data.flux(),
                          data.mask_map_edge(args.edge_cut), 
                          galmask,
                          data.mask_planets()
                         )
    snr = get_masked_map(data.snr(), 
                         data.mask_map_edge(args.edge_cut), 
                         galmask,
                         data.mask_planets()
                        )
    ## don't need to mask this because we don't use it for source finding.
    dflux = data.dflux()

    ## Tile the map into gridsize x gridsize  grids and calculate the median
    ## snr; "flatfield" the map using that median snr 
    med_ratio = get_medrat(snr,
                           get_tmap_tiles(data.time_map,
                                          args.gridsize,
                                          snr,
                                          id=f"{data.map_ctime}_{data.wafer_name}_{data.freq}",
                                         ),
                          )
    flux *= med_ratio
    dflux *= med_ratio ## this isn't in the ACT pipeline...
    snr *= med_ratio

    # initial point source detection
    ra_can, dec_can = data.extract_sources(snr_threshold=5.0,
                                           verbosity=args.verbosity,
                                          )

    print("Number of initial candidates: ", len(ra_can))
    if len(ra_can) == 0 :
        continue
    
    ## known source masking -- this will be the job of the sifter.
    # separate sources
    if sourcecat:
        crossmatch_radius = 2 ## arcmin , radius for considering something a crossmatch. This will need editing.
        sources = sourcecat[
            sourcecat["fluxJy"] > (get_sourceflux_threshold(data.freq) / 1000.0)
        ]
        catalog_match = crossmatch_mask(np.array([dec_can, ra_can]).T,
                                        np.array([sources["decDeg"], sources["RADeg"]]).T,
                                        crossmatch_radius,
                                        )
    else:
        catalog_match = np.array(len(ra_can) * [False])

    ## there is a blazar match in the ACT pipeline... this will just use source catalog

    if args.verbosity > 0:
        print(
            "Number of transient candidates after catalog matching: ",
            len(ra_can) - np.sum(catalog_match),
            '\nand\n',
            np.sum(catalog_match),
            ' catalog matches.'
            )

    ## The act pipeline then does a re-calculation of the matched filters for each source
    ## I will skip that here.
    # matched filter and renormalization
    #ra_new, dec_new, mf_sub, renorm_sub = perform_matched_filter(
    #    data, ra_can, dec_can, catalog_match, sourcemask, detection_threshold
    #
    #if verbosity > 0:
    #    print("number of candidates after matched filter: ", np.sum(mf_sub))
    #    print("number of candidates after renormalization: ", np.sum(renorm_sub))
    #

    source_candidates = []
    for c in zip(dec_can, ra_can):
        print(c)
        source_string_name = radec_to_str_name(c[1], c[0])
        print(source_string_name)
        source_candidates.append(SourceCandidate(ra=c[1]%360,
                                                 dec=c[0],
                                                 flux=flux.at(np.deg2rad(c), order=0, mask_nan=True),
                                                 dflux=dflux.at(np.deg2rad(c), order=0, mask_nan=True),
                                                 snr=snr.at(np.deg2rad(c), order=0, mask_nan=True),
                                                 ctime=data.time_map.at(np.deg2rad(c), order=0, mask_nan=True),
                                                 sourceID=source_string_name,
                                                 match_filtered=False,
                                                 renormalized=False,
                                                 catalog_crossmatch=crossmatch_mask(np.asarray(c).T,
                                                                                    np.array([sources["decDeg"], sources["RADeg"]]).T,
                                                                                    crossmatch_radius,
                                                                                   ),
                                                )
                                )
    

    for sc in source_candidates:
        ra = sc.ra
        dec = sc.dec
        plot_source_thumbnail(data,
                              ra,
                              dec,
                              source_name=sc.sourceID,
                              plot_dir = '/scratch/gpfs/amfoster/scratch/',
                              colorbar_range=1000
                              )
