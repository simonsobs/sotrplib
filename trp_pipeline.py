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

from sotrplib.maps.maps import load_maps
from sotrplib.maps.tiles import get_medrat, get_tmap_tiles
from sotrplib.maps.masks import get_masked_map
from sotrplib.utils.utils import get_sourceflux_threshold, crossmatch_mask, radec_to_str_name
from sotrplib.utils.plot import plot_source_thumbnail
from sotrplib.sources.sources import SourceCandidate


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m",
                    "--maps",
                    help="input maps... the map.fits ones.",
                    default=["/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/15386/depth1_1538613353_pa5_f150_map.fits"],
                    nargs='+',
                )
parser.add_argument("-g", 
                    "--gridsize", 
                    help="Flatfield tile gridsize (deg)", 
                    default=1.0 # Deg
                    )
parser.add_argument("--edge-cut", 
                    help="Npixels to cut from map edges", 
                    default=10, # integer
                    type=int
                    )
parser.add_argument("--solarsystem",
                    help="path asteroid ephem directories",
                    default="/scratch/gpfs/snaess/actpol/ephemerides/objects",
                    )
parser.add_argument("--plot-output",
                    help="path to output maps and code for plots",
                    default="/scratch/gpfs/SIMONSOBS/users/amfoster/scratch/",
                    )
parser.add_argument("--source-catalog",
                    help="path to source catalog",
                    default="/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits",
                    )
parser.add_argument("--galaxy-mask",
                    help="path to galaxy mask",
                    default="/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/mask_for_sources2019_plus_dust.fits",
                    )
parser.add_argument("-v", 
                    "--verbosity", 
                    help="Level of logging", 
                    default=1
                    )
parser.add_argument("-s", 
                    "--snr-threshold", 
                    help="SNR threshold for source finding", 
                    default=5.0,
                    type=float
                    )
parser.add_argument("--ignore-known-sources", 
                    help="Only record sources which do not have catalog matches.", 
                    action='store_true'
                    )
args = parser.parse_args()


def extract_sources(snr,
                    flux,
                    snr_threshold:float = 5.0,
                    verbosity:int = 1,
                    ):
        """
        Detects sources and returns center of mass by flux

        Args:
            snr: signal-to-noise ratio map for detection
            flux: flux map for position
            snr_threshold: signal detection threshold
            mapdata: other maps to include in output
            tags: data to include in every line, such as ctime or map name

        Returns:
            ra and dec of each source in deg

        """
        from scipy import ndimage
        labels, nlabel = ndimage.label(snr > snr_threshold)
        cand_pix = ndimage.center_of_mass(flux, 
                                          labels, 
                                          np.arange(nlabel) + 1
                                         )
        cand_pix = np.array(cand_pix)
        ra = []
        dec = []
        for pix in cand_pix:
            pos = flux.pix2sky(pix)
            d = pos[0] * 180.0 / np.pi
            if d < -90.0:
                d = 360.0 + d
            if d > 90.0:
                d = 360.0 - d
            dec.append(d)
            ra.append(pos[1] * 180.0 / np.pi)
        if len(ra) == 0:
            if verbosity > 0:
                print("did not find any candidates")

        return ra, dec

## will need to be replaced by querying the TDP source catalog
sourcecat = Table.read(args.source_catalog)

## will we mask the galaxy?
from pixell.enmap import read_map
galmask = read_map(args.galaxy_mask)


if len(args.maps) == 1:
    if '*' in args.maps[0]:
        args.maps = glob(args.maps[0] if '_map.fits' in args.maps[0] else args.maps[0]+'*map.fits')

for m in args.maps:
    print('loading ',m)
    mapdata = load_maps(m)
    if not mapdata:
        print(f'No data found in {m}...')
        continue

    
    ## get the masked flux, snr, and uncertainty maps
    flux = get_masked_map(mapdata.flux(),
                          mapdata.mask_map_edge(args.edge_cut), 
                          galmask,
                          mapdata.mask_planets()
                         )
    snr = get_masked_map(mapdata.snr(), 
                         mapdata.mask_map_edge(args.edge_cut), 
                         galmask,
                         mapdata.mask_planets()
                        )
    
    
    ## don't need to mask dflux because we don't use it for source finding.
    dflux = mapdata.dflux()

    ## Tile the map into gridsize x gridsize  grids and calculate the median
    ## snr; "flatfield" the map using that median snr 
    med_ratio = get_medrat(snr,
                           get_tmap_tiles(mapdata.time_map,
                                          args.gridsize,
                                          snr,
                                          id=f"{mapdata.map_ctime}_{mapdata.wafer_name}_{mapdata.freq}",
                                         ),
                          )
    flux *= med_ratio
    dflux *= med_ratio ## this isn't in the ACT pipeline...
    snr *= med_ratio

    # initial point source detection
    ra_can, dec_can = extract_sources(snr,
                                      flux,
                                      snr_threshold=args.snr_threshold,
                                      verbosity=args.verbosity,
                                     )

    print("Number of initial candidates: ", len(ra_can))
    if len(ra_can) == 0 :
        continue
    
    ## known source masking -- this will be the job of the sifter.
    # separate sources
    if sourcecat:
        crossmatch_radius = 5 ## arcmin , radius for considering something a crossmatch. This will need editing.
        sources = sourcecat[sourcecat["fluxJy"] > (get_sourceflux_threshold(mapdata.freq) / 1000.0)]
        catalog_match = crossmatch_mask(np.array([dec_can, ra_can]).T,
                                        np.array([sources["decDeg"], sources["RADeg"]]).T,
                                        crossmatch_radius,
                                        )
    else:
        catalog_match = np.array(len(ra_can) * [False])

   
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
        source_string_name = radec_to_str_name(c[1], c[0])
        if args.ignore_known_sources:
            ## use big mask (30 arcmin) to ignore bright source filtering crap
            if np.sum(crossmatch_mask(np.asarray(c).T,np.array([sources["decDeg"], sources["RADeg"]]).T,30.,)) > 0:
                continue
        print(source_string_name)
        
        ## this is only true or false, doesn't tell you the matched index.
        cmatch = crossmatch_mask(np.asarray(c).T,
                                np.array([sources["decDeg"], sources["RADeg"]]).T,
                                crossmatch_radius,
                                )
        candidate_observed_time = mapdata.time_map.at(np.deg2rad(c), order=0, mask_nan=True)
        cand = SourceCandidate(ra=c[1]%360,
                                dec=c[0],
                                flux=flux.at(np.deg2rad(c), order=0, mask_nan=True),
                                dflux=dflux.at(np.deg2rad(c), order=0, mask_nan=True),
                                snr=snr.at(np.deg2rad(c), order=0, mask_nan=True),
                                freq=str(mapdata.freq),
                                ctime=candidate_observed_time,
                                sourceID=source_string_name,
                                match_filtered=False,
                                renormalized=True,
                                catalog_crossmatch=cmatch,
                                crossmatch_name='',
                            )
        json_string_cand = cand.json()
        with open(args.plot_output+source_string_name.split(' ')[-1]+'_'+str(mapdata.wafer_name)+'_'+str(mapdata.freq)+'_'+str(mapdata.map_ctime)+'.json','w') as f:
            f.write(json_string_cand)

        source_candidates.append(cand)
    

    for sc in source_candidates:
        print(sc.catalog_crossmatch)
        if np.sum(sc.catalog_crossmatch)==0:
            print(sc.sourceID)
            plot_source_thumbnail(mapdata,
                                sc.ra,
                                sc.dec,
                                source_name='_'.join(sc.sourceID.split(' '))+'_'+mapdata.wafer_name+'_'+mapdata.freq+'_'+str(mapdata.map_ctime),
                                plot_dir = args.plot_output,
                                colorbar_range=200 if mapdata.freq!='f220' else 500
                                )
