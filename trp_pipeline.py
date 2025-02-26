"""
Emily Biermann
6/30/23

Script to test transient pipeline driver function

Updated Oct 2024, by Allen Foster 
Testing for use as SO Time domain pipeline test.

"""
import numpy as np
from glob import glob 

from sotrplib.maps.maps import load_maps, preprocess_map
from sotrplib.maps.masks import make_src_mask
from sotrplib.maps.coadding import coadd_map_group,load_coadd_maps
from sotrplib.utils.utils import get_map_groups, get_cut_radius
from sotrplib.sources.finding import extract_sources
from sotrplib.sources.forced_photometry import enmap_extract_fluxes
from sotrplib.utils.actpol_utils import get_sourceflux_threshold
from sotrplib.source_catalog.source_catalog import load_catalog,write_json_catalog
from sotrplib.sifter.crossmatch import sift
from sotrplib.utils.plot import plot_map_thumbnail

from pixell.utils import arcmin, degree

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m",
                    "--maps",
                    help="input maps... the map.fits ones.",
                    default=["/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/15386/depth1_1538613353_pa5_f150_map.fits"],
                    nargs='+',
                )
parser.add_argument("--map-dir",
                    help="Location of the depth-1 maps.",
                    default="/scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/",
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
parser.add_argument("--flux-threshold", 
                    help="Flux threshold for catalog matching. in units of Jy. If None, uses act rms values.", 
                    default=None,
                    type=float,
                    )
parser.add_argument("--ignore-known-sources", 
                    help="Only record sources which do not have catalog matches.", 
                    action='store_true'
                    )
parser.add_argument("--candidate-limit", 
                    help="Flag map as bad beyond this number of candidate transients.", 
                    action='store',
                    default=10,
                    type=int
                    )
parser.add_argument("--coadd-n-days", 
                    help="Coadd this many days before running source finder. 0 runs every map.", 
                    action='store',
                    default=0,
                    type=int
                    )
parser.add_argument("--is-coadd", 
                    help="The input is a coadd map.", 
                    action='store_true',
                    default=False,
                    )
parser.add_argument("--nighttime-only", 
                    help="Only use maps which were observed between 2300 and 1100 local time.", 
                    action='store_true',
                    default=False,
                    )
parser.add_argument("--plot-all", 
                    help="Plot various plots to check filtering and source finding.", 
                    action='store_true'
                    )
parser.add_argument("--plot-thumbnails", 
                    help="Plot thumbnails of non-cataloged sources (i.e. transients).", 
                    action='store_true'
                    )
parser.add_argument("--save-json", 
                    help="Save the source candidate information as json files.", 
                    action='store_true'
                    )
parser.add_argument("--verbose", 
                    help="Print useful things.", 
                    action='store_true'
                    )
parser.add_argument("--simulation", 
                    help="Injected sources are known and given in the source catalog. Use that to record the expected sources in each map.", 
                    action='store_true'
                    )
args = parser.parse_args()



if len(args.maps) == 1:
    if '*' in args.maps[0]:
        args.maps = glob(args.maps[0])

if not args.is_coadd:
    indexed_map_groups,indexed_map_group_time_ranges,time_bins = get_map_groups(args.maps,
                                                                                coadd_days=args.coadd_n_days,
                                                                                restrict_to_night = args.nighttime_only,
                                                                                )
else:
    ## only works for a single input map
    ## should have name [blah]_coadd_[freq]_rho.fits
    freq_arr_idx = args.maps[0].split('_')[-2]
    indexed_map_groups = {freq_arr_idx:[args.maps]}
    indexed_map_group_time_ranges = {freq_arr_idx:[[0,1]]}
    time_bins = {freq_arr_idx:[[0]]}

for freq_arr_idx in indexed_map_groups:
    if args.verbose:
        print(freq_arr_idx)
    if '_' in freq_arr_idx:
        ## assume it's [arr]_[freq]
        arr,freq=freq_arr_idx.split('_')
    else:
        arr='all'
        freq = freq_arr_idx
    map_groups = indexed_map_groups[freq_arr_idx]
    map_group_time_ranges = indexed_map_group_time_ranges[freq_arr_idx]
    for map_group in map_groups:
        if len(map_group)==0:
            continue
        elif len(map_group)==1:
            if args.verbose:
                print('Loading map ',map_group[0])
            ## need a more robust way to check this...
            if 'coadd' in str(map_group[0]):
                mapdata = load_coadd_maps(mapfile=map_group[0],arr=arr)[0]
            else:
                mapdata = load_maps(map_group[0])
        else:
            if args.verbose:
                print('Coadding map group containing the following maps:')
                for mg in map_group:
                    print(mg)
            mapdata = coadd_map_group(map_group,
                                      freq=freq,
                                      arr=arr
                                      )
        
        preprocess_map(mapdata,
                       galmask_file = args.galaxy_mask,
                       plot_output_dir=args.plot_output,
                       PLOT=args.plot_all
                      )
        
        print('Loading Source Catalog')
        catalog_sources = load_catalog(args.source_catalog,
                                       flux_threshold = 0.0,
                                       mask_outside_map=True,
                                       mask_map=mapdata.flux
                                      )
        
        ## simply grab the pixel corresponding to the ra,dec location of the source
        ## need to flag sources which are nearby other sources, especially if the fluxes are dissimilar.
        catalog_forced_photometry_flux,catalog_forced_photometry_dflux,catalog_forced_photometry_pixels = enmap_extract_fluxes(mapdata.flux,
                                                                                                                               catalog_sources['RADeg']*degree,
                                                                                                                               catalog_sources['decDeg']*degree,
                                                                                                                               snr_map=mapdata.snr,
                                                                                                                               return_pixels=True
                                                                                                                              )

        ## get source mask radius based on flux.
        source_mask_radius = get_cut_radius(mapdata.flux,
                                            mapdata.wafer_name,
                                            mapdata.freq,
                                            source_fluxes=catalog_sources['fluxJy']
                                            )

        ## simple catalog mask
        catalog_mask = make_src_mask(mapdata.flux,
                                     catalog_forced_photometry_pixels,
                                     arr=mapdata.wafer_name,
                                     freq=mapdata.freq,
                                     mask_radius=source_mask_radius
                                    )

        ## mask the flux and snr after extracting via forced photometry
        mapdata.flux*=catalog_mask
        mapdata.snr*=catalog_mask

        print('Finding sources...')
        t0 = mapdata.map_start_time if mapdata.map_start_time else 0
        extracted_sources = extract_sources(mapdata.flux,
                                            timemap=mapdata.time_map+t0,
                                            maprms=mapdata.flux/mapdata.snr,
                                            nsigma=args.snr_threshold,
                                            minrad=[0.5,1.5,3.0,5.0,10.0,20.0,60.0],
                                            sigma_thresh_for_minrad=[0,3,5,10,50,100,200],
                                            res=mapdata.res/arcmin,
                                            )

        if not extracted_sources:
            del mapdata
            continue
        
        print(len(extracted_sources.keys()),'sources found.')

        noise_red = np.sqrt(mapdata.coadd_days) if mapdata.coadd_days else 1.
        flux_thresh = args.flux_threshold if args.flux_threshold else get_sourceflux_threshold(mapdata.freq)/1000./ noise_red
        

        print('Cross-matching found sources with catalog...')
        source_candidates,transient_candidates,noise_candidates = sift(extracted_sources,
                                                                       catalog_sources,
                                                                       map_freq = mapdata.freq,
                                                                       arr=mapdata.wafer_name
                                                                      )

        if args.simulation:
            injected_sources = []
            from sotrplib.sources.sources import SourceCandidate
            for i in range(len(catalog_sources['name'])):
                if catalog_sources['err_fluxJy'][i] == 0.0:
                    source_snr = np.inf
                else:
                    source_snr = catalog_sources['fluxJy'][i]/catalog_sources['err_fluxJy'][i]
                ic = SourceCandidate(ra=catalog_sources['RADeg'][i]%360,
                                     dec=catalog_sources['decDeg'][i],
                                     flux=catalog_sources['fluxJy'][i]*1000,
                                     dflux=catalog_sources['err_fluxJy'][i]*1000,
                                     snr=source_snr,
                                     freq=str(mapdata.freq),
                                     arr=mapdata.wafer_name,
                                     ctime=mapdata.map_ctime,
                                     sourceID=catalog_sources['name'][i],
                                     crossmatch_name=catalog_sources['name'][i], 
                                    )
                injected_sources.append(ic)

        if not args.ignore_known_sources:
            if args.save_json:
                write_json_catalog(source_candidates,
                                   out_dir=args.plot_output,
                                   out_name=str(int(mapdata.map_ctime))+'_'+str(mapdata.wafer_name)+'_'+str(mapdata.freq)+'_cataloged_sources.json',
                                  )
                if args.simulation:
                    write_json_catalog(injected_sources,
                                       out_dir=args.plot_output,
                                       out_name=str(int(mapdata.map_ctime))+'_'+str(mapdata.wafer_name)+'_'+str(mapdata.freq)+'_injected_sources.json',
                                      )
                    
        if len(transient_candidates)<=args.candidate_limit:
            if args.save_json:
                write_json_catalog(transient_candidates,
                                   out_dir=args.plot_output,
                                   out_name=str(int(mapdata.map_ctime))+'_'+str(mapdata.wafer_name)+'_'+str(mapdata.freq)+'_transient_candidates.json',
                                  )
            if args.plot_thumbnails:
                for tc in transient_candidates:
                    plot_map_thumbnail(mapdata.snr,
                                        tc.ra,
                                        tc.dec,
                                        source_name='_'.join(tc.sourceID.split(' '))+'_'+str(tc.ctime)+'_'+mapdata.wafer_name+'_'+mapdata.freq,
                                        plot_dir = args.plot_output,
                                        colorbar_range=args.snr_threshold
                                        )
            if args.verbose:
                if len(transient_candidates)>0:
                    print('To extract lightcurves for transient candidates, run the following: ')
                    executable=['python','slurm_wrapper_depth1_lightcurves.py','--ra']+['%.5f'%tc.ra for tc in transient_candidates]+['--dec']+['%.5f'%tc.dec for tc in transient_candidates]+['--data-dir',args.map_dir, '--save-thumbnails']
                    print(' '.join(executable))
        else:
            print('Too many transient candidates (%i)... something fishy'%len(transient_candidates))

        del mapdata,source_candidates,transient_candidates,noise_candidates,extracted_sources,catalog_sources