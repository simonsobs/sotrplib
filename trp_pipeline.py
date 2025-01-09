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
from sotrplib.maps.coadding import coadd_map_group,load_coadd_maps
from sotrplib.utils.utils import get_map_groups
from sotrplib.sources.finding import find_sources_quick
from sotrplib.utils.actpol_utils import get_sourceflux_threshold
from sotrplib.source_catalog.source_catalog import load_act_catalog
from sotrplib.sifter.crossmatch import sift
from sotrplib.utils.plot import plot_map_thumbnail

from pixell.utils import arcmin

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
args = parser.parse_args()



if len(args.maps) == 1:
    if '*' in args.maps[0]:
        args.maps = glob(args.maps[0])
print(args.maps)
map_groups,map_group_time_windows = get_map_groups(args.maps,
                                                   coadd_days=args.coadd_n_days
                                                   )

for map_group in map_groups:
    if len(map_group)==0:
        continue
    elif len(map_group)==1:
        print('Loading map ',map_group[0])
        ## need a more robust way to check this...
        if 'coadd' in str(map_group[0]):
            mapdata = load_coadd_maps(mapfile=map_group[0])[0]
        else:
            mapdata = load_maps(map_group[0])
    else:
        print('Coadding map group containing the following maps:')
        for mg in map_group:
            print(mg)
        mapdata = coadd_map_group(map_group)
    
    preprocess_map(mapdata,
                   galmask_file = args.galaxy_mask,
                   plot_output_dir=args.plot_output,
                   PLOT=args.plot_all
                   )

    print('Finding sources...')
    t0 = mapdata.map_start_time if mapdata.map_start_time else 0
    extracted_sources = find_sources_quick(mapdata.flux,
                                            timemap=mapdata.time_map+t0,
                                            maprms=mapdata.flux/mapdata.snr,
                                            mapmean=0.0,
                                            nsigma=args.snr_threshold,
                                            minrad=[0.5,1.5,3.0,5.0,10.0,20.0,60.0],
                                            sigma_thresh_for_minrad=[0,3,5,10,50,100,200],
                                            res=mapdata.res/arcmin,
                                            deblend_sources=True
                                            )

    if not extracted_sources:
        del mapdata
        continue
    print(len(extracted_sources.keys()),'sources found.')
    noise_red = np.sqrt(mapdata.coadd_days) if mapdata.coadd_days else 1.
    flux_thresh = get_sourceflux_threshold(mapdata.freq)/1000./ noise_red
    catalog_sources = load_act_catalog(args.source_catalog,
                                       flux_threshold = flux_thresh
                                      )

    print('Cross-matching found sources with catalog...')
        
    source_candidates,transient_candidates,noise_candidates = sift(extracted_sources,
                                                                   catalog_sources,
                                                                   map_freq = mapdata.freq,
                                                                  )


    if not args.ignore_known_sources:
         if args.save_json:
                for sc in source_candidates:
                    json_string_cand = sc.json()
                    with open(args.plot_output+sc.sourceID.split(' ')[-1]+'_'+str(mapdata.wafer_name)+'_'+str(mapdata.freq)+'_'+str(int(sc.ctime))+'.json','w') as f:
                        f.write(json_string_cand)

    if len(transient_candidates)<=args.candidate_limit:
        for tc in transient_candidates:
            print(tc.sourceID)
            if args.save_json:
                json_string_cand = tc.json()
                with open(args.plot_output+tc.sourceID.split(' ')[-1]+'_'+str(mapdata.wafer_name)+'_'+str(mapdata.freq)+'_'+str(int(tc.ctime))+'.json','w') as f:
                    f.write(json_string_cand)
            if args.plot_thumbnails:
                plot_map_thumbnail(mapdata.snr,
                                   tc.ra,
                                   tc.dec,
                                   source_name='_'.join(tc.sourceID.split(' '))+'_'+mapdata.wafer_name+'_'+mapdata.freq+'_'+str(tc.ctime),
                                   plot_dir = args.plot_output,
                                   colorbar_range=args.snr_threshold
                                  )
        if len(transient_candidates)>0:
            print('To extract lightcurves for transient candidates, run the following: ')
            executable=['python','slurm_wrapper_depth1_lightcurves.py','--ra']+['%.5f'%tc.ra for tc in transient_candidates]+['--dec']+['%.5f'%tc.dec for tc in transient_candidates]+['--data-dir',args.map_dir, '--save-thumbnails']
            print(' '.join(executable))
    else:
        print('Too many transient candidates (%i)... something fishy'%len(transient_candidates))

    del mapdata,source_candidates,transient_candidates,noise_candidates,extracted_sources,catalog_sources