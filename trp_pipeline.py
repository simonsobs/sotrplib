"""
Testing for use as SO Time domain pipeline test.

"""
import numpy as np
from glob import glob 
import warnings
warnings.simplefilter("ignore",RuntimeWarning)

from sotrplib.maps.maps import load_maps, preprocess_map, load_sim_map
from sotrplib.maps.masks import make_src_mask
from sotrplib.maps.coadding import coadd_map_group,load_coadd_maps
from sotrplib.utils.utils import get_map_groups, get_cut_radius
from sotrplib.sources.finding import extract_sources
from sotrplib.sources.forced_photometry import photutils_2D_gauss_fit,convert_catalog_to_source_objects
from sotrplib.source_catalog.source_catalog import load_catalog,convert_gauss_fit_to_source_cat
from sotrplib.sifter.crossmatch import sift
from sotrplib.utils.plot import plot_map_thumbnail
from sotrplib.utils.utils import get_fwhm
from sotrplib.source_catalog.database import SourceCatalogDatabase
from sotrplib.sims.sim_utils import load_transients_from_db,load_config_yaml
from sotrplib.sims.sim_maps import inject_sources,photutils_sim_n_sources
from sotrplib.sims.sim_sources import generate_transients


from pixell.utils import arcmin,degree

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
                    default=20, # integer
                    type=int
                    )
parser.add_argument("--solarsystem",
                    help="path asteroid ephem directories",
                    default="/scratch/gpfs/snaess/actpol/ephemerides/objects",
                    )
parser.add_argument("--output-dir",
                    help="path to output databases and / or plots",
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
                    help="Flux threshold for catalog matching. in units of Jy.", 
                    default=0.0,
                    type=float,
                    )
parser.add_argument("--bright-flux-threshold", 
                    help="Flux threshold for pointing offset calculation. in units of Jy.", 
                    default=0.3,
                    type=float,
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
parser.add_argument("--save-source-thumbnails", 
                    help="Save the cataloged source thumbnails. NOT YET IMPLEMENTED", 
                    action='store_true'
                    )
parser.add_argument("--verbose", 
                    help="Print useful things.", 
                    action='store_true'
                    )
## sim parameters
parser.add_argument("--sim", 
                    help="Run in sim mode; generates sim maps on the fly using config file and injects transients.", 
                    action='store_true',
                    )
parser.add_argument("--use-map-geometry", 
                    help="Use the real map geometry for sims (including time maps).", 
                    action='store_true',
                    )
parser.add_argument("--sim-config",
                    help="Path to the config file for generating simulated maps.", 
                    default='sim_config.yaml',
                    )
parser.add_argument("--simulated-transient-database", 
                    help="Path to transient event database, generated using the sims module.", 
                    action='store',
                    default='/scratch/gpfs/SIMONSOBS/users/amfoster/scratch/act_depth1_transient_sim_db.db',
                    )
parser.add_argument("--inject-transients", 
                    help="Inject transients into maps.", 
                    action='store_true',
                    )
parser.add_argument("--overwrite-db", 
                    help="If not set, the mapid will be searched in the db, if found the observation will be skipped.", 
                    action='store_true'
                    )
args = parser.parse_args()


cataloged_sources_db = SourceCatalogDatabase(args.output_dir+'so_source_catalog.csv')
cataloged_sources_db.read_database()
transients_db = SourceCatalogDatabase(args.output_dir+'so_transient_catalog.csv')
noise_candidates_db = SourceCatalogDatabase(args.output_dir+'so_noise_catalog.csv')
injected_source_db = SourceCatalogDatabase(args.output_dir+'injected_sources.csv') 


if args.sim:
    if args.verbose:
         print('Loading sim config file: ',args.sim_config)
    sim_params = load_config_yaml(args.sim_config)
    if args.verbose:
        print('Sim params: ')
        for k,v in sim_params.items():
            print(k,v)
    if not args.use_map_geometry:
        freq_arr_idx = sim_params['array_info']['arr']+'_'+sim_params['array_info']['freq']
        indexed_map_groups = {freq_arr_idx:[[i] for i in np.arange(sim_params['maps']['n_realizations'])]}
        indexed_map_group_time_ranges = {freq_arr_idx:[[t] for t in np.linspace(float(sim_params['maps']['min_time']),float(sim_params['maps']['max_time']),int(sim_params['maps']['n_realizations']))]}

if not args.sim or args.use_map_geometry:
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
        if args.sim and not args.use_map_geometry:
            if args.verbose:
                print('Loading simulated map group: ',map_group[0])
            mapdata = load_sim_map(sim_params,
                                   ctime=map_group_time_ranges[map_group[0]][0]
                                   )    
            if args.verbose:
                print('Simulated map time: ',mapdata.map_ctime)
        elif len(map_group)==1:
            if args.verbose:
                print('Loading map ',map_group[0])
            if not args.overwrite_db:
                map_id = '_'.join(map_group[0].split('/')[-1].split('_')[-4:-1])
                if cataloged_sources_db.observation_exists(map_id):
                    print('Map already exists in db... skipping.')
                    continue
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
        if not mapdata:
            print('No map data found... skipping.')
            continue    
        catalog_sources = []
        cataloged_sources_db._initialize_database()
        map_id = str(int(mapdata.map_ctime))+'_'+str(mapdata.wafer_name)+'_'+str(mapdata.freq)
        t0 = mapdata.map_start_time if mapdata.map_start_time else 0.0
        band_fwhm = get_fwhm(mapdata.freq,mapdata.wafer_name)*arcmin
        
        if not args.sim:
            preprocess_map(mapdata,
                            galmask_file = args.galaxy_mask,
                            plot_output_dir=args.output_dir,
                            PLOT=args.plot_all,
                            tilegrid=float(args.gridsize),
                            )
        else:
            if args.verbose:
                print('Injecting Sources into simulated map.')
            from pixell import enmap
            if args.use_map_geometry:
                ## use the map geometry for the simulated map
                mapdata.flux = enmap.zeros(mapdata.time_map.shape,wcs=mapdata.time_map.wcs)
                del mapdata.rho_map,mapdata.kappa_map
            ## inject static sources    
            if args.verbose:
                print('Injecting static sources into simulated map using sim param config.')
            mapdata.flux,catalog_sources = photutils_sim_n_sources(mapdata.flux,
                                                                    sim_params['injected_sources']['n_sources'],
                                                                    min_flux_Jy=sim_params['injected_sources']['min_flux'],
                                                                    max_flux_Jy=sim_params['injected_sources']['max_flux'],
                                                                    map_noise_Jy=sim_params['maps']['map_noise'],
                                                                    fwhm_uncert_frac=sim_params['injected_sources']['fwhm_uncert_frac'],
                                                                    gauss_fwhm_arcmin=band_fwhm/arcmin,
                                                                    )
            mapdata.snr = mapdata.flux/sim_params['maps']['map_noise']
            
            injected_source_db.add_sources(convert_catalog_to_source_objects(catalog_sources,
                                                                             mapdata.freq,
                                                                             mapdata.wafer_name,
                                                                             mapid=map_id,
                                                                             ctime=mapdata.time_map+t0,
                                                                            )
                                          )
            injected_source_db.write_database()
            
        injected_sources = []
        if args.inject_transients:
            if args.simulated_transient_database:
                if args.verbose:
                    print('Loading simulated transients from database.')
            
                transient_sources = load_transients_from_db(args.simulated_transient_database)
                mapdata.flux,injected_sources = inject_sources(mapdata.flux,
                                                            transient_sources,
                                                            mapdata.time_map+t0,
                                                            freq=mapdata.freq,
                                                            arr=mapdata.wafer_name,
                                                            debug=args.verbose,
                                                            )
                injected_source_db.add_sources(injected_sources)
                injected_source_db.write_database()

            if args.sim:
                if args.verbose:
                    print('Generating transients using sim config.')
                
                if args.use_map_geometry:
                    corners = mapdata.flux.corners()/degree
                    ra_lims = (corners[0][1]%360,corners[1][1]%360)
                    dec_lims = (corners[0][0],corners[1][0])
                else:
                    ra_lims = (sim_params['maps']['center_ra']-sim_params['maps']['width_ra'],
                            sim_params['maps']['center_ra']+sim_params['maps']['width_ra']
                            )
                    dec_lims = (sim_params['maps']['center_dec']-sim_params['maps']['width_dec'],
                                sim_params['maps']['center_dec']+sim_params['maps']['width_dec']
                            )
                    
                transients_to_inject = generate_transients(n=sim_params['injected_transients']['n_transients'],
                                                          ra_lims=ra_lims,
                                                          dec_lims=dec_lims,
                                                          peak_amplitudes=(sim_params['injected_transients']['min_flux'],sim_params['injected_transients']['max_flux']),
                                                          peak_times=(mapdata.map_ctime,mapdata.map_ctime),
                                                          flare_widths=(sim_params['injected_transients']['min_width'],sim_params['injected_transients']['max_width']),
                                                         )
                mapdata.flux,injected_sources = inject_sources(mapdata.flux,
                                                            transients_to_inject,
                                                            mapdata.time_map+t0,
                                                            freq=mapdata.freq,
                                                            arr=mapdata.wafer_name,
                                                            debug=args.verbose,
                                                            )
                if args.verbose:
                    print(f'Injected {len(injected_sources)} transients into simulated map')
        
        from matplotlib import pyplot as plt
        plt.figure(figsize=(20, 10), dpi=200)
        ax = plt.subplot(projection=mapdata.flux.wcs)
        im = ax.imshow(mapdata.flux, vmin=-0.05, vmax=1)
        ax.scatter(catalog_sources['RADeg'], catalog_sources['decDeg'], transform=ax.get_transform('icrs'),
                        marker='o', s=100, facecolors='none', edgecolors='cyan')
        for t in injected_sources:
            if t.flux>5*sim_params['maps']['map_noise']:
                ax.scatter(t.ra, t.dec, transform=ax.get_transform('icrs'),
                        marker='d', s=100, facecolors='none', edgecolors='r')
            else:
                ax.scatter(t.ra, t.dec, transform=ax.get_transform('icrs'),
                        marker='o', s=10, facecolors='none', edgecolors='r')
        plt.colorbar(im, ax=ax)
        plt.savefig('/scratch/gpfs/SIMONSOBS/users/amfoster/scratch/sim_map_%s.png'%map_id)
        plt.close()
        
        if not args.sim:
            print('Loading Source Catalog')
            catalog_sources = load_catalog(args.source_catalog,
                                        flux_threshold = args.flux_threshold,
                                        mask_outside_map=True,
                                        mask_map=mapdata.flux
                                        )

        gauss_fits,thumbs = photutils_2D_gauss_fit(mapdata.flux,
                                                   mapdata.flux/mapdata.snr,
                                                   catalog_sources,
                                                   size_deg=band_fwhm/degree,
                                                   PLOT=args.plot_all,
                                                   reproject_thumb= not args.sim,
                                                   flux_lim_fit_centroid=args.bright_flux_threshold,
                                                   return_thumbnails=args.save_source_thumbnails
                                                  )
        
        catalog_sources = convert_gauss_fit_to_source_cat(gauss_fits)
        del gauss_fits,thumbs

        known_sources = []
        if catalog_sources:
            known_sources = convert_catalog_to_source_objects(catalog_sources,
                                                            mapdata.freq,
                                                            mapdata.wafer_name,
                                                            mapid=map_id,
                                                            ctime=mapdata.time_map+t0,
                                                            )
            
            ## update the source catalog
            cataloged_sources_db.add_sources(known_sources)
                
            ## get source mask radius based on flux.
            source_mask_radius = get_cut_radius(mapdata.res/arcmin,
                                                mapdata.wafer_name,
                                                mapdata.freq,
                                                source_amplitudes=catalog_sources['fluxJy'],
                                                map_noise=catalog_sources['err_fluxJy'],
                                                max_radius_arcmin=120.0,
                                                matched_filtered=True
                                                )

            ## simple catalog mask
            catalog_mask = make_src_mask(mapdata.flux,
                                        catalog_sources['pix'],
                                        arr=mapdata.wafer_name,
                                        freq=mapdata.freq,
                                        mask_radius=source_mask_radius
                                        )

            ## mask the flux and snr after extracting via forced photometry
            mapdata.flux*=catalog_mask
            mapdata.snr*=catalog_mask
            del catalog_mask
        
        
        print('Finding sources...')
        ## get mask radius for each sigma such that the mask goes to the 
        ## noise floor.
        sigma_thresh_for_minrad = np.arange(100)
        pix_rad = get_cut_radius(mapdata.res/arcmin,
                                 mapdata.wafer_name,
                                 mapdata.freq,
                                 matched_filtered=True,
                                 source_amplitudes=sigma_thresh_for_minrad,
                                 map_noise=np.ones(len(sigma_thresh_for_minrad)),
                                 max_radius_arcmin=120.
                                )
        
        extracted_sources = extract_sources(mapdata.flux,
                                            timemap=mapdata.time_map+t0,
                                            maprms=mapdata.flux/mapdata.snr if not args.sim else sim_params['maps']['map_noise']*np.ones(mapdata.flux.shape),
                                            nsigma=args.snr_threshold,
                                            minrad=pix_rad/(mapdata.res/arcmin),
                                            sigma_thresh_for_minrad=sigma_thresh_for_minrad,
                                            res=mapdata.res/arcmin,
                                            )
        
        if not extracted_sources:
            ## no sources were extracted from the map.
            del mapdata,known_sources
            continue
        
        print(len(extracted_sources.keys()),'sources found.')

        print('Cross-matching found sources with catalog...')
        ## need to figure out what distribution "good" sources have to set the cuts.
        default_cuts = {'fwhm':[0.5,2*band_fwhm/arcmin],
                        'ellipticity':[-1,1],
                        'elongation': [-1,2],
                        'fwhm_a': [0.5*band_fwhm/arcmin,2*band_fwhm/arcmin],
                        'fwhm_b': [0.5*band_fwhm/arcmin,2*band_fwhm/arcmin],
                        'snr': [args.snr_threshold,np.inf],
                        }
        source_candidates,transient_candidates,noise_candidates = sift(extracted_sources,
                                                                       catalog_sources,
                                                                       imap=mapdata.flux,
                                                                       map_freq = mapdata.freq,
                                                                       arr=mapdata.wafer_name,
                                                                       mapid=map_id,
                                                                       cuts=default_cuts,
                                                                       #debug=args.verbose
                                                                      )
        
        if len(source_candidates)>0:
            print('Source found in blind search which is associated with catalog... masking may have a problem.')
            ## do we do anything here? add to updated source catalog?
            ## should only be possible if the masking radius and the source matching radius are significantly different.

        if len(transient_candidates)<=args.candidate_limit:
            transients_db.add_sources(transient_candidates)
            ## add transients to the source catalog as well
            cataloged_sources_db.add_sources(transient_candidates)
            if args.plot_thumbnails:
                for tc in transient_candidates:

                    plot_map_thumbnail(mapdata.snr if not args.sim else mapdata.flux/sim_params['maps']['map_noise'],
                                        tc.ra,
                                        tc.dec,
                                        source_name='_'.join(tc.sourceID.split(' '))+'_'+str(tc.ctime)+'_'+mapdata.wafer_name+'_'+mapdata.freq,
                                        plot_dir = args.output_dir,
                                        colorbar_range=args.snr_threshold
                                        )
            if args.verbose:
                if len(transient_candidates)>0:
                    print('To extract lightcurves for transient candidates, run the following: ')
                    executable=['python','slurm_wrapper_depth1_lightcurves.py','--ra']+['%.5f'%tc.ra for tc in transient_candidates]+['--dec']+['%.5f'%tc.dec for tc in transient_candidates]+['--data-dir',args.map_dir, '--save-thumbnails']
                    print(' '.join(executable))
        else:
            print('Too many transient candidates (%i)... something fishy'%len(transient_candidates))
            print('Writing to noise candidates database.')
            noise_candidates_db.add_sources(transient_candidates)

        noise_candidates_db.add_sources(noise_candidates)

        del mapdata,source_candidates,transient_candidates,noise_candidates,extracted_sources,catalog_sources,known_sources,injected_sources

        ## update databases and clear
        cataloged_sources_db.write_database()
        cataloged_sources_db.read_database()
        noise_candidates_db.write_database()
        noise_candidates_db._initialize_database()
        transients_db.write_database()
        transients_db._initialize_database()
        injected_source_db._initialize_database()