"""
Testing for use as SO Time domain pipeline test.

"""

import argparse
import warnings
from glob import glob

import numpy as np
import structlog
from pixell.utils import arcmin, degree

from sotrplib.maps.coadding import coadd_map_group, load_coadd_maps
from sotrplib.maps.maps import load_map, preprocess_map
from sotrplib.sifter.core import DefaultSifter
from sotrplib.sifter.crossmatch import crossmatch_position_and_flux
from sotrplib.sims.sim_maps import inject_simulated_sources, make_noise_map
from sotrplib.sims.sim_utils import get_sim_map_group, load_config_yaml
from sotrplib.source_catalog.database import SourceCatalogDatabase
from sotrplib.source_catalog.source_catalog import load_catalog
from sotrplib.sources.finding import extract_sources
from sotrplib.sources.forced_photometry import (
    convert_catalog_to_source_objects,
    photutils_2D_gauss_fit,
)
from sotrplib.utils.plot import plot_map_thumbnail
from sotrplib.utils.utils import get_cut_radius, get_fwhm, get_map_groups

warnings.simplefilter("ignore", RuntimeWarning)
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--maps",
    help="input maps... the map.fits ones.",
    default=[
        "/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/15386/depth1_1538613353_pa5_f150_map.fits"
    ],
    nargs="+",
)
parser.add_argument(
    "--map-dir",
    help="Location of the depth-1 maps.",
    default="/scratch/gpfs/SIMONSOBS/so/maps/actpol/depth1/",
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
    default=20,  # integer
    type=int,
)
parser.add_argument(
    "--solarsystem",
    help="path asteroid ephem directories",
    default="/scratch/gpfs/snaess/actpol/ephemerides/objects",
)
parser.add_argument(
    "--output-dir",
    help="path to output databases and / or plots",
    default="/scratch/gpfs/SIMONSOBS/users/amfoster/scratch/",
)
parser.add_argument(
    "--source-catalog",
    help="path to source catalog",
    default="/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits",
)
parser.add_argument(
    "--galaxy-mask",
    help="path to galaxy mask",
    default="/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/mask_for_sources2019_plus_dust.fits",
)
parser.add_argument("-v", "--verbosity", help="Level of logging", default=1)
parser.add_argument(
    "-s",
    "--snr-threshold",
    help="SNR threshold for source finding",
    default=5.0,
    type=float,
)
parser.add_argument(
    "--flux-threshold",
    help="Flux threshold for catalog matching. in units of Jy.",
    default=0.0,
    type=float,
)
parser.add_argument(
    "--bright-flux-threshold",
    help="Flux threshold for pointing offset calculation. in units of Jy.",
    default=0.3,
    type=float,
)
parser.add_argument(
    "--candidate-limit",
    help="Flag map as bad beyond this number of candidate transients.",
    action="store",
    default=10,
    type=int,
)
parser.add_argument(
    "--coadd-n-days",
    help="Coadd this many days before running source finder. 0 runs every map.",
    action="store",
    default=0,
    type=int,
)
parser.add_argument(
    "--is-coadd",
    help="The input is a coadd map.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--nighttime-only",
    help="Only use maps which were observed between 2300 and 1100 local time.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--plot-all",
    help="Plot various plots to check filtering and source finding.",
    action="store_true",
)
parser.add_argument(
    "--plot-thumbnails",
    help="Plot thumbnails of non-cataloged sources (i.e. transients).",
    action="store_true",
)
parser.add_argument(
    "--save-source-thumbnails",
    help="Save the cataloged source thumbnails. NOT YET IMPLEMENTED",
    action="store_true",
)
parser.add_argument("--verbose", help="Print useful things.", action="store_true")
parser.add_argument(
    "--box",
    default=[],
    nargs="+",
    help="RA and Dec limits for the map in the order of dec_min ra_min dec_max ra_max, in degrees.",
)
parser.add_argument(
    "--million-quasar-catalog",
    help="Path to the million quasar catalog.",
    default="/scratch/gpfs/SIMONSOBS/users/amfoster/scratch/milliquas.fits",
)
## sim parameters
parser.add_argument(
    "--sim",
    help="Run in sim mode; generates sim maps on the fly using config file and injects transients.",
    action="store_true",
)
parser.add_argument(
    "--use-map-geometry",
    help="Use the real map geometry for sims (including time maps).",
    action="store_true",
)
parser.add_argument(
    "--sim-config",
    help="Path to the config file for generating simulated maps.",
    default="sim_config.yaml",
)
parser.add_argument(
    "--simulated-transient-database",
    help="Path to transient event database, generated using the sims module.",
    action="store",
    default="/scratch/gpfs/SIMONSOBS/users/amfoster/scratch/act_depth1_transient_sim_db.db",
)
parser.add_argument(
    "--inject-transients",
    help="Inject transients into maps.",
    action="store_true",
)
parser.add_argument(
    "--overwrite-db",
    help="If not set, the map_id will be searched in the db, if found the observation will be skipped.",
    action="store_true",
)
args = parser.parse_args()

logger = structlog.get_logger()

logger = logger.bind(args=args)
logger.info("pipeline.parsed")


cataloged_sources_db = SourceCatalogDatabase(args.output_dir + "so_source_catalog.csv")
cataloged_sources_db.read_database()
transients_db = SourceCatalogDatabase(args.output_dir + "so_transient_catalog.csv")
noise_candidates_db = SourceCatalogDatabase(args.output_dir + "so_noise_catalog.csv")
injected_source_db = SourceCatalogDatabase(args.output_dir + "injected_sources.csv")


additional_catalogs = {}
if args.million_quasar_catalog and not args.sim:
    from sotrplib.source_catalog.source_catalog import load_million_quasar_catalog

    million_quasar_catalog = load_million_quasar_catalog(args.million_quasar_catalog)
    additional_catalogs["million_quasar"] = million_quasar_catalog


sim_params = {}
logger = logger.bind(sim=args.sim)
if args.sim:
    logger = logger.bind(sim_config=args.sim_config)
    sim_params = load_config_yaml(args.sim_config)
    logger.debug("pipeline.sim.parsed")
    logger = logger.bind(**sim_params)
    logger.debug("pipeline.sim.params")
    if not args.use_map_geometry:
        freq_arr_idx, indexed_map_groups, indexed_map_group_time_ranges = (
            get_sim_map_group(sim_params)
        )

if not args.sim or args.use_map_geometry:
    if len(args.maps) == 1:
        if "*" in args.maps[0]:
            all_maps = glob(args.maps[0])
            if len(all_maps) == 0:
                raise ValueError("No maps found with the given pattern. ", args.maps)
            args.maps = all_maps

    if not args.is_coadd:
        indexed_map_groups, indexed_map_group_time_ranges, time_bins = get_map_groups(
            args.maps,
            coadd_days=args.coadd_n_days,
            restrict_to_night=args.nighttime_only,
        )

    else:
        ## only works for a single input map
        ## should have name [blah]_coadd_[freq]_rho.fits
        freq_arr_idx = args.maps[0].split("_")[-2]
        indexed_map_groups = {freq_arr_idx: [args.maps]}
        indexed_map_group_time_ranges = {freq_arr_idx: [[0, 1]]}
        time_bins = {freq_arr_idx: [[0]]}

## set up the map geometry for loading box region if given.
if args.box:
    if len(args.box) != 4:
        raise ValueError(
            "Box must be given as [dec_min ra_min dec_max ra_max] in degrees."
        )
    box = (
        np.array(
            [
                [float(args.box[0]), float(args.box[1])],
                [float(args.box[2]), float(args.box[3])],
            ]
        )
        * degree
    )
else:
    box = None

logger.info("pipeline.initialized")

for freq_arr_idx in indexed_map_groups:
    logger = logger.new()
    if "_" in freq_arr_idx:
        ## assume it's [arr]_[freq]
        arr, freq = freq_arr_idx.split("_")
    else:
        arr = "all"
        freq = freq_arr_idx
    logger = logger.bind(arr=arr, freq=freq)
    map_groups = indexed_map_groups[freq_arr_idx]
    map_group_time_ranges = indexed_map_group_time_ranges[freq_arr_idx]
    for map_group in map_groups:
        logger = logger.bind(map_group=map_group)
        if len(map_group) == 0:
            logger.warning("pipeline.map.empty_group")
            continue
        if len(map_group) == 1:
            logger.debug("pipeline.map.loading")
            if not args.overwrite_db:
                map_id = "_".join(map_group[0].split("/")[-1].split("_")[-4:-1])
                if cataloged_sources_db.observation_exists(map_id):
                    logger.info("pipeline.map.exists")
                    continue

            ## need a more robust way to check this...
            if "coadd" in str(map_group[0]):
                logger.debug("pipeline.map.load_coadd")
                mapdata = load_coadd_maps(mapfile=map_group[0], arr=arr, box=box)[0]
            else:
                logger.debug("pipeline.map.load_map")
                mapdata = load_map(
                    map_path=map_group[0],
                    map_sim_params=sim_params,
                    box=box,
                    use_map_geometry=args.use_map_geometry,
                    ctime=None,
                    verbose=args.verbose,
                )

        else:
            logger.info("pipeline.map.coadding")
            mapdata = coadd_map_group(map_group, freq=freq, arr=arr, box=box)

        ## end of map loading
        if not mapdata:
            logger.info("pipeline.map.not_found")
            continue

        ## when using the real map geometry, need to load the map first
        ## so here set the simulated map flux and snr
        if args.sim and args.use_map_geometry:
            mapdata.flux = make_noise_map(
                mapdata.time_map,
                map_noise_Jy=sim_params["maps"]["map_noise"],
            )
            mapdata.snr = (
                mapdata.flux / sim_params["maps"]["map_noise"]
                if sim_params["maps"]["map_noise"] > 0
                else np.ones(mapdata.flux.shape)
            )
            del mapdata.rho_map, mapdata.kappa_map

        catalog_sources = []
        cataloged_sources_db.initialize_database()
        map_id = (
            str(int(mapdata.map_ctime))
            + "_"
            + str(mapdata.wafer_name)
            + "_"
            + str(mapdata.freq)
        )
        logger = logger.bind(map_id=map_id)
        band_fwhm = get_fwhm(mapdata.freq, mapdata.wafer_name) * arcmin

        preprocess_map(
            mapdata,
            galmask_file=args.galaxy_mask,
            skip=args.sim,
            tilegrid=0.5,
            sigmaclip=5.0,
            log=logger,
        )

        if not mapdata:
            logger.info("pipeline.preprocess.bad_map")
            continue

        catalog_sources, injected_sources = inject_simulated_sources(
            mapdata,
            injected_source_db,
            sim_params,
            map_id,
            verbose=args.verbose,
            inject_transients=args.inject_transients,
            use_map_geometry=args.use_map_geometry,
            simulated_transient_database=args.simulated_transient_database,
            log=logger,
        )

        logger = logger.bind(
            source_catalog=args.source_catalog, flux_threshold=args.flux_threshold
        )

        if not args.sim:
            logger.info("pipeline.sources.load_catalog")
            catalog_sources += load_catalog(
                args.source_catalog,
                flux_threshold=args.flux_threshold,
                mask_outside_map=True,
                mask_map=mapdata.flux,
                return_source_cand_list=True,
                log=logger,
            )

        logger.info("pipeline.sources.fitting")
        catalog_sources, thumbs = photutils_2D_gauss_fit(
            mapdata.flux,
            mapdata.flux / mapdata.snr
            if not args.sim
            else mapdata.flux / sim_params["maps"]["map_noise"],
            catalog_sources,
            size_deg=band_fwhm / degree,
            PLOT=args.plot_all,
            reproject_thumb=not args.sim,
            flux_lim_fit_centroid=args.bright_flux_threshold,
            return_thumbnails=args.save_source_thumbnails,
            log=logger,
        )

        ## Do something with the thumbnails?
        logger.warning("Not doing anything with the thumbnails")
        del thumbs

        known_sources = []
        if catalog_sources:
            known_sources = convert_catalog_to_source_objects(
                catalog_sources,
                mapdata.freq,
                mapdata.wafer_name,
                map_id=map_id,
                ctime=mapdata.time_map,
                log=logger,
            )

            ## update the source catalog
            cataloged_sources_db.add_sources(known_sources)

            ## subtract the detected (known) sources from the map
            ## ignore sources with wonky fits.
            srcmodel = mapdata.subtract_sources(
                known_sources,
                cuts={
                    "flux": [0.0, np.inf],
                    "err_flux": [-1.0, 1.0],
                    "fwhm_a": [0.25, 6.0],
                    "fwhm_b": [0.25, 6.0],
                },
                log=logger,
            )

        logger.info("pipeline.sources.search")
        ## get mask radius for each sigma such that the mask goes to the
        ## noise floor.
        sigma_thresh_for_minrad = np.arange(100)
        pix_rad = get_cut_radius(
            mapdata.res / arcmin,
            mapdata.wafer_name,
            mapdata.freq,
            matched_filtered=True,
            source_amplitudes=sigma_thresh_for_minrad,
            map_noise=np.ones(len(sigma_thresh_for_minrad)),
            max_radius_arcmin=120.0,
        )
        logger.info(
            "pipeline.sources.cut_radii_set",
            sigma_thresh_for_minrad=sigma_thresh_for_minrad,
            pix_rad=pix_rad,
        )

        extracted_sources = extract_sources(
            mapdata.flux,
            timemap=mapdata.time_map,
            maprms=abs(mapdata.flux / mapdata.snr)
            if not args.sim
            else sim_params["maps"]["map_noise"] * np.ones(mapdata.flux.shape),
            nsigma=args.snr_threshold,
            minrad=pix_rad / (mapdata.res / arcmin),
            sigma_thresh_for_minrad=sigma_thresh_for_minrad,
            res=mapdata.res / arcmin,
            log=logger,
        )

        logger = logger.bind(n_sources=len(extracted_sources))
        logger.info("pipeline.sources.extracted")

        ## need to figure out what distribution "good" sources have to set the cuts.
        default_cuts = {
            "fwhm": [0.5, 2 * band_fwhm / arcmin],
            "ellipticity": [-1, 1],
            "elongation": [-1, 2],
            "fwhm_a": [0.5 * band_fwhm / arcmin, 2 * band_fwhm / arcmin],
            "fwhm_b": [0.5 * band_fwhm / arcmin, 2 * band_fwhm / arcmin],
            "snr": [args.snr_threshold, np.inf],
        }

        logger = logger.bind(map_id=map_id, fit_cuts=default_cuts)
        logger.info("pipeline.sift")

        sifter = DefaultSifter(
            radius_1Jy=1.5 if args.sim else 2 * band_fwhm / arcmin,
            cuts=default_cuts,
            crossmatch_with_gaia=not args.sim,
            crossmatch_with_million_quasar=not args.sim,
            additional_catalogs=additional_catalogs,
            log=logger,
            debug=args.verbose,
        )

        sifter_result = sifter.sift(
            extracted_sources=extracted_sources,
            catalog_sources=catalog_sources,
            flux_map=mapdata.flux,
            map_id=map_id,
            map_freq=mapdata.freq,
            arr=mapdata.wafer_name,
        )

        catalog_matches = sifter_result.source_candidates
        transient_candidates = sifter_result.transient_candidates
        noise_candidates = sifter_result.noise_candidates

        if args.sim:
            ## make sure the simulated sources are properly masked, and the injected transients are recovered
            ## including with the correct fluxes.
            if len(catalog_matches) > 0:
                raise ValueError(
                    "Catalog matches found in simulated map... something is wrong."
                )
            ## crossmatch the recovered sources with the injected sources
            _, _ = crossmatch_position_and_flux(
                injected_sources,
                transient_candidates,
                spatial_threshold=band_fwhm / degree,
                flux_threshold=5 * sim_params["maps"]["map_noise"],
                fail_flux_mismatch=True,
                fail_unmatched=True,
            )

        logger = logger.bind(n_catalog_matches=len(catalog_matches))
        if len(catalog_matches) > 0:
            if args.plot_thumbnails:
                for cm in catalog_matches:
                    logger = logger.bind(
                        match=cm, source_name="_".join(cm.sourceID.split(" "))
                    )
                    logger.info("pipeline.sift.blind_in_catalog")
                    plot_map_thumbnail(
                        mapdata.snr
                        if not args.sim
                        else mapdata.flux / sim_params["maps"]["map_noise"],
                        cm.ra,
                        cm.dec,
                        source_name="_".join(cm.sourceID.split(" "))
                        + "_"
                        + str(cm.ctime)
                        + "_"
                        + mapdata.wafer_name
                        + "_"
                        + mapdata.freq,
                        plot_dir=args.output_dir,
                        colorbar_range=args.snr_threshold,
                    )
            ## do we do anything here? add to updated source catalog?
            ## should only be possible if the masking radius and the source matching radius are significantly different.

        if len(transient_candidates) <= (args.candidate_limit + len(injected_sources)):
            transients_db.add_sources(transient_candidates)
            ## add transients to the source catalog as well
            cataloged_sources_db.add_sources(transient_candidates)
            if args.plot_thumbnails:
                for tc in transient_candidates:
                    plot_map_thumbnail(
                        mapdata.snr
                        if not args.sim
                        else mapdata.flux / sim_params["maps"]["map_noise"],
                        tc.ra,
                        tc.dec,
                        source_name="_".join(tc.sourceID.split(" "))
                        + "_"
                        + str(tc.ctime)
                        + "_"
                        + mapdata.wafer_name
                        + "_"
                        + mapdata.freq,
                        plot_dir=args.output_dir,
                        colorbar_range=args.snr_threshold,
                    )
            if args.verbose:
                if len(transient_candidates) > 0:
                    print(
                        "To extract lightcurves for transient candidates, run the following: "
                    )
                    executable = (
                        ["python", "slurm_wrapper_depth1_lightcurves.py", "--ra"]
                        + ["%.5f" % tc.ra for tc in transient_candidates]
                        + ["--dec"]
                        + ["%.5f" % tc.dec for tc in transient_candidates]
                        + ["--data-dir", args.map_dir, "--save-thumbnails"]
                    )
                    print(" ".join(executable))
        else:
            print(
                "Too many transient candidates (%i)... something fishy"
                % len(transient_candidates)
            )
            print("Writing to noise candidates database.")
            noise_candidates_db.add_sources(transient_candidates)

        noise_candidates_db.add_sources(noise_candidates)

        del (
            mapdata,
            catalog_matches,
            transient_candidates,
            noise_candidates,
            extracted_sources,
            catalog_sources,
            known_sources,
            injected_sources,
        )

        ## update databases and clear
        cataloged_sources_db.write_database()
        cataloged_sources_db.read_database()
        noise_candidates_db.write_database()
        noise_candidates_db.initialize_database()
        transients_db.write_database()
        transients_db.initialize_database()
        injected_source_db.write_database()
        injected_source_db.initialize_database()
