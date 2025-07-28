import numpy as np
from pixell import enmap

from ..maps.maps import Depth1Map
from ..source_catalog.database import SourceCatalogDatabase


def make_enmap(
    center_ra: float = 0.0,
    center_dec: float = 0.0,
    width_ra: float = 1.0,
    width_dec: float = 1.0,
    resolution: float = 0.5,
    map_noise: float = None,
):
    """ """
    from pixell.enmap import geometry, zeros
    from pixell.utils import arcmin, degree

    if (width_ra <= 0) or (width_dec <= 0):
        raise ValueError("Map width must be positive")

    min_dec = center_dec - width_dec
    max_dec = center_dec + width_dec
    if min_dec < -90:
        raise ValueError("Minimum declination, %.1f, is below -90" % min_dec)
    if max_dec > 90:
        raise ValueError("Maximum declination, %.1f, is above 90" % max_dec)

    min_ra = center_ra - width_ra
    max_ra = center_ra + width_ra
    ## need to do something with ra limits unless geometry knows how to do the wrapping.

    box = np.array([[min_dec, min_ra], [max_dec, max_ra]])
    shape, wcs = geometry(box * degree, res=resolution * arcmin)
    if map_noise:
        return make_noise_map(
            zeros(shape, wcs=wcs), map_noise_Jy=map_noise, map_mean_Jy=0.0, seed=None
        )
    else:
        return zeros(shape, wcs=wcs)


def make_noise_map(
    imap: enmap.ndmap,
    map_noise_Jy: float = 0.01,
    map_mean_Jy: float = 0.0,
    seed: int = None,
):
    """
    Create a noise map with the same shape and WCS as the input map.

    Parameters:
    - imap: enmap.ndmap, the input map to create a noise map for.
    - map_noise_Jy: float, the standard deviation of the noise in Jy.
    - map_mean_Jy: float, the mean of the noise in Jy.
    - seed: int, random seed for reproducibility.

    Returns:
    - noise_map: enmap.ndmap, the generated noise map.
    """
    from photutils.datasets import make_noise_image

    from ..maps.maps import edge_map

    shape = imap.shape
    wcs = imap.wcs
    noise_map = enmap.zeros(shape, wcs=wcs)
    if not seed:
        # print('No seed provided, setting seed with current ctime')
        seed = np.random.seed()
    mask = edge_map(imap)
    if np.all(mask == 0):
        mask = np.ones(
            shape
        )  # If the mask is all zeros, invert it to create a full mask.
    noise_map += make_noise_image(
        shape, distribution="gaussian", mean=map_mean_Jy, stddev=map_noise_Jy, seed=seed
    )
    noise_map *= mask
    return noise_map


def photutils_sim_n_sources(
    sim_map: enmap.ndmap | Depth1Map,
    n_sources: float,
    min_flux_Jy: float = 0.1,
    max_flux_Jy: float = 2.0,
    map_noise_Jy: float = 0.01,
    gauss_fwhm_arcmin: float = 2.2,
    fwhm_uncert_frac: float = 0.01,
    gauss_theta_min: float = 0,
    gauss_theta_max: float = 90,
    min_sep_arcmin: float = 5,
    seed: int = None,
    freq: str = "f090",
    arr: str = "sim",
    map_id: str = None,
    ctime: float | enmap.ndmap = None,
):
    """ """
    from photutils.psf import GaussianPSF, make_psf_model_image
    from pixell.utils import arcmin, degree

    from ..sources.forced_photometry import convert_catalog_to_source_objects
    from .sim_utils import convert_photutils_qtable_to_json

    model = GaussianPSF()
    if isinstance(sim_map, Depth1Map):
        shape = sim_map.flux.shape
        wcs = sim_map.flux.wcs
    elif isinstance(sim_map, enmap.ndmap):
        shape = sim_map.shape
        wcs = sim_map.wcs
    else:
        raise ValueError("Input map must be a Depth1Map or enmap.ndmap object")

    mapres = abs(wcs.wcs.cdelt[0]) * degree
    gaussfwhm = gauss_fwhm_arcmin * arcmin
    # Omega_b is the beam solid angle for a Gaussian beam in sr
    ## I've found that the recovered flux is low by 2% ish...
    omega_b = 1.02 * (np.pi / 4 / np.log(2)) * (gaussfwhm) ** 2

    gaussfwhm /= mapres
    minfwhm = gaussfwhm - (fwhm_uncert_frac * gaussfwhm)
    maxfwhm = gaussfwhm + (fwhm_uncert_frac * gaussfwhm)

    minsep = min_sep_arcmin * arcmin / mapres

    if not seed:
        # print('No seed provided, setting seed with current ctime')
        seed = np.random.seed()

    if min_flux_Jy >= max_flux_Jy:
        raise ValueError("Min injected flux is less than or equal to max injected flux")

    ## set the window for each source to be 5x fwhm
    model_window = (int(5 * gaussfwhm), int(5 * gaussfwhm))
    ## make the source map and add to input map
    data, params = make_psf_model_image(
        shape,
        model,
        n_sources,
        model_shape=model_window,
        min_separation=minsep,
        border_size=int(minsep),
        flux=(min_flux_Jy, max_flux_Jy),
        x_fwhm=(minfwhm, maxfwhm),
        y_fwhm=(minfwhm, maxfwhm),
        theta=(gauss_theta_min, gauss_theta_max),
        seed=seed,
        normalize=False,
    )
    ## photutils distributes the flux over the solid angle
    ## so this map should be the intensity map in Jy/sr
    ## multiply by beam area in sq pixels
    simflux = data * (omega_b / mapres**2)
    if isinstance(sim_map, Depth1Map):
        sim_map.snr += simflux / (sim_map.flux / sim_map.snr)
        sim_map.flux += simflux

    elif isinstance(sim_map, enmap.ndmap):
        sim_map += simflux

    ## add noise to the map
    if map_noise_Jy > 0:
        if isinstance(sim_map, Depth1Map):
            sim_map.flux += make_noise_map(
                sim_map.flux, map_noise_Jy=map_noise_Jy, map_mean_Jy=0.0, seed=seed
            )
        elif isinstance(sim_map, enmap.ndmap):
            sim_map += make_noise_map(
                sim_map, map_noise_Jy=map_noise_Jy, map_mean_Jy=0.0, seed=seed
            )

    params["x_fwhm"] = params["x_fwhm"] * mapres / arcmin
    params["y_fwhm"] = params["y_fwhm"] * mapres / arcmin

    ## photutils parameter table is in Astropy QTable
    ## convert to the json format we've been using.
    injected_sources = convert_photutils_qtable_to_json(
        params,
        imap=sim_map if isinstance(sim_map, enmap.ndmap) else sim_map.flux,
    )
    injected_sources = convert_catalog_to_source_objects(
        injected_sources,
        freq=freq,
        arr=arr,
        ctime=ctime,
        map_id=map_id if map_id else "",
        source_type="simulated",
    )

    return sim_map, injected_sources


def inject_sources(
    imap: enmap.ndmap | Depth1Map,
    sources: list,
    observation_time: float | enmap.ndmap,
    freq: str = "f090",
    arr: str = None,
    map_id: str = None,
    debug: bool = False,
):
    """
    Inject a list of SimTransient objects into the input map.

    Parameters:
    - imap: enmap.ndmap, the input map to inject sources into.
    - sources: list of SimTransient objects.
    - observation_time: float or enmap.ndmap, the time of the observation.
    - freq: str, the frequency of the observation.
    - arr: str, the array name (optional).
    - map_id: str, the map ID (optional).
    - debug: bool, if True, print debug information including tqdm.

    Returns:
    - imap: enmap.ndmap, the map with sources injected.
    - injected_sources: list of SourceCandidate objects representing the injected sources.
    """
    from photutils.datasets import make_model_image
    from photutils.psf import GaussianPSF
    from pixell.enmap import sky2pix
    from pixell.utils import arcmin, degree
    from tqdm import tqdm

    from ..sources.sources import SourceCandidate
    from ..utils.utils import get_fwhm
    from .sim_utils import make_2d_gaussian_model_param_table

    if not sources:
        return imap, []

    if isinstance(imap, Depth1Map):
        shape = imap.flux.shape
        wcs = imap.flux.wcs
    elif isinstance(imap, enmap.ndmap):
        shape = imap.shape
        wcs = imap.wcs
    else:
        raise ValueError("Input map must be a Depth1Map or enmap.ndmap object")

    mapres = abs(wcs.wcs.cdelt[0]) * degree
    fwhm = get_fwhm(freq, arr=arr) * arcmin
    fwhm_pixels = fwhm / mapres

    injected_sources = []
    removed_sources = {"not_flaring": 0, "out_of_bounds": 0}
    for source in tqdm(sources, desc="Injecting sources", disable=not debug):
        # Check if source is within the map bounds
        ra, dec = source.ra, source.dec
        pix = sky2pix(shape, wcs, np.asarray([dec, ra]) * degree)
        if not (0 <= pix[0] < shape[-2] and 0 <= pix[1] < shape[-1]):
            removed_sources["out_of_bounds"] += 1
            continue
        if isinstance(imap, Depth1Map):
            mapval = imap.flux[int(pix[0]), int(pix[1])]
        else:
            mapval = imap[int(pix[0]), int(pix[1])]

        if not np.isfinite(mapval) or abs(mapval) == 0:
            removed_sources["out_of_bounds"] += 1
            continue

        # Check if the source is flaring at the observation time
        if isinstance(observation_time, float):
            source_obs_time = observation_time
        else:
            source_obs_time = observation_time[int(pix[0]), int(pix[1])]

        if abs(source.peak_time - source_obs_time) > 3 * source.flare_width:
            removed_sources["not_flaring"] += 1
            continue

        flux = source.get_flux(source_obs_time)

        inj_source = SourceCandidate(
            ra=ra,
            err_ra=0.0,
            dec=dec,
            err_dec=0.0,
            flux=flux,
            err_flux=0.0,
            snr=np.inf,
            freq=freq,
            arr=arr if arr else "any",
            ctime=source_obs_time,
            fwhm_a=fwhm / arcmin,
            fwhm_b=fwhm / arcmin,
            source_type="simulated",
            orientation=0.0,
            map_id=map_id if map_id else "",
        )
        injected_sources.append(inj_source)

    model_params = make_2d_gaussian_model_param_table(
        imap if isinstance(imap, enmap.ndmap) else imap.flux,
        sources=injected_sources,
        nominal_fwhm_arcmin=fwhm / arcmin,
        cuts={},
        verbose=debug,
    )

    psf_image = make_model_image(
        shape,
        GaussianPSF(),
        model_params,
        model_shape=(int(5 * fwhm_pixels), int(5 * fwhm_pixels)),
    )

    if isinstance(imap, Depth1Map):
        imap.snr += psf_image / (imap.flux / imap.snr)
        imap.flux += psf_image
    elif isinstance(imap, enmap.ndmap):
        imap += psf_image

    if debug:
        print(f"Removed sources: {removed_sources}")
        print(f"Injected sources: {len(injected_sources)}")

    return imap, injected_sources


def inject_sources_from_db(
    mapdata: Depth1Map,
    injected_source_db: SourceCatalogDatabase,
    t0: float = 0.0,
    verbose=False,
):
    ## inject static sources
    if verbose:
        print("Injecting static sources into simulated map using sim param config.")
    catalog_sources = injected_source_db.read_database()
    mapdata, injected_sources = inject_sources(
        mapdata,
        catalog_sources,
        mapdata.time_map + t0,
        freq=mapdata.freq,
        arr=mapdata.wafer_name,
        map_id=mapdata.map_id,
        debug=verbose,
    )
    return injected_sources


def inject_random_sources(
    mapdata,
    sim_params: dict,
    map_id: str,
    t0: float = 0.0,
    fwhm_arcmin: float = 2.2,
    add_noise: bool = False,
):
    """
    Inject sources into a map using photutils.
    Add injected sources to the injected_source_db.
    Parameters:
    - mapdata: Depth1Map object containing the map to inject sources into.
    - injected_source_db: Database object to store injected sources.
    - sim_params: Dictionary containing simulation parameters.
    - map_id: Unique identifier for the map.
    - t0: Time offset for the map.
    - fwhm_arcmin: Bandwidth full width at half maximum (FWHM) in arcminutes.
    - return_source_object_list: bool, if True, return the list of injected source objects.
        otherwise, return a source catalog dict.
    - add_noise: bool, if True, add sim_params['maps']['map_noise'] to the map.
        Assumes this is already injected since sim maps load this by default.
    """
    if not sim_params:
        return []
    mapdata, injected_sources = photutils_sim_n_sources(
        mapdata,
        sim_params["injected_sources"]["n_sources"],
        min_flux_Jy=sim_params["injected_sources"]["min_flux"],
        max_flux_Jy=sim_params["injected_sources"]["max_flux"],
        map_noise_Jy=sim_params["maps"]["map_noise"] if add_noise else 0.0,
        fwhm_uncert_frac=sim_params["injected_sources"]["fwhm_uncert_frac"],
        gauss_fwhm_arcmin=fwhm_arcmin,
        freq=mapdata.freq,
        arr=mapdata.wafer_name,
        map_id=map_id if map_id else "",
        ctime=mapdata.time_map + t0,
    )

    return injected_sources


def inject_simulated_sources(
    mapdata: Depth1Map,
    injected_source_db: SourceCatalogDatabase = None,
    sim_params: dict = {},
    map_id: str = None,
    t0: float = 0.0,
    verbose: bool = False,
    inject_transients: bool = False,
    use_map_geometry: bool = False,
    simulated_transient_database: str = None,
):
    """
    Inject sources into a map using photutils.
    Add injected sources to the injected_source_db.
    Parameters:
    - mapdata: Depth1Map object containing the map to inject sources into.
    - injected_source_db: Database object to store injected sources.
    - sim_params: Dictionary containing simulation parameters.
    - map_id: Unique identifier for the map.
    - t0: Time offset for the map.
    - band_fwhm: Bandwidth full width at half maximum (FWHM) in arcminutes.
    - verbose: bool, if True, print debug information.
    - inject_transients: bool, if True, inject transients into the map.
    - use_map_geometry: bool, if True, use the map geometry for transient injection.
    - simulated_transient_database: str, path to the database for simulated transients.
    """
    if not sim_params and not injected_source_db and not simulated_transient_database:
        return [], []

    from ..utils.utils import get_fwhm
    from .sim_sources import generate_transients
    from .sim_utils import load_transients_from_db

    if use_map_geometry:
        from ..maps.maps import edge_map

    catalog_sources = []
    if injected_source_db:
        catalog_sources = inject_sources_from_db(
            mapdata,
            injected_source_db,
            t0=t0,
        )

    catalog_sources += inject_random_sources(
        mapdata,
        sim_params,
        map_id=map_id if map_id else "",
        t0=t0,
        fwhm_arcmin=get_fwhm(mapdata.freq, arr=mapdata.wafer_name),
        add_noise=False,
    )

    injected_sources = []
    if inject_transients:
        transient_sources = load_transients_from_db(simulated_transient_database)
        mapdata, inj_sources = inject_sources(
            mapdata,
            transient_sources,
            mapdata.time_map + t0,
            freq=mapdata.freq,
            arr=mapdata.wafer_name,
            map_id=map_id if map_id else "",
            debug=verbose,
        )
        injected_sources += inj_sources
        if sim_params:
            if sim_params["injected_transients"]["n_transients"] > 0:
                if verbose:
                    print("Generating transients using sim config.")
                if use_map_geometry:
                    ## use imap to generate random pixels
                    ra_lims = None
                    dec_lims = None
                    ## get the hits map using the time map if it exists,
                    ## otherwise use the flux map (setting nans to zero). Using flux hides the masked region too, though.
                    if isinstance(mapdata.time_map, enmap.ndmap):
                        hits_map = edge_map(mapdata.time_map)
                    else:
                        hits_map = edge_map(np.nan_to_num(mapdata.flux))
                else:
                    ra_lims = (
                        sim_params["maps"]["center_ra"]
                        - sim_params["maps"]["width_ra"],
                        sim_params["maps"]["center_ra"]
                        + sim_params["maps"]["width_ra"],
                    )
                    dec_lims = (
                        sim_params["maps"]["center_dec"]
                        - sim_params["maps"]["width_dec"],
                        sim_params["maps"]["center_dec"]
                        + sim_params["maps"]["width_dec"],
                    )
                    hits_map = None

                transients_to_inject = generate_transients(
                    n=sim_params["injected_transients"]["n_transients"],
                    imap=hits_map,
                    ra_lims=ra_lims,
                    dec_lims=dec_lims,
                    peak_amplitudes=(
                        sim_params["injected_transients"]["min_flux"],
                        sim_params["injected_transients"]["max_flux"],
                    ),
                    peak_times=(mapdata.map_ctime, mapdata.map_ctime),
                    flare_widths=(
                        sim_params["injected_transients"]["min_width"],
                        sim_params["injected_transients"]["max_width"],
                    ),
                    uniform_on_sky=True if isinstance(hits_map, type(None)) else False,
                )
                mapdata, inj_sources = inject_sources(
                    mapdata,
                    transients_to_inject,
                    mapdata.time_map + t0,
                    freq=mapdata.freq,
                    arr=mapdata.wafer_name,
                    map_id=map_id if map_id else "",
                    debug=verbose,
                )
                injected_sources += inj_sources

    if verbose:
        print(f"Injected {len(catalog_sources)} sources into simulated map")
        print(f"Injected {len(injected_sources)} transients into simulated map")

    if isinstance(injected_source_db, SourceCatalogDatabase):
        ## add the catalog sources to the injected_source_db
        injected_source_db.add_sources(injected_sources)
        injected_source_db.add_sources(catalog_sources)

    return catalog_sources, injected_sources
