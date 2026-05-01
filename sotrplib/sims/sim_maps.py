from datetime import datetime, timezone

import numpy as np
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sims.sim_sources import SimulatedSource
from sotrplib.sources.forced_photometry import (
    convert_catalog_to_registered_source_objects,
)
from sotrplib.sources.sources import MeasuredSource


def make_enmap(
    center_ra: AstroPydanticQuantity[u.deg] = 0.0 * u.deg,
    center_dec: AstroPydanticQuantity[u.deg] = 0.0 * u.deg,
    width_ra: AstroPydanticQuantity[u.deg] = 1.0 * u.deg,
    width_dec: AstroPydanticQuantity[u.deg] = 1.0 * u.deg,
    resolution: AstroPydanticQuantity[u.arcmin] = 0.5 * u.arcmin,
    map_noise: AstroPydanticQuantity[u.Jy] = None,
    log: FilteringBoundLogger | None = None,
):
    """ """
    from pixell.enmap import geometry, zeros

    log = log if log else get_logger()
    log = log.bind(func_name="make_enmap")
    if (width_ra <= 0 * u.deg) or (width_dec <= 0 * u.deg):
        log.error(
            "make_enmap.invalid_map_dimensions", width_ra=width_ra, width_dec=width_dec
        )
        raise ValueError("Map width must be positive")

    min_dec = center_dec - width_dec
    max_dec = center_dec + width_dec
    if min_dec < -90 * u.deg:
        log.error("make_enmap.invalid_min_dec", min_dec=min_dec)
        raise ValueError(
            "Minimum declination, %.1f deg, is below -90 deg" % min_dec.to(u.deg).value
        )
    if max_dec > 90 * u.deg:
        log.error("make_enmap.invalid_max_dec", max_dec=max_dec)
        raise ValueError(
            "Maximum declination, %.1f deg, is above 90 deg" % max_dec.to(u.deg).value
        )

    min_ra = center_ra - width_ra
    max_ra = center_ra + width_ra
    ## need to do something with ra limits unless geometry knows how to do the wrapping.

    box = np.array(
        [
            [min_dec.to_value(u.rad), min_ra.to_value(u.rad)],
            [max_dec.to_value(u.rad), max_ra.to_value(u.rad)],
        ]
    )
    shape, wcs = geometry(box, res=resolution.to_value(u.rad))
    log.bind(box=box, shape=shape, wcs=wcs)
    if map_noise is not None:
        return make_noise_map(
            zeros(shape, wcs=wcs),
            map_noise_Jy=map_noise,
            map_mean_Jy=0.0 * u.Jy,
            seed=None,
            log=log,
        )
    else:
        log.info("make_enmap.empty_map_created")
        return zeros(shape, wcs=wcs)


def make_time_map(
    imap: enmap.ndmap,
    start_time: datetime,
    end_time: datetime,
) -> enmap.ndmap:
    """
    A simple time map simulation where each pixel is observed at a unique time.
    """
    shape = imap.shape
    start_timestamp = start_time.timestamp()
    end_timestamp = end_time.timestamp()

    time_offset_y = (end_timestamp - start_timestamp) / shape[1]

    times, sub_times = np.meshgrid(
        np.linspace(start_timestamp, end_timestamp, shape[1]),
        np.linspace(0.0, time_offset_y, shape[0]),
    )

    time_map = enmap.enmap(times + sub_times, imap.wcs, dtype=np.float64)

    return time_map


def make_noise_map(
    imap: enmap.ndmap,
    map_noise_Jy: AstroPydanticQuantity[u.Jy] = 0.01 * u.Jy,
    map_mean_Jy: AstroPydanticQuantity[u.Jy] = 0.0 * u.Jy,
    seed: int = None,
    log: FilteringBoundLogger | None = None,
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

    log = log if log else get_logger()
    shape = imap.shape
    wcs = imap.wcs
    noise_map = enmap.zeros(shape, wcs=wcs)
    log = log.bind(
        func_name="make_noise_map",
        shape=shape,
        map_noise_Jy=map_noise_Jy,
        map_mean_Jy=map_mean_Jy,
        seed=seed,
    )
    if not seed:
        seed = np.random.seed()
        log.warning("make_noise_map.randomizing_seed", seed=seed)
    mask = edge_map(
        imap,
    )
    if np.all(mask == 0):
        mask = np.ones(
            shape
        )  # If the mask is all zeros, invert it to create a full mask.
    noise_map += make_noise_image(
        shape,
        distribution="gaussian",
        mean=map_mean_Jy.to_value(u.Jy),
        stddev=map_noise_Jy.to_value(u.Jy),
        seed=seed,
    )
    noise_map *= mask
    log.info("make_noise_map.noise_map_created", shape=shape)
    return noise_map


def photutils_sim_n_sources(
    sim_map: enmap.ndmap | ProcessableMap,
    n_sources: float,
    min_flux_Jy: u.Quantity[u.Jy] = 0.1 * u.Jy,
    max_flux_Jy: u.Quantity[u.Jy] = 2.0 * u.Jy,
    map_noise_Jy: u.Quantity[u.Jy] = 0.01 * u.Jy,
    gauss_fwhm: u.Quantity[u.arcmin] = 2.2 * u.arcmin,
    fwhm_uncert_frac: float = 0.01,
    gauss_theta_min: u.Quantity[u.deg] = 0 * u.deg,
    gauss_theta_max: u.Quantity[u.deg] = 90 * u.deg,
    min_sep: u.Quantity[u.arcmin] = 5 * u.arcmin,
    seed: int = None,
    log: FilteringBoundLogger | None = None,
):
    """ """
    from photutils.psf import GaussianPSF, make_psf_model_image

    from .sim_utils import convert_photutils_qtable_to_json

    log = log if log else get_logger()
    log = log.bind(func_name="photutils_sim_n_sources")

    if min_flux_Jy >= max_flux_Jy:
        log.error(
            "photutils_sim_n_sources.invalid_flux_range",
            min_flux_Jy=min_flux_Jy,
            max_flux_Jy=max_flux_Jy,
        )
        raise ValueError("Min injected flux is less than or equal to max injected flux")

    model = GaussianPSF()
    if isinstance(sim_map, ProcessableMap):
        mapres = sim_map.map_resolution
        shape = sim_map.flux.shape
        wcs = sim_map.flux.wcs
    elif isinstance(sim_map, enmap.ndmap):
        shape = sim_map.shape
        wcs = sim_map.wcs
        mapres = u.Quantity(abs(wcs.wcs.cdelt[0]), wcs.wcs.cunit[0])
    else:
        log.error("photutils_sim_n_sources.invalid_map_type", sim_map=sim_map)
        raise ValueError("Input map must be a ProcessableMap or enmap.ndmap object")

    # Omega_b is the beam solid angle for a Gaussian beam in sr
    ## I've found that the recovered flux is low by 2% ish...
    omega_b = 1.02 * (np.pi / 4 / np.log(2)) * (gauss_fwhm.to_value(u.rad)) ** 2
    gauss_fwhm = gauss_fwhm.to(u.deg).value / mapres.to(u.deg).value  # in pixels
    minfwhm = gauss_fwhm - (fwhm_uncert_frac * gauss_fwhm)
    maxfwhm = gauss_fwhm + (fwhm_uncert_frac * gauss_fwhm)
    minsep = min_sep.to(u.arcmin).value / mapres.to(u.arcmin).value  # in pixels
    log.bind(minfwhm=minfwhm, maxfwhm=maxfwhm, minsep=minsep, omega_b=omega_b)
    if not seed:
        seed = np.random.seed()
        log.debug("photutils_sim_n_sources.seed_randomized", seed=seed)

    ## set the window for each source to be 5x fwhm
    model_window = (int(5 * gauss_fwhm), int(5 * gauss_fwhm))
    ## make the source map and add to input map
    data, params = make_psf_model_image(
        shape,
        model,
        n_sources,
        model_shape=model_window,
        min_separation=minsep,
        border_size=int(minsep),
        flux=(min_flux_Jy.to(u.Jy).value, max_flux_Jy.to(u.Jy).value),
        x_fwhm=(minfwhm, maxfwhm),
        y_fwhm=(minfwhm, maxfwhm),
        theta=(gauss_theta_min.to(u.deg).value, gauss_theta_max.to(u.deg).value),
        seed=seed,
        normalize=False,
    )
    log.info(
        "photutils_sim_n_sources.model_image_created", shape=shape, n_sources=n_sources
    )
    ## photutils distributes the flux over the solid angle
    ## so this map should be the intensity map in Jy/sr
    ## multiply by beam area in sq pixels
    simflux = data * (omega_b / mapres.to_value(u.rad) ** 2)
    if isinstance(sim_map, ProcessableMap):
        sim_map.snr += simflux / (sim_map.flux / sim_map.snr)
        sim_map.flux += simflux

    elif isinstance(sim_map, enmap.ndmap):
        sim_map += simflux

    log.info("photutils_sim_n_sources.beam_convolved")

    ## add noise to the map
    if map_noise_Jy > 0 * u.Jy:
        if isinstance(sim_map, ProcessableMap):
            sim_map.flux += make_noise_map(
                sim_map.flux,
                map_noise_Jy=map_noise_Jy,
                map_mean_Jy=0.0 * u.Jy,
                seed=seed,
                log=log,
            )
        elif isinstance(sim_map, enmap.ndmap):
            sim_map += make_noise_map(
                sim_map,
                map_noise_Jy=map_noise_Jy,
                map_mean_Jy=0.0 * u.Jy,
                seed=seed,
                log=log,
            )
        log.info("photutils_sim_n_sources.noise_added", map_noise_Jy=map_noise_Jy)

    ## photutils parameter table is in Astropy QTable
    ## convert to the json format we've been using.
    injected_sources = convert_photutils_qtable_to_json(
        params,
        imap=sim_map if isinstance(sim_map, enmap.ndmap) else sim_map.flux,
    )
    injected_sources = convert_catalog_to_registered_source_objects(
        injected_sources, source_type="simulated", log=log
    )

    log.info(
        "photutils_sim_n_sources.sources_injected", n_sources=len(injected_sources)
    )
    return sim_map, injected_sources


def inject_sources(
    imap: enmap.ndmap | ProcessableMap,
    sources: list[SimulatedSource],
    observation_time: float | enmap.ndmap,
    freq: str = "f090",
    arr: str | None = None,
    instrument: str | None = None,
    debug: bool = False,
    log: FilteringBoundLogger | None = None,
):
    """
    Inject a list of SimulatedSource objects into the input map.

    Parameters:
    - imap: enmap.ndmap, the input map to inject sources into.
    - sources: list of SimulatedSource objects.
    - observation_time: float or enmap.ndmap, the time of the observation.
    - freq: str, the frequency of the observation.
    - arr: str, the array name (optional).
    - map_id: str, the map ID (optional).
    - debug: bool, if True, print debug information including tqdm.

    Returns:
    - imap: enmap.ndmap, the map with sources injected.
    - injected_sources: list of MeasuredSource objects representing the injected sources.
    """
    from photutils.datasets import make_model_image
    from photutils.psf import GaussianPSF
    from pixell.enmap import sky2pix
    from tqdm import tqdm

    from ..utils.utils import get_fwhm
    from .sim_utils import make_2d_gaussian_model_param_table

    log = log if log else get_logger()
    log = log.bind(func_name="inject_sources")
    if not sources:
        log.warning("inject_sources.no_sources", n_sources=0)
        return imap, []

    if isinstance(imap, ProcessableMap):
        shape = imap.flux.shape
        wcs = imap.flux.wcs
    elif isinstance(imap, enmap.ndmap):
        shape = imap.shape
        wcs = imap.wcs
    else:
        log.error("inject_sources.invalid_map_type", map_type=type(imap))
        raise ValueError("Input map must be a ProcessableMap or enmap.ndmap object")

    mapres = abs(wcs.wcs.cdelt[0]) * u.deg
    fwhm = get_fwhm(freq, arr=arr, instrument=instrument)
    fwhm_pixels = (fwhm / mapres).value

    log.info("inject_sources.injecting_sources", n_sources=len(sources))
    injected_sources = []
    removed_sources = {"not_flaring": 0, "out_of_bounds": 0}
    for source in tqdm(sources, desc="Injecting sources", disable=not debug):
        # Check if source is within the map bounds
        source_position = source.position(observation_time)
        ra, dec = source_position.ra, source_position.dec
        pix = sky2pix(shape, wcs, np.asarray([dec.to("rad").value, ra.to("rad").value]))
        if not (0 <= pix[0] < shape[-2] and 0 <= pix[1] < shape[-1]):
            removed_sources["out_of_bounds"] += 1
            continue
        if isinstance(imap, ProcessableMap):
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

        if (
            hasattr(source, "peak_time")
            and hasattr(source, "flare_width")
            and source.peak_time is not None
            and source.flare_width is not None
        ):
            if (
                abs(source.peak_time.timestamp() - source_obs_time)
                > 3 * source.flare_width.total_seconds()
            ):
                removed_sources["not_flaring"] += 1
                continue
        ## flux requires an aware time, so convert the timestamp to a datetime with utc timezone
        awaretime = datetime.fromtimestamp(source_obs_time, tz=timezone.utc)
        flux = source.flux(awaretime)

        inj_source = MeasuredSource(
            ra=ra,
            dec=dec,
            flux=flux,
            source_type="simulated",
            fit_failed=False,
            fwhm_ra=fwhm,
            fwhm_dec=fwhm,
            frequency=float(freq[1:]) * u.GHz if freq.startswith("f") else None,
        )
        injected_sources.append(inj_source)
    log.info("inject_sources.sources_to_inject", injected_sources=injected_sources)

    model_params = make_2d_gaussian_model_param_table(
        imap if isinstance(imap, enmap.ndmap) else imap.flux,
        sources=injected_sources,
        nominal_fwhm=fwhm,
        cuts={},
        verbose=debug,
        log=log,
    )
    log.info("inject_sources.making_psf_image", model_params=model_params)
    psf_image = make_model_image(
        shape,
        GaussianPSF(),
        model_params,
        model_shape=(int(5 * fwhm_pixels), int(5 * fwhm_pixels)),
    )

    if isinstance(imap, ProcessableMap):
        imap.snr += psf_image / (imap.flux / imap.snr)
        imap.flux += psf_image
    elif isinstance(imap, enmap.ndmap):
        imap += psf_image
    log.info("inject_sources.sources_injected", n_sources=len(injected_sources))
    if debug:
        log.debug(
            "inject_sources.source_injection_summary",
            removed_sources=removed_sources,
            injected_sources=injected_sources,
        )

    return imap, injected_sources


def inject_random_sources(
    mapdata,
    sim_params: dict,
    fwhm_arcmin: u.Quantity[u.arcmin] = 2.2 * u.arcmin,
    add_noise: bool = False,
    log: FilteringBoundLogger | None = None,
):
    """
    Inject sources into a map using photutils.
    Add injected sources to the injected_source_db.
    Parameters:
    - mapdata: ProcessableMap object containing the map to inject sources into.
    - injected_source_db: Database object to store injected sources.
    - sim_params: Dictionary containing simulation parameters.
    - map_id: Unique identifier for the map.
    - fwhm_arcmin: Bandwidth full width at half maximum (FWHM) in arcminutes.
    - return_source_object_list: bool, if True, return the list of injected source objects.
        otherwise, return a source catalog dict.
    - add_noise: bool, if True, add sim_params['maps']['map_noise'] to the map.
        Assumes this is already injected since sim maps load this by default.
    """
    log = log if log else get_logger()
    log = log.bind(func_name="inject_random_sources")
    if not sim_params:
        log.info("inject_random_sources.no_sim_params", sim_params=sim_params)
        return []

    mapdata, injected_sources = photutils_sim_n_sources(
        mapdata,
        sim_params["injected_sources"]["n_sources"],
        min_flux_Jy=sim_params["injected_sources"]["min_flux"],
        max_flux_Jy=sim_params["injected_sources"]["max_flux"],
        map_noise_Jy=sim_params["maps"]["map_noise"] if add_noise else 0.0 * u.Jy,
        fwhm_uncert_frac=sim_params["injected_sources"]["fwhm_uncert_frac"],
        gauss_fwhm=fwhm_arcmin,
        log=log,
    )
    log.info(
        "inject_random_sources.injected_sources", injected_sources=injected_sources
    )
    return injected_sources


def inject_simulated_sources(
    mapdata: ProcessableMap,
    sim_params: dict = {},
    verbose: bool = False,
    inject_transients: bool = False,
    use_map_geometry: bool = False,
    simulated_transient_database: str = None,
    log: FilteringBoundLogger | None = None,
):
    """
    Inject sources into a map using photutils.
    Add injected sources to the injected_source_db.
    Parameters:
    - mapdata: ProcessableMap object containing the map to inject sources into.
    - injected_source_db: Database object to store injected sources.
    - sim_params: Dictionary containing simulation parameters.
    - map_id: Unique identifier for the map.
    - band_fwhm: Bandwidth full width at half maximum (FWHM) in arcminutes.
    - verbose: bool, if True, print debug information.
    - inject_transients: bool, if True, inject transients into the map.
    - use_map_geometry: bool, if True, use the map geometry for transient injection.
    - simulated_transient_database: str, path to the database for simulated transients.
    - log: structlog bound logger object
    """
    log = log or get_logger()
    log = log.bind(func_name="inject_simulated_sources")
    if not sim_params and not simulated_transient_database:
        log.info("inject_simulated_sources.skip")
        return [], []

    from ..utils.utils import get_fwhm
    from .sim_sources import generate_transients
    from .sim_utils import load_transients_from_db

    log = log.bind(sim_params=sim_params)

    if use_map_geometry:
        from ..maps.maps import edge_map
    log = log.bind(use_map_geometry=use_map_geometry)

    catalog_sources = []
    log = log.bind(injected_sources=catalog_sources)
    log.info("inject_simulated_sources.injected_source_db")

    catalog_sources += inject_random_sources(
        mapdata,
        sim_params,
        fwhm_arcmin=get_fwhm(
            mapdata.frequency, arr=mapdata.array, instrument=mapdata.instrument
        ),
        add_noise=False,
        log=log,
    )
    log = log.bind(injected_sources=catalog_sources)
    log.info("inject_simulated_sources.inject_random_sources")

    injected_sources = []
    if inject_transients:
        log.info("inject_simulated_sources.inject_transients")
        transient_sources = load_transients_from_db(
            simulated_transient_database, log=log
        )
        log = log.bind(transient_sources=transient_sources)
        log.info("inject_simulated_sources.inject_transients.load_transients_from_db")

        mapdata, inj_sources = inject_sources(
            mapdata,
            transient_sources,
            mapdata.time_mean,
            freq=mapdata.frequency,
            arr=mapdata.array,
            debug=verbose,
            log=log,
        )
        injected_sources += inj_sources
        log = log.bind(injected_sources=inj_sources)
        log.info("inject_simulated_sources.inject_transients.inject_sources")

        if sim_params:
            if sim_params["injected_transients"]["n_transients"] > 0:
                if use_map_geometry:
                    ## use imap to generate random pixels
                    ra_lims = None
                    dec_lims = None
                    ## get the hits map using the time map if it exists,
                    ## otherwise use the flux map (setting nans to zero). Using flux hides the masked region too, though.
                    if isinstance(mapdata.time_mean, enmap.ndmap):
                        hits_map = edge_map(mapdata.time_mean)
                    else:
                        hits_map = edge_map(np.nan_to_num(mapdata.flux))
                else:
                    ra_lims = (
                        sim_params["maps"]["center_ra"]
                        - 0.5 * sim_params["maps"]["width_ra"],
                        sim_params["maps"]["center_ra"]
                        + 0.5 * sim_params["maps"]["width_ra"],
                    )
                    dec_lims = (
                        sim_params["maps"]["center_dec"]
                        - 0.5 * sim_params["maps"]["width_dec"],
                        sim_params["maps"]["center_dec"]
                        + 0.5 * sim_params["maps"]["width_dec"],
                    )
                    hits_map = None
                log = log.bind(ra_lims=ra_lims, dec_lims=dec_lims, hits_map=hits_map)

                min_time = np.min(mapdata.time_first[mapdata.time_first > 0])
                max_time = np.max(mapdata.time_last[np.isfinite(mapdata.time_last)])
                transients_to_inject = generate_transients(
                    n=sim_params["injected_transients"]["n_transients"],
                    imap=hits_map,
                    ra_lims=ra_lims,
                    dec_lims=dec_lims,
                    peak_amplitudes=(
                        sim_params["injected_transients"]["min_flux"],
                        sim_params["injected_transients"]["max_flux"],
                    ),
                    peak_times=(min_time, max_time),
                    flare_widths=(
                        sim_params["injected_transients"]["min_width"],
                        sim_params["injected_transients"]["max_width"],
                    ),
                    uniform_on_sky=True if isinstance(hits_map, type(None)) else False,
                    log=log,
                )
                log.info(
                    "inject_simulated_sources.inject_transients.generate_transients"
                )
                mapdata, inj_sources = inject_sources(
                    mapdata,
                    transients_to_inject,
                    mapdata.time_mean,
                    freq=mapdata.frequency,
                    arr=mapdata.array,
                    debug=verbose,
                    log=log,
                )
                injected_sources += inj_sources
                log = log.bind(injected_sources=inj_sources)
                log.info(
                    "inject_simulated_sources.inject_transients.generate_transients.inject_sources"
                )

    log.info(
        "inject_simulated_sources.final_counts",
        n_injected_sources=len(catalog_sources),
        n_injected_transients=len(injected_sources),
    )

    return catalog_sources, injected_sources
