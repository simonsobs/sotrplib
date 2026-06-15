from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from structlog import get_logger
from structlog.types import FilteringBoundLogger

if TYPE_CHECKING:
    from sotrplib.sources.sources import MeasuredSource


def generate_random_positions_in_map(
    n: int,
    imap: enmap.ndmap,
    log: FilteringBoundLogger | None = None,
):
    """Generate ``n`` random positions uniformly distributed in pixel space.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    imap : enmap.ndmap
        Input map; its shape and WCS define the pixel grid.
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    list of tuple
        List of ``(dec, ra)`` pairs as astropy Quantities.
    """
    log = log or get_logger()
    x = np.random.uniform(0, imap.shape[0], n)
    y = np.random.uniform(0, imap.shape[1], n)
    positions = []
    for i in range(n):
        positions.append(tuple(imap.pix2sky((x[i], y[i])) * u.rad))

    return positions


def generate_random_positions(
    n: int,
    imap: enmap.ndmap = None,
    ra_lims: tuple[AstroPydanticQuantity[u.deg]] | None = None,
    dec_lims: tuple[AstroPydanticQuantity[u.deg]] | None = None,
    edge_buffer: AstroPydanticQuantity[u.deg] = 5 * u.arcmin,
    log: FilteringBoundLogger | None = None,
):
    """Generate ``n`` random positions uniformly distributed on the sphere.

    Parameters
    ----------
    n : int
        Number of positions to generate.
    imap : enmap.ndmap, optional
        If provided, RA/Dec limits are derived from the map bounding box.
    ra_lims : tuple of Quantity[deg], optional
        ``(min_ra, max_ra)`` limits.  Required if ``imap`` is not given.
        RA wrap-around (e.g. ``(350 deg, 10 deg)``) is handled automatically.
    dec_lims : tuple of Quantity[deg], optional
        ``(min_dec, max_dec)`` limits.  Required if ``imap`` is not given.
    edge_buffer : Quantity[deg], optional
        Inset applied to map-derived limits (default 5 arcmin).
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    list of tuple
        List of ``(dec, ra)`` pairs as astropy Quantities in degrees.
    """
    log = log or get_logger()
    if ra_lims is None or dec_lims is None:
        if imap is not None:
            shape, wcs = imap.shape, imap.wcs
            dec_min, dec_max = enmap.box(shape, wcs)[:, 0] * u.rad
            ra_max, ra_min = enmap.box(shape, wcs)[:, 1] * u.rad
            ra_lims = (ra_min + edge_buffer, ra_max - edge_buffer)
            dec_lims = (dec_min + edge_buffer, dec_max - edge_buffer)
        else:
            raise ValueError(
                "Either ra_lims and dec_lims must be provided, or imap must be supplied."
            )
    ## assume that if limits are something like (350,10) that the ra limits wrap 0, so -10,10
    if ra_lims[0] > ra_lims[1]:
        if ra_lims[0] > 180 * u.deg:
            ra_lims = (ra_lims[0] - 360 * u.deg, ra_lims[1])

    # Generate RA uniformly between ra_lims
    ra = (
        np.random.uniform(ra_lims[0].to_value(u.rad), ra_lims[1].to_value(u.rad), n)
        * u.rad
    )

    # Generate Dec uniformly on the sphere
    z_min = np.sin(dec_lims[0].to_value(u.rad))
    z_max = np.sin(dec_lims[1].to_value(u.rad))
    z = np.random.uniform(z_min, z_max, n)
    dec = np.arcsin(z) * u.rad

    ra = (ra.to_value(u.deg) % 360) * u.deg
    positions = list(zip(dec.to(u.deg), ra.to(u.deg)))
    return positions


def generate_random_flare_times(
    n: int,
    start_time: float | str = 1.4e9,
    end_time: float | str = 1.7e9,
    log=None,
):
    """Generate ``n`` random flare peak times between ``start_time`` and ``end_time``.

    Parameters
    ----------
    n : int
        Number of times to generate.
    start_time : float or str
        Start of the time window as a Unix timestamp or
        ``"YYYY-MM-DD HH:MM:SS"`` string.
    end_time : float or str
        End of the time window as a Unix timestamp or
        ``"YYYY-MM-DD HH:MM:SS"`` string.
    log : optional
        Unused; retained for API compatibility.

    Returns
    -------
    ndarray
        Array of ``n`` Unix timestamps drawn uniformly from the window.
    """
    from datetime import datetime

    # Convert string times to unix timestamps if necessary
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp()
    if isinstance(end_time, str):
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp()

    return np.random.uniform(start_time, end_time, n)


def generate_random_flare_widths(
    n: int,
    min_width: float = 0.1,
    max_width: float = 10.0,
    log=None,
):
    """Generate ``n`` random flare FWHMs drawn uniformly from a range.

    Parameters
    ----------
    n : int
        Number of widths to generate.
    min_width : float, optional
        Minimum flare width (default 0.1).
    max_width : float, optional
        Maximum flare width (default 10.0).
    log : optional
        Unused; retained for API compatibility.

    Returns
    -------
    ndarray
        Array of ``n`` flare widths.
    """
    return np.random.uniform(min_width, max_width, n)


def generate_random_flare_amplitudes(
    n: int,
    min_amplitude: AstroPydanticQuantity[u.Jy] = 0.1 * u.Jy,
    max_amplitude: AstroPydanticQuantity[u.Jy] = 10.0 * u.Jy,
    log: FilteringBoundLogger | None = None,
):
    """Generate ``n`` random flare peak amplitudes drawn uniformly from a range.

    Parameters
    ----------
    n : int
        Number of amplitudes to generate.
    min_amplitude : Quantity[Jy], optional
        Minimum amplitude (default 0.1 Jy).
    max_amplitude : Quantity[Jy], optional
        Maximum amplitude (default 10.0 Jy).
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    Quantity[Jy]
        Array of ``n`` flare amplitudes.
    """
    return (
        np.random.uniform(min_amplitude.to_value(u.Jy), max_amplitude.to_value(u.Jy), n)
        * u.Jy
    )


def make_gaussian_flare(
    unix_times: np.ndarray,
    flare_peak_time: float = 0.0,
    flare_fwhm_s: float = 1.0,
    flare_peak_Jy: float = 0.1,
    log: FilteringBoundLogger | None = None,
):
    """Evaluate a Gaussian light curve at an array of Unix timestamps.

    Parameters
    ----------
    unix_times : ndarray
        Sample times as Unix timestamps.
    flare_peak_time : float, optional
        Unix timestamp of the flare peak (default 0.0).
    flare_fwhm_s : float, optional
        Flare FWHM in seconds (default 1.0).
    flare_peak_Jy : float, optional
        Peak flux in Jy (default 0.1).
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    ndarray
        Flare flux in Jy sampled at each time in ``unix_times``.
    """
    # Convert FWHM to standard deviation
    sigma = flare_fwhm_s / (2 * np.sqrt(2 * np.log(2)))

    # Compute the Gaussian flare
    flare = flare_peak_Jy * np.exp(-0.5 * ((unix_times - flare_peak_time) / sigma) ** 2)

    return flare


def convert_photutils_qtable_to_json(
    params,
    imap: enmap.ndmap = None,
    log: FilteringBoundLogger | None = None,
):
    """Convert a photutils parameter QTable to a JSON-compatible dict.

    Parameters
    ----------
    params : QTable
        Photutils source parameter table with columns
        ``id``, ``x_0``, ``y_0``, ``flux``, ``x_fwhm``, ``y_fwhm``, ``theta``.
    imap : enmap.ndmap, optional
        If provided, RA/Dec are computed for each source and added to the output.
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    dict
        Dictionary of arrays with keys matching QTable columns plus
        ``RADeg``, ``decDeg``, ``name``, ``fluxJy``, ``err_fluxJy``, and
        ``forced``.
    """
    log = log or get_logger()
    log.bind(func_name="convert_photutils_qtable_to_json")
    json_cat_out = {}
    for p in params:
        for k in p.keys():
            if k not in json_cat_out:
                json_cat_out[k] = []
            json_cat_out[k].append(p[k])

        if isinstance(imap, enmap.ndmap):
            dec, ra = np.degrees(imap.pix2sky([p["y_0"], p["x_0"]]))
        else:
            ra, dec = np.nan, np.nan
        if "RADeg" not in json_cat_out:
            json_cat_out["RADeg"] = []
            json_cat_out["decDeg"] = []
            json_cat_out["name"] = []

        json_cat_out["RADeg"].append(ra)
        json_cat_out["decDeg"].append(dec)
        json_cat_out["name"].append(str(p["id"]))
    for key in json_cat_out:
        json_cat_out[key] = np.array(json_cat_out[key])
    flux = json_cat_out.pop("flux")
    ## this will be biased by the source
    json_cat_out["err_fluxJy"] = imap.std() * np.ones_like(flux)
    json_cat_out["fluxJy"] = flux
    json_cat_out["forced"] = np.ones_like(flux)

    return json_cat_out


def ra_lims_valid(
    ra_lims: AstroPydanticQuantity[u.deg] = None,
    log: FilteringBoundLogger | None = None,
):
    """Check whether RA limits are a valid ``(min_ra, max_ra)`` pair.

    Parameters
    ----------
    ra_lims : Quantity[deg], optional
        Two-element tuple of RA limits.
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    bool
        ``True`` if limits are a length-2 tuple within ``[0, 360]`` deg.
    """
    if ra_lims is None:
        return False
    if len(ra_lims) != 2:
        return False
    if ra_lims[0] < 0 * u.deg or ra_lims[1] > 360 * u.deg:
        return False

    return True


def dec_lims_valid(
    dec_lims: AstroPydanticQuantity[u.deg] | None = None,
    log: FilteringBoundLogger | None = None,
):
    """Check whether Dec limits are a valid ``(min_dec, max_dec)`` pair.

    Parameters
    ----------
    dec_lims : Quantity[deg], optional
        Two-element tuple of Dec limits.
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    bool
        ``True`` if limits are a length-2 tuple within ``[-90, 90]`` deg.
    """
    if dec_lims is None:
        return False
    if len(dec_lims) != 2:
        return False
    if dec_lims[0] < -90 * u.deg or dec_lims[1] > 90 * u.deg:
        return False

    return True


def load_config_yaml(
    config_path: str,
    log: FilteringBoundLogger | None = None,
):
    """Load a YAML configuration file into a dictionary.

    Parameters
    ----------
    config_path : str
        Path to the YAML file.
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    dict
        Parsed configuration, or an empty dict if the file does not exist.
    """
    import os

    import yaml

    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_sim_map_group(
    sim_params,
    log: FilteringBoundLogger | None = None,
):
    freq_arr_idx = (
        sim_params["array_info"]["arr"] + "_" + sim_params["array_info"]["freq"]
    )
    indexed_map_groups = {
        freq_arr_idx: [[i] for i in np.arange(sim_params["maps"]["n_realizations"])]
    }
    indexed_map_group_time_ranges = {
        freq_arr_idx: [
            [t]
            for t in np.linspace(
                float(sim_params["maps"]["min_time"]),
                float(sim_params["maps"]["max_time"]),
                int(sim_params["maps"]["n_realizations"]),
            )
        ]
    }
    return freq_arr_idx, indexed_map_groups, indexed_map_group_time_ranges


def make_2d_gaussian_model_param_table(
    imap: enmap.ndmap,
    sources: list[MeasuredSource],
    nominal_fwhm: u.Quantity = 2.2 * u.arcmin,
    verbose: bool = False,
    cuts={},
    log: FilteringBoundLogger | None = None,
):
    """Build a photutils-compatible 2-D Gaussian parameter table from measured sources.

    Parameters
    ----------
    imap : enmap.ndmap
        Map used to convert sky positions to pixel coordinates.
    sources : list of MeasuredSource
        Source measurements to tabulate.
    nominal_fwhm : Quantity, optional
        Fallback FWHM when a source fit did not converge (default 2.2 arcmin).
    verbose : bool, optional
        Log debug output for each source (default ``False``).
    cuts : dict, optional
        Quality cuts as ``{field: [min, max]}`` applied before tabulation.
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    QTable
        Astropy QTable with columns ``x_0``, ``y_0``, ``x_fwhm``, ``y_fwhm``,
        ``flux``, ``theta``, ``id`` suitable for ``photutils.psf.make_model_image``.
    """
    from astropy.table import QTable

    log = log or get_logger()
    log = log.bind(func_name="make_2d_gaussian_model_param_table")
    model_params = {
        "x_0": [],
        "y_0": [],
        "x_fwhm": [],
        "y_fwhm": [],
        "flux": [],
        "theta": [],
        "id": [],
    }
    id_num = 0
    map_res = abs(imap.wcs.wcs.cdelt[0]) * u.deg
    log.info(
        "make_2d_gaussian_model_param_table.initialize_params",
        model_params=model_params,
    )
    for i in range(len(sources)):
        s = sources[i]
        pix = imap.sky2pix(np.array([s.dec.to(u.rad).value, s.ra.to(u.rad).value]))

        ## unfortunate naming, a,b are semi-major and semi-minor axes,
        ## but really they're x,y from photutils gaussian fit.
        cut = False
        if s.fit_failed:
            cut = True
            if verbose:
                log.debug(
                    "make_2d_gaussian_model_param_table.cut_fit_failed",
                    source_id=i,
                    cutkey="fit_failed",
                )
            continue

        for cutkey in cuts.keys():
            if cutkey not in s.model_dump():
                log.error(
                    "make_2d_gaussian_model_param_table.cutkey_not_found", cutkey=cutkey
                )
                raise ValueError(f"Cut {cutkey} not found in source attributes.")
            if s.model_dump()[cutkey] is None:
                cut = True
                break
            if (
                np.isnan(s.model_dump()[cutkey])
                or (s.model_dump()[cutkey] < cuts[cutkey][0])
                or (s.model_dump()[cutkey] > cuts[cutkey][1])
            ):
                if verbose:
                    log.debug(
                        "make_2d_gaussian_model_param_table.cut_failed",
                        source_id=i,
                        cutkey=cutkey,
                        value=s.__dict__[cutkey],
                    )
                cut = True
                break
        if not cut:
            try:
                fwhm_x = (
                    math.sqrt(2) * s.fwhm_ra / math.cos(s.dec.to_value(u.rad))
                )  ## account for the declination
                fwhm_y = math.sqrt(2) * s.fwhm_dec
            except Exception:
                fwhm_x = None
                fwhm_y = None
            if fwhm_x is None:
                fwhm_x = nominal_fwhm
            if fwhm_y is None:
                fwhm_y = nominal_fwhm
            try:
                fit_params = s.fit_params if s.fit_params is not None else {}
            except Exception:
                fit_params = {}
            if "theta" in fit_params.keys():
                theta = fit_params["theta"]
            else:
                theta = 0.0 * u.deg

            omega_b = (math.pi / 4.0 / math.log(2)) * (fwhm_x * fwhm_y / map_res**2)

            # Define the PSF model parameters
            model_params["x_0"].append(pix[1])
            model_params["y_0"].append(pix[0])
            model_params["x_fwhm"].append((fwhm_x / map_res).value)
            model_params["y_fwhm"].append((fwhm_y / map_res).value)
            model_params["flux"].append(s.flux.to(u.Jy).value * omega_b)
            model_params["theta"].append(theta.to(u.rad).value)
            model_params["id"].append(id_num)
            id_num += 1
    if verbose:
        log.debug(
            "make_2d_gaussian_model_param_table.complete", model_params=model_params
        )

    return QTable(model_params)
