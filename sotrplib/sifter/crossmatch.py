from itertools import combinations

import numpy as np
import uuid7
from astropy import units as u
from astropy.coordinates import angular_separation
from astropydantic import AstroPydanticQuantity
from pixell import utils as pixell_utils
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.source_catalog.socat import SOCat

from ..sources.sources import CrossMatch, MeasuredSource, RegisteredSource


def crossmatch_mask(
    sources,
    crosscat,
    radius: float,
    mode: str = "all",
    return_matches: bool = False,
):
    """Crossmatch source positions against a catalog and return a match mask.

    Parameters
    ----------
    sources : ndarray
        Source positions as ``[[dec, ra]]`` in degrees.  A 1-D array is
        treated as a single source ``[dec, ra]``.
    crosscat : ndarray
        Catalog of positions to match against, as ``[[dec, ra]]`` in degrees.
    radius : float or array-like
        Search radius in arcmin.  If array-like, each element gives the
        matching radius for the source at the same index.
    mode : {"all", "closest"}, optional
        Return all matches or only the closest (default ``"all"``).
    return_matches : bool, optional
        If ``True``, also return the raw match index list (default ``False``).

    Returns
    -------
    mask : bool or ndarray of bool
        ``True`` where a source has at least one catalog match.
    matches : list, optional
        Raw match indices, returned only when ``return_matches=True``.
    """
    crosspos_ra = crosscat[:, 1] * pixell_utils.degree
    crosspos_dec = crosscat[:, 0] * pixell_utils.degree
    crosspos = np.array([crosspos_ra, crosspos_dec]).T

    # if only one source
    if len(sources.shape) == 1:
        source_ra = sources[1] * pixell_utils.degree
        source_dec = sources[0] * pixell_utils.degree
        sourcepos = np.array([[source_ra, source_dec]])
        if isinstance(radius, (list, np.ndarray)):
            r = radius[0] * pixell_utils.arcmin
        else:
            r = radius * pixell_utils.arcmin
        match = pixell_utils.crossmatch(sourcepos, crosspos, r, mode=mode)

        if len(match) > 0:
            return True if not return_matches else True, match
        else:
            return False if not return_matches else False, match

    mask = np.zeros(len(sources), dtype=bool)
    matches = []
    for i, _ in enumerate(mask):
        source_ra = sources[i, 1] * pixell_utils.degree
        source_dec = sources[i, 0] * pixell_utils.degree
        sourcepos = np.array([[source_ra, source_dec]])
        if isinstance(radius, (list, np.ndarray)):
            r = radius[i] * pixell_utils.arcmin
        else:
            r = radius * pixell_utils.arcmin
        match = pixell_utils.crossmatch(sourcepos, crosspos, r, mode=mode)
        if len(match) > 0:
            mask[i] = True
        matches.append([(i, m[1]) for m in match])
    if return_matches:
        return mask, matches
    else:
        return mask


def crossmatch_with_million_quasar_catalog(
    extracted_sources: list[MeasuredSource],
    mq_catalog: SOCat | None = None,
    match_threshold: AstroPydanticQuantity[u.arcmin] = 0.5 * u.arcmin,
    log: FilteringBoundLogger | None = None,
):
    """Crossmatch source candidates against the million-quasar catalog.

    Parameters
    ----------
    extracted_sources : list of MeasuredSource
        Detections to crossmatch.
    mq_catalog : SOCat, optional
        Million-quasar catalog.  If ``None``, returns an all-``False`` mask.
    match_threshold : Quantity[arcmin], optional
        Matching radius (default 0.5 arcmin).
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    isin_mq_cat : ndarray of bool
        ``True`` for each source matched in the quasar catalog.
    match_info : dict
        Per-source match detail keyed by source index.
    """
    log = log if log else get_logger()
    log = log.bind(func_name="crossmatch_million_quasar_catalog")

    if mq_catalog is None:
        log.warning("crossmatch_with_million_quasar_catalog.no_catalog")
        return np.zeros(len(extracted_sources), dtype=bool), {}

    isin_mq_cat = np.zeros(len(extracted_sources), dtype=bool)
    log.info("crossmatching", n_sources=len(extracted_sources))
    matches = []
    for i, src in enumerate(extracted_sources):
        matches.append(
            mq_catalog.get_nearby_source(
                ra=src["ra"], dec=src["dec"], radius=match_threshold
            )
        )

    match_info = {}
    for i in range(len(matches)):
        match_info[i] = {}
        if matches[i]:
            isin_mq_cat[i] = True
            match_info[i] = {
                "matches": matches[i],
                "pvalue": None,
            }  # TODO need to calculate probability.
    return isin_mq_cat, match_info


def gaia_match(
    cand: MeasuredSource,
    maxmag: float = 16,
    maxsep: AstroPydanticQuantity[u.deg] = 5 * u.arcmin,
    parallax_required=True,
    mag_key="phot_g_mean_mag",
    sep_key="dist",
):
    """Query Gaia for a bright stellar counterpart near a source candidate.

    Parameters
    ----------
    cand : MeasuredSource
        Source candidate to check.
    maxmag : float, optional
        Maximum Gaia magnitude to consider (default 16).
    maxsep : Quantity[deg], optional
        Cone-search radius (default 5 arcmin).
    parallax_required : bool, optional
        Require a finite Gaia parallax (default ``True``).
    mag_key : str, optional
        Gaia column name for magnitude (default ``"phot_g_mean_mag"``).
    sep_key : str, optional
        Gaia column name for angular separation (default ``"dist"``).

    Returns
    -------
    astropy.table.Table
        Matching Gaia sources; empty table if no match found.
    """
    from ..source_catalog.query_tools import cone_query_gaia

    gaia_results = cone_query_gaia(
        cand.ra,
        cand.dec,
        radius_arcmin=maxsep,
    )

    if gaia_results:
        parallax = (
            (np.isfinite(gaia_results["parallax"]))
            if parallax_required
            else (np.ones(len(gaia_results["parallax"]), dtype=bool))
        )
        gaia_valid = gaia_results[
            (gaia_results[mag_key] < maxmag)
            & parallax
            & (gaia_results[sep_key] < maxsep)
        ]
    else:
        gaia_valid = {}
    ## calculate pvalue, sort on pvalue.
    ## need to implement.
    if len(gaia_valid) > 0:
        gaia_valid["pvalue"] = [None] * len(gaia_valid)

    return gaia_valid


def alert_on_flare(
    previous_flux: u.Quantity[u.Jy],
    new_flux: u.Quantity[u.Jy],
    ratio_threshold: float = 5.0,
):
    """Return ``True`` if the new flux exceeds the previous flux by a large ratio.

    Parameters
    ----------
    previous_flux : Quantity[Jy]
        Reference flux (e.g. catalog value).
    new_flux : Quantity[Jy]
        Newly measured flux.
    ratio_threshold : float, optional
        Ratio ``new_flux / previous_flux`` above which a flare is flagged
        (default 5.0).

    Returns
    -------
    bool
        ``True`` if the ratio exceeds the threshold; ``False`` if either flux
        is ``None``.
    """
    if previous_flux is None or new_flux is None:
        return False
    return new_flux.to(u.Jy).value / previous_flux.to(u.Jy).value > ratio_threshold


def sift(
    extracted_sources: list[MeasuredSource],
    catalog_sources: list[RegisteredSource] | None = None,
    input_map: ProcessableMap = None,
    radius1Jy: AstroPydanticQuantity[u.arcmin] = 10.0 * u.arcmin,
    min_match_radius: AstroPydanticQuantity[u.arcmin] = 1.5 * u.arcmin,
    ra_jitter: AstroPydanticQuantity[u.arcmin] = 0.0 * u.arcmin,
    dec_jitter: AstroPydanticQuantity[u.arcmin] = 0.0 * u.arcmin,
    source_fluxes: AstroPydanticQuantity[u.Jy] = None,
    map_freq: str | None = None,
    arr: str | None = None,
    cuts: dict = {
        "fwhm": [0.2, 5.0],
        "snr": [5.0, np.inf],
        "observation_mean_time": [1, np.inf],
    },
    crossmatch_with_gaia: bool = True,
    crossmatch_with_million_quasar: bool = True,
    additional_catalogs: dict = {},
    debug: bool = False,
    log: FilteringBoundLogger | None = None,
):
    """Crossmatch extracted sources against catalogs and classify candidates.

    Uses a flux-scaled matching radius (``radius1Jy`` for a 1 Jy source,
    ``min_match_radius`` as the floor) capped at 2 degrees.

    Parameters
    ----------
    extracted_sources : list of MeasuredSource
        Detections to classify.
    catalog_sources : list of RegisteredSource, optional
        Known-source catalog for crossmatching.
    input_map : ProcessableMap, optional
        Map the sources were extracted from (used for FWHM and frequency lookup).
    radius1Jy : Quantity[arcmin], optional
        Matching radius for a 1 Jy source (default 10 arcmin).
    min_match_radius : Quantity[arcmin], optional
        Minimum matching radius (default 1.5 arcmin).
    ra_jitter : Quantity[arcmin], optional
        Extra RA position uncertainty added in quadrature (default 0 arcmin).
    dec_jitter : Quantity[arcmin], optional
        Extra Dec position uncertainty added in quadrature (default 0 arcmin).
    source_fluxes : Quantity[Jy], optional
        Fluxes of extracted sources; inferred from ``extracted_sources`` if
        ``None``.
    map_freq : str, optional
        Frequency band string (e.g. ``"f090"``).
    arr : str, optional
        Array identifier.
    cuts : dict, optional
        Quality cuts as ``{field: [min, max]}``.
    crossmatch_with_gaia : bool, optional
        Query Gaia for unmatched sources (default ``True``).
    crossmatch_with_million_quasar : bool, optional
        Crossmatch with the million-quasar catalog (default ``True``).
    additional_catalogs : dict, optional
        Extra named catalogs for crossmatching.
    debug : bool, optional
        Enable verbose debug logging (default ``False``).
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    source_candidates : list of MeasuredSource
        Detections matched to a known catalog source.
    transient_candidates : list of MeasuredSource
        Detections with no catalog match.
    noise_candidates : list of MeasuredSource
        Detections rejected by quality cuts.
    """
    from ..utils.utils import get_fwhm, radec_to_str_name

    log = log if log else get_logger()
    log = log.bind(func_name="sift")

    fwhm = get_fwhm(
        map_freq, arr=arr, instrument=input_map.instrument if input_map else None
    )
    ## apply cuts in units of fwhm
    if "fwhm" in cuts:
        cuts["fwhm_ra"] = [cuts["fwhm"][0] * fwhm, cuts["fwhm"][1] * fwhm]
        cuts["fwhm_dec"] = [cuts["fwhm"][0] * fwhm, cuts["fwhm"][1] * fwhm]

    ## log the parameters
    log.info(
        "sift.start",
        radius1Jy=radius1Jy,
        min_match_radius=min_match_radius,
        ra_jitter=ra_jitter,
        dec_jitter=dec_jitter,
        cuts=cuts,
        crossmatch_with_gaia=crossmatch_with_gaia,
        crossmatch_with_million_quasar=crossmatch_with_million_quasar,
    )

    angle_coords = u.deg

    extracted_ra = np.asarray(
        [es.ra.to_value(angle_coords) for es in extracted_sources]
    )
    extracted_dec = np.asarray(
        [es.dec.to_value(angle_coords) for es in extracted_sources]
    )

    if source_fluxes is None:
        source_fluxes = (
            np.asarray([es.flux.to(u.Jy).value for es in extracted_sources]) * u.Jy
        )

    if catalog_sources is None:
        isin_cat = np.zeros(len(extracted_ra), dtype=bool)
        catalog_match = [] * len(extracted_ra)
        log.warning("sift.catalog_sources.not_found")
    else:
        crossmatch_radius = np.minimum(
            np.maximum(source_fluxes.to(u.Jy).value * radius1Jy, min_match_radius),
            120 * u.arcmin,
        )

        cat_ra = np.asarray([cs.ra.to_value(angle_coords) for cs in catalog_sources])
        cat_dec = np.asarray([cs.dec.to_value(angle_coords) for cs in catalog_sources])
        isin_cat, catalog_match = crossmatch_mask(
            np.stack([extracted_dec, extracted_ra], axis=1),
            np.stack([cat_dec, cat_ra], axis=1),
            list(crossmatch_radius.to(u.arcmin).value),
            mode="closest",
            return_matches=True,
        )
        log.info("sift.catalog_sources.crossmatched")

    if crossmatch_with_million_quasar:
        isin_mq_cat, mq_catalog_match = crossmatch_with_million_quasar_catalog(
            extracted_sources,
            mq_catalog=additional_catalogs.get("million_quasar", None),
            log=log,
        )

    source_candidates = []
    transient_candidates = []
    noise_candidates = []
    for source, cand_pos in enumerate(zip(extracted_ra, extracted_dec)):
        source_measurement = extracted_sources[source]
        source_string_name = radec_to_str_name(cand_pos[0], cand_pos[1])

        if isin_cat[source]:
            cm = catalog_sources[catalog_match[source][0][1]]
            source_measurement.crossmatches = cm.crossmatches
            source_measurement.crossmatches[0].angular_separation = angular_separation(
                source_measurement.ra, source_measurement.dec, cm.ra, cm.dec
            )
        else:
            source_measurement.crossmatches = []

        if isin_mq_cat[source]:
            source_measurement.crossmatches.append(
                CrossMatch(
                    mq_catalog_match[source].source_id, mq_catalog_match[source].pvalue
                )
            )
        log.debug(
            "sift.candidate_info",
            source_name=source_string_name,
            crossmatches=source_measurement.crossmatches,
            flux=catalog_sources[catalog_match[source][0][1]].flux
            if isin_cat[source]
            else None,
            forced_photometry_info=source_measurement,
        )
        ## get the ra,dec uncertainties as quadrature sum of sigma/sqrt(SNR) and any pointing uncertainty (ra,dec jitter)
        if source_measurement.err_ra is None or source_measurement.err_dec is None:
            source_measurement.err_ra = fwhm / np.sqrt(
                source_measurement.flux / source_measurement.err_flux
            )
            source_measurement.err_dec = fwhm / np.sqrt(
                source_measurement.flux / source_measurement.err_flux
            )

        source_measurement.err_ra = np.sqrt(source_measurement.err_ra**2 + ra_jitter**2)
        source_measurement.err_dec = np.sqrt(
            source_measurement.err_dec**2 + dec_jitter**2
        )

        is_cut = get_cut_decision(source_measurement, cuts, debug=debug)

        if isin_cat[source] and not is_cut:
            log.debug(
                "sift.source_crossmatch",
                source=source,
                crossmatches=source_measurement.crossmatches,
                flux=catalog_sources[catalog_match[source][0][1]].flux,
            )

            if alert_on_flare(
                catalog_sources[catalog_match[source][0][1]].flux,
                source_measurement.flux,
            ):
                log.info(
                    "sift.source_crossmatch.flare_alert",
                    source_name=source_string_name,
                    crossmatches=source_measurement.crossmatches,
                    flux=catalog_sources[catalog_match[source][0][1]].flux,
                    cand_flux=source_measurement.flux,
                )
                transient_candidates.append(source_measurement)
            else:
                source_candidates.append(source_measurement)
        elif is_cut or not np.isfinite(source_measurement.flux):
            noise_candidates.append(source_measurement)
            log.debug(
                "sift.source_cut",
                source_name=source_string_name,
                crossmatches=source_measurement.crossmatches,
                flux=catalog_sources[catalog_match[source][0][1]].flux
                if isin_cat[source]
                else None,
                forced_photometry_info=source_measurement,
            )
        else:
            if crossmatch_with_gaia:
                ## if running on compute node, can't access internet, so can't query gaia.
                try:
                    gaia_match_result = gaia_match(
                        source_measurement,
                        maxsep=fwhm,
                    )
                    if gaia_match_result:
                        ## just grab the first result
                        if len(gaia_match_result["designation"]) > 0:
                            # TODO: update this to crossmatch objects
                            source_measurement.update_crossmatches(
                                match_names=[gaia_match_result["designation"][0]],
                                match_probabilities=[gaia_match_result["pvalue"][0]],
                            )
                    log = log.bind(gaia_match_result=gaia_match_result)
                    log.info("sift.gaia_match")
                except Exception:
                    log.info("sift.gaia_match.failed")
                    pass
            # TODO: may have to add radec to source name to make it unique
            ## give the transient candidate source a name indicating that it is a transient
            source_measurement.source_id = "-T".join(source_string_name.split("-S"))
            transient_candidates.append(source_measurement)
            log.info("sift.transient_candidate", transient_cand=source_measurement)

    log.info(
        "sift.initial_candidates",
        transient_candidates=len(transient_candidates),
        noise_candidates=len(noise_candidates),
        source_candidates=len(source_candidates),
    )

    transient_candidates, new_noise_candidates = recalculate_local_snr(
        transient_candidates,
        input_map,
        thumb_size=np.maximum(0.25 * u.deg, 5 * fwhm),
        fwhm=fwhm,
        snr_cut=cuts["snr"][0],
    )
    if new_noise_candidates:
        log = log.bind(updated_noise_candidates=len(new_noise_candidates))
        log.info("sift.recalc_snr")
    noise_candidates.extend(new_noise_candidates)

    log.info(
        "sift.final_counts",
        source_candidates=len(source_candidates),
        transient_candidates=len(transient_candidates),
        noise_candidates=len(noise_candidates),
    )
    return source_candidates, transient_candidates, noise_candidates


def get_cut_decision(
    candidate: MeasuredSource,
    cuts: dict | None = None,
    debug: bool = False,
    log: FilteringBoundLogger | None = None,
) -> bool:
    """Decide whether a source candidate should be rejected by quality thresholds.

    Parameters
    ----------
    candidate : MeasuredSource
        Source measurement to evaluate.
    cuts : dict, optional
        Quality cuts as ``{field: [min, max]}``.  Supported keys include
        ``"fwhm"``, ``"snr"``, and ``"observation_mean_time"``.
    debug : bool, optional
        Log details of each cut decision (default ``False``).
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    bool
        ``True`` if the candidate is cut (outside allowed range);
        ``False`` if all cuts are passed.
    """
    log = log if log else get_logger()
    log = log.bind(func_name="get_cut_decision")
    if cuts is None:
        return False

    cut = False
    for c in cuts.keys():
        val = getattr(candidate, c, None)
        if val is None:
            log.debug("get_cut_decision.missing_value", cut_name=c)
            continue
        ## this is stupid, but how i've decided to do it.
        if c == "observation_mean_time":
            val = val.unix
        cut |= (val < cuts[c][0]) | (val > cuts[c][1])
        if debug and (val < cuts[c][0]) | (val > cuts[c][1]):
            log.debug(
                "get_cut_decision.cut_made",
                measured_value=val,
                cut_name=c,
                cut_min=cuts[c][0],
                cut_max=cuts[c][1],
            )
    return cut


def recalculate_local_snr(
    transient_candidates: list[MeasuredSource],
    imap: ProcessableMap,
    thumb_size: u.Quantity = 0.25 * u.deg,
    fwhm: u.Quantity = 2.2 * u.arcmin,
    snr_cut: float = 5.0,
    ratio_cut: float = 10.0,
    log: FilteringBoundLogger | None = None,
):
    """Recalculate per-source SNR using local map noise and re-classify.

    Computes the RMS of a thumbnail submap (with the source core masked) and
    rejects candidates whose new SNR is too low or whose old/new SNR ratio
    exceeds ``ratio_cut`` (indicating an unexpectedly noisy region).

    Parameters
    ----------
    transient_candidates : list of MeasuredSource
        Candidate transient sources to re-evaluate.
    imap : ProcessableMap
        Map from which to extract thumbnails.
    thumb_size : Quantity[deg], optional
        Size of the thumbnail extracted around each source (default 0.25 deg).
    fwhm : Quantity[arcmin], optional
        Beam FWHM used to compute the source-masking radius (default 2.2 arcmin).
    snr_cut : float, optional
        Minimum SNR for a candidate to survive re-evaluation (default 5.0).
    ratio_cut : float, optional
        Old/new SNR ratio above which a candidate is demoted to noise
        (default 10.0).
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    updated_transient_candidates : list of MeasuredSource
        Candidates that pass the local SNR check.
    updated_noise_candidates : list of MeasuredSource
        Candidates rejected by the local SNR check.
    """
    from ..maps.maps import get_submap
    from ..utils.utils import get_pix_from_peak_to_noise

    log = log if log else get_logger()
    log = log.bind(func_name="recalculate_local_snr")
    updated_transient_candidates = []
    updated_noise_candidates = []
    ## same mask for each one
    mask = None
    for candidate in transient_candidates:
        # Extract a thumbnail of the transient source
        ra, dec = candidate.ra, candidate.dec
        thumbnail = get_submap(
            imap.flux,
            ra.to(u.deg).value,
            dec.to(u.deg).value,
            size_deg=thumb_size.to(u.deg).value,
        )

        # Mask the center out to 2 times the FWHM
        # if isinstance(mask,type(None)):
        center = tuple(int(t / 2) for t in thumbnail.shape)
        fwhm_pix = (
            fwhm.to(u.arcmin).value
            / u.Quantity(abs(thumbnail.wcs.wcs.cdelt[0]), thumbnail.wcs.wcs.cunit[0])
            .to(u.arcmin)
            .value
        )

        mask_radius = get_pix_from_peak_to_noise(
            candidate.flux.to(u.Jy).value,
            candidate.flux.to(u.Jy).value / candidate.snr,
            fwhm_pix=fwhm_pix,
        )[0]
        mask_radius *= np.sqrt(2)  ## for matched filter size
        Y, X = np.ogrid[: thumbnail.shape[0], : thumbnail.shape[1]]
        dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)
        mask = dist_from_center >= mask_radius

        # Calculate the RMS noise of the unmasked region
        unmasked_flux = thumbnail[mask > 0]
        rms_noise = np.nanstd(unmasked_flux)
        # Recalculate the SNR using the new RMS noise
        old_snr = candidate.snr
        candidate.snr = candidate.flux.to(imap.flux_units).value / rms_noise
        new_snr = candidate.snr
        snr_ratio = old_snr / new_snr
        # print('oldsnr: %.1f, newsnr: %.1f, ratio o/n: %.2f, --- flux: %.1f,err_flux: %.1f'%(old_snr,new_snr,snr_ratio,candidate.flux.to_value(imap.flux_units),rms_noise))
        if candidate.snr > snr_cut and snr_ratio < ratio_cut and np.isfinite(snr_ratio):
            updated_transient_candidates.append(candidate)
        else:
            updated_noise_candidates.append(candidate)

    return updated_transient_candidates, updated_noise_candidates


def crossmatch_position_and_flux(
    injected_sources: list,
    recovered_sources: list,
    flux_threshold: float = 0.01,
    fractional_flux: float = 0.01,
    spatial_threshold: float = 0.05,
    fail_unmatched: bool = False,
    fail_flux_mismatch: bool = False,
    log=None,
):
    """Verify spatial and flux recovery of injected sources against recovered sources.

    Parameters
    ----------
    injected_sources : list of SimulatedSource
        Sources injected into the map.
    recovered_sources : list of MeasuredSource
        Sources recovered from the map.
    flux_threshold : float, optional
        Absolute flux tolerance in Jy (default 0.01).
    fractional_flux : float, optional
        Fractional flux tolerance; effective threshold is
        ``sqrt(flux_threshold**2 + (fractional_flux * injected_flux)**2)``
        (default 0.01).
    spatial_threshold : float, optional
        Spatial matching radius in degrees (default 0.05).
    fail_unmatched : bool, optional
        If ``True``, log an error when injected sources have no spatial match
        (default ``False``).
    fail_flux_mismatch : bool, optional
        If ``True``, raise ``ValueError`` when recovered fluxes are mismatched
        (default ``False``).
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    matched_mask : ndarray of bool
        ``True`` for each injected source that has a spatial match.
    similar_fluxes : list of bool
        ``True`` for each matched pair whose fluxes agree within tolerance.
    """
    from pixell.utils import arcmin, degree

    log.bind(func_name="crossmatch_position_and_flux")

    # Convert injected and recovered sources to numpy arrays of positions
    injected_positions = np.array([[src.dec, src.ra] for src in injected_sources])
    recovered_positions = np.array([[src.dec, src.ra] for src in recovered_sources])
    log.info(
        "crossmatch_position_and_flux.positions",
        injected_positions=injected_positions,
        recovered_positions=recovered_positions,
    )
    # Perform crossmatch to find matches
    matched_mask, matches = crossmatch_mask(
        injected_positions,
        recovered_positions,
        radius=spatial_threshold * degree / arcmin,
        mode="closest",
        return_matches=True,
    )

    if np.sum(np.logical_not(matched_mask)) > 0 and fail_unmatched:
        log.error(
            "crossmatch_position_and_flux.unmatched_sources",
            unmatched_count=np.sum(np.logical_not(matched_mask)),
        )
        print(ValueError("Some injected sources were not recovered."))

    # Check flux similarity for matched sources
    similar_fluxes = []
    in_flux = []
    out_flux = []
    in_ra = []
    out_ra = []
    in_dec = []
    out_dec = []

    for i, match in enumerate(matches):
        if match:  # If there is a match
            injected_flux = injected_sources[i].flux
            recovered_flux = recovered_sources[match[0][1]].flux
            in_flux.append(injected_flux)
            out_flux.append(recovered_flux)
            in_ra.append(injected_sources[i].ra)
            out_ra.append(recovered_sources[match[0][1]].ra)
            in_dec.append(injected_sources[i].dec)
            out_dec.append(recovered_sources[match[0][1]].dec)
            # get input and recovered ra,dec
            if abs(injected_flux - recovered_flux) <= np.sqrt(
                flux_threshold**2 + (fractional_flux * injected_flux) ** 2
            ):
                similar_fluxes.append(True)
            else:
                similar_fluxes.append(False)
        else:
            ij = injected_sources[i]
            log.debug(
                "crossmatch_position_and_flux.no_match",
                source_index=i,
                ra=ij.ra,
                dec=ij.dec,
                flux=ij.flux,
                fwhm_a=ij.fwhm_a,
                fwhm_b=ij.fwhm_b,
            )
            similar_fluxes.append(False)
    log.info(
        "crossmatch_position_and_flux.similar_fluxes", similar_fluxes=similar_fluxes
    )
    if np.sum(np.logical_not(similar_fluxes)) > 0:
        log.warning(
            "crossmatch_position_and_flux.failed_matches",
            failed_count=np.sum(np.logical_not(similar_fluxes)),
        )

    log.warning("crossmatch_position_and_flux.ascii_plot_because_I_was_bored")
    from ..utils.plot import ascii_scatter, ascii_vertical_histogram

    meandec = np.mean(in_dec)
    delta_ra = (
        np.subtract(in_ra, out_ra) * np.cos(np.degrees(meandec)) * degree / arcmin
    )
    delta_dec = np.subtract(in_dec, out_dec) * degree / arcmin
    print("\n###########################################################\n")
    ascii_scatter(delta_ra, delta_dec)
    print("RA offset (X-axis) vs DEC offset (Y-axis) in arcmin")
    print("\n###########################################################")
    ascii_scatter(in_flux, out_flux)
    print("Injected flux (X-axis) vs Recovered flux (Y-axis) in Jy")

    print("\n###########################################################")

    ascii_vertical_histogram(
        1e3 * np.subtract(in_flux, out_flux),
        bin_width=10,
        min_val=-100.0,
        max_val=100.0,
        height=20,
    )
    print("Recovered Flux difference (X-axis) in mJy\n")
    print("###########################################################")
    print("\n")
    if np.sum(np.logical_not(similar_fluxes)) > 0 and fail_flux_mismatch:
        raise ValueError("Some injected sources have mismatched fluxes.")

    return matched_mask, similar_fluxes


def n_wise_crossmatch(
    matches: list[list[tuple]], candidates: list[list[MeasuredSource]]
) -> dict[str, list[tuple[str, str]]]:
    """Merge pairwise crossmatch results into a global source-identity mapping.

    Parameters
    ----------
    matches : list of list of tuple
        Pairwise match lists; each inner list corresponds to a map pair and
        contains ``(i, j)`` index tuples.
    candidates : dict
        Map-ID-keyed dict of ``MeasuredSource`` lists.

    Returns
    -------
    dict
        Mapping from UUID source key to a list of ``(map_id, measurement_id)``
        tuples for all observations of that source.
    """
    map_pairs = list(combinations(candidates.keys(), 2))
    match_by_pair = {
        pair: {i: j for match in match_list for i, j in match}
        for match_list, pair in zip(matches, map_pairs)
    }

    full_matches: dict = {}
    node_to_key: dict = {}

    for (map_id_1, map_id_2), d in match_by_pair.items():
        for i, j in d.items():
            k1 = node_to_key.get((map_id_1, i))
            k2 = node_to_key.get((map_id_2, j))
            if k1 is None and k2 is None:
                # this sources hasn't been seen before, assign a new ID
                new_key = str(uuid7.create())
                full_matches[new_key] = {map_id_1: i, map_id_2: j}
                node_to_key[(map_id_1, i)] = new_key
                node_to_key[(map_id_2, j)] = new_key
            elif k1 is None:
                full_matches[k2][map_id_1] = i
                node_to_key[(map_id_1, i)] = k2
            elif k2 is None:
                full_matches[k1][map_id_2] = j
                node_to_key[(map_id_2, j)] = k1

    output: dict[str, list[tuple[str, str]]] = {}
    for key, indices_by_result in full_matches.items():
        output[key] = []
        for map_id, idx in indices_by_result.items():
            candidates[map_id][idx].source_id = key
            measurement_id = candidates[map_id][idx].measurement_id
            output[key].append((map_id, measurement_id))
    return output
