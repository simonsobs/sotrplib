import numpy as np
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
    """Determines if source matches with masked objects

    Args:
        sources: np.array of sources [[dec, ra]] in deg
        crosscat: catalog of masked objects [[dec, ra]] in deg
        radius: radius to search for matches in arcmin, float or list of length sources.
        mode: return `all` pairs, or just `closest`
    Returns:
        mask column for sources, 1 matches with at least one source, 0 no matches

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
    """
    Crossmatch the source candidates with the million quasar catalog.
    Returns a list of source candidates that are matched with the million quasar catalog
    and a dict with the crossmatch information.

    Parameters:
    - mq_catalog: The million quasar catalog as an astropy table.
    - sources: catalog of source positions [[dec, ra]] in deg (same format as crossmatch_mask).
    - match_threshold: The matching radius.

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
    """
    Check if the new flux is significantly larger than the previous flux.
    If the ratio of new flux to previous flux is greater than the threshold, return True.
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
    },
    crossmatch_with_gaia: bool = True,
    crossmatch_with_million_quasar: bool = True,
    additional_catalogs: dict = {},
    debug: bool = False,
    log: FilteringBoundLogger | None = None,
):
    from ..utils.utils import get_fwhm, radec_to_str_name

    """
     Perform crossmatching of extracted sources from `extract_sources` and the cataloged sources.
     Return lists of dictionaries containing each source which matches the catalog, may be noise, or appears to be a transient.

     Uses a flux-based matching radius with `radius1Jy` the radius, in arcmin, for a 1Jy source and 
     `min_match_radius` the radius, in arcmin, for a zero flux source, up to a max of 2 degrees.

     Args:
       extracted_sources:dict
           sources returned from extract_sources function
       catalog_sources:list
           source catalog as list of Registered objects
       radius1Jy:float=30.0
           matching radius for a 1Jy source, arcmin
       min_match_radius:float=1.5
           minimum matching radius, i.e. for a zero flux source, arcmin
       source_fluxes:list = None,
           a list of the fluxes of the extracted sources, if None will pull it from `extracted_sources` dict. 
       cuts: dict
           snr cut and a simple cut on fwhm, in units of fwhm, outside of range is considered noise.
       ra_jitter:float=0.0
              jitter in the ra direction, in arcmin, to add to the uncertainty of the source position.
       dec_jitter:float=0.0
              jitter in the dec direction, in arcmin, to add to the uncertainty of the source position. 
       crossmatch_with_gaia:bool=True
              if True, will crossmatch with gaia catalog and add the gaia source name to the source_id.
       crossmatch_with_million_quasar:bool=True
                if True, will crossmatch with the million quasar catalog and add the source name to the source_id.
       additional_catalogs:dict={}
                a dictionary of additional catalogs to crossmatch with, in the form of {name:catalog}.
                for example: {"million_quasar":mq_catalog}
                where catalog is loaded from source_catalog.py
                
     Returns:
        source_candidates, transient_candidates, noise_candidates : list
            list of dictionaries with information about the detected source.

    """
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
    cuts: dict = {},
    debug: bool = False,
    log: FilteringBoundLogger | None = None,
) -> bool:
    """
    cuts contains dictionary with key values describing the cuts.

    acceptable cut keys : ['fwhm']

    each cut key has [min,max] value on which to make cuts.

    for example,
    if i want a source iwth measured fwhm >0.5  and less than 5 times nominal fwhm,
    cuts={'fwhm':[0.5,5.0]}
    cut = (candidate['fwhm']<cuts['fwhm'][0]) | (candidate['fwhm']>cuts['fwhm'][1])

    Returns:

    True : cut source
    False : within ranges
    """
    log = log if log else get_logger()
    log = log.bind(func_name="get_cut_decision")
    cut = False
    for c in cuts.keys():
        val = getattr(candidate, c, None)
        if val is None:
            log.debug("get_cut_decision.missing_value", cut_name=c)
            continue
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
    """
    Recalculate the local SNR for each transient source.

    Parameters:
    - transient_candidates: List of transient source candidates.
    - imap: The map data object.
    - thumb_size_deg: The size of the thumbnail in degrees.
    - fwhm_arcmin: The band full width at half maximum in arcmin.
    - snr_cut: The SNR cut to use for the new local noise.
    - ratio_cut: The ratio of the old SNR to the new SNR above which to cut.

    assumes that if the new snr is significantly different from the old snr, the region is noisier than expected.
    empircally 30% change seems to indicate a noisy region....
    todo: however because an unmasked source creates filtering wings, the snr can be significantly different.
          so for now, set the ratio_cut fairly high to avoid cutting real transients.

    Returns:
    - updated_transient_candidates: List of transient source candidates with updated SNR.
    - updated_noise_candidates: List of noise source candidates with updated SNR.
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
    """
    Crossmatch the injected sources and transients with the recovered ones.
    Ensure the recovered fluxes are within 3 sigma of the injected fluxes.
    Count and report the fraction of failures.

    Parameters:
    - injected_sources: List of injected static sources.
    - recovered_sources: List of recovered static sources.
    - flux_threshold: Threshold for similar fluxes (Jy).
    - spatial_threshold: Spatial threshold for matching (degrees).

    Returns:
    - A dictionary containing the failure fractions for sources and transients.
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
