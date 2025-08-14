import math
import os
import os.path as op
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from pixell import enmap
from pixell import utils as pixell_utils
from scipy import spatial

from ..sources.sources import SourceCandidate


def crossmatch_mask(
    sources, crosscat, radius: float, mode: str = "all", return_matches: bool = False
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
    mq_catalog,
    sources,
    radius_arcmin: float = 0.5,
):
    """
    Crossmatch the source candidates with the million quasar catalog.
    Returns a list of source candidates that are matched with the million quasar catalog
    and a dict with the crossmatch information.

    Parameters:
    - mq_catalog: The million quasar catalog as an astropy table.
    - sources: catalog of source positions [[dec, ra]] in deg (same format as crossmatch_mask).
    - radius_arcmin: The matching radius in arcmin.

    """
    isin_cat, catalog_match = crossmatch_mask(
        sources,
        np.asarray([mq_catalog["decDeg"], mq_catalog["RADeg"]]).T,
        radius_arcmin,
        mode="closest",
        return_matches=True,
    )
    match_info = {}
    for i in range(len(isin_cat)):
        match_info[i] = {}
        if isin_cat[i]:
            cm = catalog_match[i][0][1]
            for key in mq_catalog.keys():
                match_info[i][key] = mq_catalog[key][cm]
            match_info[i]["pvalue"] = None  ## need to calculate probability.
    return isin_cat, match_info


def gaia_match(
    cand: SourceCandidate,
    maxmag: float = 16,
    maxsep_deg: float = 1 * pixell_utils.degree,
    parallax_required=True,
    mag_key="phot_g_mean_mag",
    sep_key="dist",
):
    from ..source_catalog.query_tools import cone_query_gaia

    gaia_results = cone_query_gaia(cand.ra, cand.dec, radius_arcmin=cand.fwhm)

    if gaia_results:
        parallax = (
            (np.isfinite(gaia_results["parallax"]))
            if parallax_required
            else (np.ones(len(gaia_results["parallax"]), dtype=bool))
        )
        gaia_valid = gaia_results[
            (gaia_results[mag_key] < maxmag)
            & parallax
            & (gaia_results[sep_key] < maxsep_deg)
        ]
    else:
        gaia_valid = {}
    ## calculate pvalue, sort on pvalue.
    ## need to implement.
    if len(gaia_valid) > 0:
        gaia_valid["pvalue"] = [None] * len(gaia_valid)

    return gaia_valid


def alert_on_flare(
    previous_flux: float,
    new_flux: float,
    ratio_threshold: float = 5.0,
):
    """
    Check if the new flux is significantly larger than the previous flux.
    If the ratio of new flux to previous flux is greater than the threshold, return True.
    """
    return new_flux / previous_flux > ratio_threshold


def sift(
    extracted_sources,
    catalog_sources,
    imap: enmap.ndmap = None,
    radius1Jy: float = 30.0,
    min_match_radius: float = 1.5,
    ra_jitter: float = 0.0,
    dec_jitter: float = 0.0,
    source_fluxes: list = None,
    map_id: str = "",
    map_freq: str = None,
    arr: str = None,
    cuts: dict = {
        "fwhm": [0.5, 5.0],
        "snr": [5.0, np.inf],
    },
    crossmatch_with_gaia=True,
    crossmatch_with_million_quasar=True,
    additional_catalogs={},
    debug=False,
    log=None
):
    from ..utils.utils import radec_to_str_name
    from ..utils.utils import get_fwhm
    """
     Perform crossmatching of extracted sources from `extract_sources` and the cataloged sources.
     Return lists of dictionaries containing each source which matches the catalog, may be noise, or appears to be a transient.

     Uses a flux-based matching radius with `radius1Jy` the radius, in arcmin, for a 1Jy source and 
     `min_match_radius` the radius, in arcmin, for a zero flux source, up to a max of 2 degrees.

     Args:
       extracted_sources:dict
           sources returned from extract_sources function
       catalog_sources:list
           source catalog as list of SourceCandidate objects
       radius1Jy:float=30.0
           matching radius for a 1Jy source, arcmin
       min_match_radius:float=1.5
           minimum matching radius, i.e. for a zero flux source, arcmin
       source_fluxes:list = None,
           a list of the fluxes of the extracted sources, if None will pull it from `extracted_sources` dict. 
       fwhm_cut = 5.0
           a simple cut on fwhm, in arcmin, above which something is considered noise.
       ra_jitter:float=0.0
              jitter in the ra direction, in arcmin, to add to the uncertainty of the source position.
       dec_jitter:float=0.0
              jitter in the dec direction, in arcmin, to add to the uncertainty of the source position. 
       crossmatch_with_gaia:bool=True
              if True, will crossmatch with gaia catalog and add the gaia source name to the sourceID.
       crossmatch_with_million_quasar:bool=True
                if True, will crossmatch with the million quasar catalog and add the source name to the sourceID.
       additional_catalogs:dict={}
                a dictionary of additional catalogs to crossmatch with, in the form of {name:catalog}.
                for example: {"million_quasar":mq_catalog}
                where catalog is loaded from source_catalog.py
                
     Returns:
        source_candidates, transient_candidates, noise_candidates : list
            list of dictionaries with information about the detected source.

    """
    
    fwhm_arcmin = get_fwhm(map_freq, arr=arr)
    
    if isinstance(extracted_sources, type(None)):
        return [], [], []

    if isinstance(source_fluxes, type(None)):
        source_fluxes = np.asarray(
            [extracted_sources[f]["peakval"] for f in extracted_sources]
        )

    extracted_ra = np.asarray([extracted_sources[f]["ra"] for f in extracted_sources])
    extracted_dec = np.asarray([extracted_sources[f]["dec"] for f in extracted_sources])

    if not catalog_sources:
        isin_cat = np.zeros(len(extracted_ra), dtype=bool)
        catalog_match = [] * len(extracted_ra)
    else:
        crossmatch_radius = np.minimum(
            np.maximum(source_fluxes * radius1Jy, min_match_radius), 120
        )
        isin_cat, catalog_match = crossmatch_mask(
            np.asarray(
                [
                    extracted_dec / pixell_utils.degree,
                    extracted_ra / pixell_utils.degree,
                ]
            ).T,
            np.asarray([catalog_sources["decDeg"], catalog_sources["RADeg"]]).T,
            list(crossmatch_radius),
            mode="closest",
            return_matches=True,
        )
    log.info("pipeline.sift.crossmatched_to_catalog")
    if crossmatch_with_million_quasar:
        if "million_quasar" not in additional_catalogs:
            if debug:
                print("No million quasar catalog provided... cant do crossmatch")
            isin_mq_cat = np.zeros(len(extracted_ra), dtype=bool)
            mq_catalog_match = [] * len(extracted_ra)
        else:
            mq_catalog = additional_catalogs["million_quasar"]
            if mq_catalog is not None:
                isin_mq_cat, mq_catalog_match = crossmatch_with_million_quasar_catalog(
                    mq_catalog,
                    np.asarray(
                        [
                            extracted_dec / pixell_utils.degree,
                            extracted_ra / pixell_utils.degree,
                        ]
                    ).T,
                    radius_arcmin=0.5,
                )
        log.info("pipeline.sift.crossmatched_to_milliquas")

    source_candidates = []
    transient_candidates = []
    noise_candidates = []
    for source, cand_pos in enumerate(zip(extracted_ra, extracted_dec)):
        forced_photometry_info = extracted_sources[source]
        source_string_name = radec_to_str_name(
            cand_pos[0] / pixell_utils.degree, cand_pos[1] / pixell_utils.degree
        )

        if isin_cat[source]:
            crossmatch_name = catalog_sources["name"][catalog_match[source][0][1]]
        else:
            crossmatch_name = ""

        log.debug("pipeline.sift.candidate_info",
                     source_name=source_string_name,
                     crossmatch_name=crossmatch_name,
                     flux=catalog_sources["fluxJy"][catalog_match[source][0][1]] if isin_cat[source] else None,
                     forced_photometry_info=forced_photometry_info
                     )
        ## get the ra,dec uncertainties as quadrature sum of sigma/sqrt(SNR) and any pointing uncertainty (ra,dec jitter)
        if (
            "err_ra" not in forced_photometry_info
            or "err_dec" not in forced_photometry_info
        ):
            forced_photometry_info["err_ra"] = (
                pixell_utils.fwhm
                * fwhm_arcmin
                * pixell_utils.arcmin
                / np.sqrt(forced_photometry_info["peaksig"])
            )
            forced_photometry_info["err_dec"] = (
                pixell_utils.fwhm
                * fwhm_arcmin
                * pixell_utils.arcmin
                / np.sqrt(forced_photometry_info["peaksig"])
            )
            forced_photometry_info["err_ra"] = np.sqrt(
                forced_photometry_info["err_ra"] ** 2 + ra_jitter**2
            )
            forced_photometry_info["err_dec"] = np.sqrt(
                forced_photometry_info["err_dec"] ** 2 + dec_jitter**2
            )

        ## peakval is the max value within the kron radius while kron_flux is the integrated flux (i.e. assuming resolved)
        ## since filtered maps are convolved w beam, use max pixel value.
        cand = SourceCandidate(
            ra=cand_pos[0] / pixell_utils.degree % 360,
            dec=cand_pos[1] / pixell_utils.degree,
            err_ra=forced_photometry_info["err_ra"] / pixell_utils.degree,
            err_dec=forced_photometry_info["err_dec"] / pixell_utils.degree,
            flux=forced_photometry_info["peakval"],
            err_flux=forced_photometry_info["peakval"]
            / forced_photometry_info["peaksig"],
            kron_flux=forced_photometry_info["kron_flux"],
            kron_fluxerr=forced_photometry_info["kron_fluxerr"],
            kron_radius=forced_photometry_info["kron_radius"],
            snr=forced_photometry_info["peaksig"],
            freq=str(map_freq),
            arr=arr,
            ctime=forced_photometry_info["time"],
            map_id=map_id,
            sourceID=source_string_name,
            matched_filtered=True,
            renormalized=True,
            catalog_crossmatch=isin_cat[source],
            ellipticity=forced_photometry_info["ellipticity"],
            elongation=forced_photometry_info["elongation"],
            fwhm=forced_photometry_info["fwhm"],
            fwhm_a=forced_photometry_info["semimajor_sigma"] * 2.355,
            fwhm_b=forced_photometry_info["semiminor_sigma"] * 2.355,
            orientation=forced_photometry_info["orientation"],
        )
        cand.update_crossmatches(
            match_names=[crossmatch_name], match_probabilities=[None]
        )
        if isin_mq_cat[source]:
            mq_name_columns = ["NAME", "XNAME", "RNAME"]
            mq_names = []
            mq_probs = []
            for mq_name_column in mq_name_columns:
                if mq_name_column in mq_catalog_match[source]:
                    mq_names.append(mq_catalog_match[source][mq_name_column])
                    mq_probs.append(mq_catalog_match[source]["pvalue"])
            cand.update_crossmatches(
                match_names=mq_names,
                match_probabilities=mq_probs,  ## need to calculate probability.
            )
        log = log.bind(cand=cand)
        ## do sifting operations here...
        is_cut = get_cut_decision(cand, cuts, debug=debug)
    
        if isin_cat[source] and not is_cut:
            log.debug(
                "pipeline.sift.source_crossmatch",
                source=source,
                crossmatch_name=crossmatch_name,
                flux=catalog_sources["fluxJy"][catalog_match[source][0][1]],
            )
            
            if alert_on_flare(
                catalog_sources["fluxJy"][catalog_match[source][0][1]], cand.flux
            ):
                print(
                    "ALERT: %s has increased flux from %.2f to %.2f Jy"
                    % (
                        crossmatch_name,
                        catalog_sources["fluxJy"][catalog_match[source][0][1]],
                        cand.flux,
                    )
                )
                log.info("pipeline.sift.source_crossmatch.flare_alert",
                            source_name=source_string_name,
                            crossmatch_name=crossmatch_name,
                            flux=catalog_sources["fluxJy"][catalog_match[source][0][1]],
                            cand_flux=cand.flux
                            )
                transient_candidates.append(cand)
            else:
                source_candidates.append(cand)
        elif (
            is_cut
            or not np.isfinite(forced_photometry_info["kron_flux"])
            or not np.isfinite(forced_photometry_info["kron_fluxerr"])
            or not np.isfinite(forced_photometry_info["peakval"])
        ):
            noise_candidates.append(cand)
            log.debug("pipeline.sift.cut_candidate",
                         source_name=source_string_name,
                         crossmatch_name=crossmatch_name,
                         flux=catalog_sources["fluxJy"][catalog_match[source][0][1]] if isin_cat[source] else None,
                         forced_photometry_info=forced_photometry_info
                         )
        else:
            if crossmatch_with_gaia:
                ## if running on compute node, can't access internet, so can't query gaia.
                try:
                    gaia_match_result = gaia_match(
                        cand,
                        maxsep_deg=cand.fwhm
                        * pixell_utils.arcmin
                        / pixell_utils.degree,
                    )
                    if gaia_match_result:
                        ## just grab the first result
                        if len(gaia_match_result["designation"]) > 0:
                            cand.update_crossmatches(
                                match_names=[gaia_match_result["designation"][0]],
                                match_probabilities=[gaia_match_result["pvalue"][0]],
                            )
                    log = log.bind(gaia_match_result=gaia_match_result)
                    log.info("pipeline.sift.gaia_match")
                except Exception:
                    log.info("pipeline.sift.gaia_match.failed")
                    pass

            ## give the transient candidate source a name indicating that it is a transient
            cand.sourceID = "-T".join(cand.sourceID.split("-S"))
            transient_candidates.append(cand)
            log = log.bind(transient_cand=cand)
            log.info("pipeline.sift.transient_candidate")


    if isinstance(imap, enmap.ndmap):
        transient_candidates, new_noise_candidates = recalculate_local_snr(
            transient_candidates,
            imap,
            thumb_size_deg=0.5,
            fwhm_arcmin=fwhm_arcmin,
            snr_cut=cuts["snr"][0],
        )
        if new_noise_candidates:
            log = log.bind(new_noise_candidates=new_noise_candidates)
            log.info("pipeline.sift.recalc_snr")
        noise_candidates.extend(new_noise_candidates)

    log = log.bind(source_candidates=len(source_candidates),
                   transient_candidates=len(transient_candidates),
                   noise_candidates=len(noise_candidates)
                   )
    log.info("pipeline.sift.final_counts")
    return source_candidates, transient_candidates, noise_candidates


def get_cut_decision(candidate: SourceCandidate, cuts: dict = {}, debug=False) -> bool:
    """
    cuts contains dictionary with key values describing the cuts.

    acceptable cut keys : ['fwhm']

    each cut key has [min,max] value on which to make cuts.

    for example,
    if i want a source iwth measured fwhm >0.5 arcmin and less than 5 arcmin
    cuts={'fwhm':[0.5,5.0]}
    cut = (candidate['fwhm']<cuts['fwhm'][0]) | (candidate['fwhm']>cuts['fwhm'][1])

    Returns:

    True : cut source
    False : within ranges
    """
    cut = False
    for c in cuts.keys():
        val = getattr(candidate, c)
        cut |= (val < cuts[c][0]) | (val > cuts[c][1])
        if debug and (val < cuts[c][0]) | (val > cuts[c][1]):
            print(
                "cut %s: %.2f < %.2f or %.2f > %.2f"
                % (c, val, cuts[c][0], val, cuts[c][1])
            )
    return cut


def recalculate_local_snr(
    transient_candidates: list,
    imap: enmap.ndmap,
    thumb_size_deg: float = 0.5,
    fwhm_arcmin: float = 2.2,
    snr_cut: float = 5.0,
    ratio_cut: float = 1.3,
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
    empircally 30% change seems to indicate a noisy region.

    Returns:
    - updated_transient_candidates: List of transient source candidates with updated SNR.
    - updated_noise_candidates: List of noise source candidates with updated SNR.
    """
    from pixell.utils import arcmin, degree

    from ..maps.maps import get_submap
    from ..utils.utils import get_pix_from_peak_to_noise

    updated_transient_candidates = []
    updated_noise_candidates = []
    ## same mask for each one
    mask = None
    for candidate in transient_candidates:
        # Extract a thumbnail of the transient source
        ra, dec = candidate.ra, candidate.dec
        thumbnail = get_submap(
            imap,
            ra,
            dec,
            size_deg=thumb_size_deg,
        )

        # Mask the center out to 2 times the FWHM
        # if isinstance(mask,type(None)):
        center = tuple(int(t / 2) for t in thumbnail.shape)
        fwhm_pix = fwhm_arcmin / abs(thumbnail.wcs.wcs.cdelt[0] * degree / arcmin)

        mask_radius = get_pix_from_peak_to_noise(
            candidate.flux,
            candidate.flux / candidate.snr,
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
        candidate.snr = candidate.flux / rms_noise
        new_snr = candidate.snr
        snr_ratio = old_snr / new_snr
        # print('oldsnr: %.1f, newsnr: %.1f, ratio o/n: %.2f, --- flux: %.1f,err_flux: %.1f'%(old_snr,new_snr,snr_ratio,candidate.flux,rms_noise))
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

    # Convert injected and recovered sources to numpy arrays of positions
    injected_positions = np.array([[src.dec, src.ra] for src in injected_sources])
    recovered_positions = np.array([[src.dec, src.ra] for src in recovered_sources])

    # Perform crossmatch to find matches
    matched_mask, matches = crossmatch_mask(
        injected_positions,
        recovered_positions,
        radius=spatial_threshold * degree / arcmin,
        mode="closest",
        return_matches=True,
    )

    if np.sum(np.logical_not(matched_mask)) > 0 and fail_unmatched:
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
            print("no match to source", i)
            ij = injected_sources[i]
            print(
                "%.3f,%3f, flux=%.2f, fwhm a,b=(%.2f,%.2f)"
                % (ij.ra, ij.dec, ij.flux, ij.fwhm_a, ij.fwhm_b)
            )
            similar_fluxes.append(False)

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


####################
## Act crossmatching between wafers / bands etc.
####################


def get_cats(
    path, ctime, return_fnames=False, mfcut=True, renromcut=False, sourcecut=True
):
    keys = [
        "pa4_f220",
        "pa4_f150",
        "pa5_f150",
        "pa5_f090",
        "pa6_f150",
        "pa6_f090",
        "pa7_f040",
        "pa7_f030",
    ]
    cats = []
    fnames = []
    for i, key in enumerate(keys):
        f = op.join(path, f"depth1_{ctime}_{key}_ocat.fits")

        # check if file exists. If not, append empty array
        if not op.exists(f):
            cats.append(
                pd.DataFrame({"ra": [], "dec": [], "flux": [], "snr": [], "ctime": []})
            )
            fnames.append(np.nan)
            continue

        fname = op.join(path, f)
        t = Table.read(fname)
        t = t.to_pandas()
        if len(t) != 0:
            if mfcut:
                t = t[t["mf"].astype(bool)]
            if renromcut:
                t = t[t["renorm"].astype(bool)]
            if sourcecut:
                t = t[~t["source"].astype(bool)]
        # reindex
        t.index = np.arange(len(t))
        cats.append(t)
        fnames.append(fname)

    cats = dict(zip(keys, cats))

    if return_fnames:
        return cats, fnames

    else:
        return cats


def crossmatch_array(cats_in_order, N, radius):
    """crossmatch btw 6 catalogs to get a matched catalog with inds from individual cats, the matched candidate should be detected by at least N arrays

    Args:
        cats_in_order: a list of 2D np.array of source, each with [[dec,ra]] in deg, order is pa4_f220,pa4_f150, pa5_f150,pa5f090,pa6f150,pa6f090. For non-existin cats, put an empty 2D array in the position in list
        N: integer that candidate should be detected by at least N arrays
        radius: radius to search for matches in arcmin
    """
    arr_names = ["pa4", "pa4", "pa5", "pa5", "pa6", "pa6", "pa7", "pa7"]
    # keys = [
    #     "pa4_f220",
    #     "pa4_f150",
    #     "pa5_f150",
    #     "pa5_f090",
    #     "pa6_f150",
    #     "pa6_f090",
    #     "pa7_f040",
    #     "pa7_f030",
    # ]
    rad_deg = radius / 60.0
    skip = 0
    cat_full = []
    for i, cat in enumerate(cats_in_order):
        if cat.shape[0] == 0:
            skip += 1
            continue
        else:
            patch1 = np.arange(0, cat.shape[0], dtype=int)[np.newaxis].T
            patch2 = np.ones(shape=(cat.shape[0], 1)) * i  # i for index of the cats
            cat = np.concatenate((cat, patch1), axis=1)
            cat = np.concatenate((cat, patch2), axis=1)
            if i - skip == 0:
                cat_full = cat
            else:
                cat_full = np.concatenate((cat_full, cat), axis=0)
    del cat
    if len(cat_full) == 0:
        new_cat = np.zeros((0, 8))
    else:
        tree = spatial.cKDTree(cat_full[:, 0:2])
        idups_raw = tree.query_ball_tree(tree, rad_deg)
        new_cat = []
        idups_merge = []
        for i, srcs in enumerate(idups_raw):
            if i == 0:
                idups_merge.append(srcs)
            else:
                overlap = 0
                for srcs2 in idups_merge:
                    if bool(set(srcs) & set(srcs2)):
                        srcs_m = list(set(srcs + srcs2))
                        idups_merge.remove(srcs2)
                        idups_merge.append(srcs_m)
                        overlap = 1
                if overlap == 0:
                    idups_merge.append(srcs)
        idups_redu = []
        for i, srcs in enumerate(idups_merge):
            inds = cat_full[srcs, -1].astype(int)
            arrs = np.unique(np.array(arr_names)[inds])
            if arrs.shape[0] > N - 1 and srcs not in idups_redu:
                row = np.ones(8) * float("nan")
                for src in srcs:
                    ind = int(cat_full[src, -1])
                    row[ind] = cat_full[src, -2]
                new_cat.append(row)
                idups_redu.append(srcs)

    new_cat = np.array(new_cat)
    if len(new_cat) == 0:
        new_cat = np.zeros((0, 8))
    ocat_dtype = [
        ("pa4_f220", "1f"),
        ("pa4_f150", "1f"),
        ("pa5_f150", "1f"),
        ("pa5_f090", "1f"),
        ("pa6_f150", "1f"),
        ("pa6_f090", "1f"),
        ("pa7_f040", "1f"),
        ("pa7_f030", "1f"),
    ]
    ocat = np.zeros(len(new_cat), ocat_dtype).view(np.recarray)
    ocat.pa4_f220 = new_cat[:, 0]
    ocat.pa4_f150 = new_cat[:, 1]
    ocat.pa5_f150 = new_cat[:, 2]
    ocat.pa5_f090 = new_cat[:, 3]
    ocat.pa6_f150 = new_cat[:, 4]
    ocat.pa6_f090 = new_cat[:, 5]
    ocat.pa7_f040 = new_cat[:, 6]
    ocat.pa7_f030 = new_cat[:, 7]
    return ocat


def crossmatch_array_new(cats_in_order, N, radius, ctime):
    """crossmatch btw 8 catalogs to get a matched catalog with inds from individual cats, the matched candidate should be detected by at least N arrays
       but it would still keep the catalog crossmatched by two bands if there's only 1 array's data available

    Args:
        cats_in_order: a list of 2D np.array of source, each with [[dec,ra]] in deg, order is pa4_f220,pa4_f150, pa5_f150,pa5f090,pa6f150,pa6f090. For non-existin cats, put an empty 2D array in the position in list
        N: integer that candidate should be detected by at least N arrays
        radius: radius to search for matches in arcmin
    """
    from ..utils.utils import obj_dist

    arr_names = ["pa4", "pa4", "pa5", "pa5", "pa6", "pa6", "pa7", "pa7"]
    keys = [
        "pa4_f220",
        "pa4_f150",
        "pa5_f150",
        "pa5_f090",
        "pa6_f150",
        "pa6_f090",
        "pa7_f040",
        "pa7_f030",
    ]
    rad_deg = radius / 60.0
    skip = 0
    cat_full = []
    for i, cat in enumerate(cats_in_order):
        if cat.shape[0] == 0:
            skip += 1
            continue
        else:
            patch1 = np.arange(0, cat.shape[0], dtype=int)[np.newaxis].T
            patch2 = np.ones(shape=(cat.shape[0], 1)) * i  # i for index of the cats
            cat = np.concatenate((cat, patch1), axis=1)
            cat = np.concatenate((cat, patch2), axis=1)
            if i - skip == 0:
                cat_full = cat
            else:
                cat_full = np.concatenate((cat_full, cat), axis=0)
    del cat
    if len(cat_full) == 0:
        new_cat = np.zeros((0, 8))
    else:
        tree = spatial.cKDTree(cat_full[:, 0:2])
        idups_raw = tree.query_ball_tree(tree, rad_deg)
        new_cat = []
        idups_merge = []
        for i, srcs in enumerate(idups_raw):
            if i == 0:
                idups_merge.append(srcs)
            else:
                overlap = 0
                for srcs2 in idups_merge:
                    if bool(set(srcs) & set(srcs2)):
                        srcs_m = list(set(srcs + srcs2))
                        idups_merge.remove(srcs2)
                        idups_merge.append(srcs_m)
                        overlap = 1
                if overlap == 0:
                    idups_merge.append(srcs)
        idups_redu = []
        for i, srcs in enumerate(idups_merge):
            inds = cat_full[srcs, -1].astype(int)
            arrs = np.unique(np.array(arr_names)[inds])
            keys_srcs = np.unique(np.array(keys)[inds])
            if arrs.shape[0] > N - 1 and srcs not in idups_redu:
                row = np.ones(8) * float("nan")
                for src in srcs:
                    ind = int(cat_full[src, -1])
                    row[ind] = cat_full[src, -2]
                new_cat.append(row)
                idups_redu.append(srcs)
            # check if all other freq arr data is not available
            elif arrs.shape[0] == 1 and keys_srcs.shape[0] == 2:
                src = srcs[0]
                ra_deg = cat_full[src, 0]
                dec_deg = cat_full[src, 1]
                keys_left = [key for key in keys if key not in keys_srcs]
                count_ava = 0
                for key in keys_left:
                    ivar_file = (
                        "/scratch/gpfs/snaess/actpol/maps/depth1/release/%s/depth1_%s_%s_ivar.fits"
                        % (ctime[:5], ctime, key)
                    )
                    if os.path.exists(ivar_file):
                        ivar = enmap.read_map(ivar_file)
                        if enmap.contains(
                            ivar.shape,
                            ivar.wcs,
                            [
                                dec_deg * pixell_utils.degree,
                                ra_deg * pixell_utils.degree,
                            ],
                        ):
                            dist = obj_dist(ivar, np.array([[dec_deg, ra_deg]]))[0]
                            dec_pix, ra_pix = ivar.sky2pix(
                                [
                                    dec_deg * pixell_utils.degree,
                                    ra_deg * pixell_utils.degree,
                                ]
                            )
                            ivar_src = ivar[int(dec_pix), int(ra_pix)]
                            if ivar_src != 0.0 and dist > 10.0:
                                count_ava += 1
                if count_ava == 0 and srcs not in idups_redu:
                    row = np.ones(8) * float("nan")
                    for src in srcs:
                        ind = int(cat_full[src, -1])
                        row[ind] = cat_full[src, -2]
                    new_cat.append(row)
                    idups_redu.append(srcs)
    new_cat = np.array(new_cat)
    if len(new_cat) == 0:
        new_cat = np.zeros((0, 8))
    ocat_dtype = [
        ("pa4_f220", "1f"),
        ("pa4_f150", "1f"),
        ("pa5_f150", "1f"),
        ("pa5_f090", "1f"),
        ("pa6_f150", "1f"),
        ("pa6_f090", "1f"),
        ("pa7_f040", "1f"),
        ("pa7_f030", "1f"),
    ]
    ocat = np.zeros(len(new_cat), ocat_dtype).view(np.recarray)
    ocat.pa4_f220 = new_cat[:, 0]
    ocat.pa4_f150 = new_cat[:, 1]
    ocat.pa5_f150 = new_cat[:, 2]
    ocat.pa5_f090 = new_cat[:, 3]
    ocat.pa6_f150 = new_cat[:, 4]
    ocat.pa6_f090 = new_cat[:, 5]
    ocat.pa7_f040 = new_cat[:, 6]
    ocat.pa7_f030 = new_cat[:, 7]
    return ocat


def crossmatch_freq_array(cats_in_order, N, radius):
    """crossmatch btw 6 catalogs to get a matched catalog with inds from individual cats, the matched candidate should be detected by at least N arrays

    Args:
        cats_in_order: a list of 2D np.array of source, each with [[dec,ra]] in deg, order is pa4_f220,pa4_f150, pa5_f150,pa5f090,pa6f150,pa6f090. For non-existin cats, put an empty 2D array in the position in list
        N: integer that candidate should be detected by at least N arrays
        radius: radius to search for matches in arcmin
    """
    # arr_names = ["pa4", "pa4", "pa5", "pa5", "pa6", "pa6", "pa7", "pa7"]
    keys = [
        "pa4_f220",
        "pa4_f150",
        "pa5_f150",
        "pa5_f090",
        "pa6_f150",
        "pa6_f090",
        "pa7_f040",
        "pa7_f030",
    ]
    rad_deg = radius / 60.0
    skip = 0
    cat_full = []
    for i, cat in enumerate(cats_in_order):
        if cat.shape[0] == 0:
            skip += 1
            continue
        else:
            patch1 = np.arange(0, cat.shape[0], dtype=int)[np.newaxis].T
            patch2 = np.ones(shape=(cat.shape[0], 1)) * i  # i for index of the cats
            cat = np.concatenate((cat, patch1), axis=1)
            cat = np.concatenate((cat, patch2), axis=1)
            if i - skip == 0:
                cat_full = cat
            else:
                cat_full = np.concatenate((cat_full, cat), axis=0)
    del cat
    if len(cat_full) == 0:
        new_cat = np.zeros((0, 8))
    else:
        tree = spatial.cKDTree(cat_full[:, 0:2])
        idups_raw = tree.query_ball_tree(tree, rad_deg)
        new_cat = []
        idups_merge = []
        for i, srcs in enumerate(idups_raw):
            if i == 0:
                idups_merge.append(srcs)
            else:
                overlap = 0
                for srcs2 in idups_merge:
                    if bool(set(srcs) & set(srcs2)):
                        srcs_m = list(set(srcs + srcs2))
                        idups_merge.remove(srcs2)
                        idups_merge.append(srcs_m)
                        overlap = 1
                if overlap == 0:
                    idups_merge.append(srcs)
        idups_redu = []
        for i, srcs in enumerate(idups_merge):
            inds = cat_full[srcs, -1].astype(int)
            keys_srcs = np.unique(np.array(keys)[inds])
            if keys_srcs.shape[0] > N - 1 and srcs not in idups_redu:
                row = np.ones(8) * float("nan")
                for src in srcs:
                    ind = int(cat_full[src, -1])
                    row[ind] = cat_full[src, -2]
                new_cat.append(row)
                idups_redu.append(srcs)

    new_cat = np.array(new_cat)
    if len(new_cat) == 0:
        new_cat = np.zeros((0, 8))
    ocat_dtype = [
        ("pa4_f220", "1f"),
        ("pa4_f150", "1f"),
        ("pa5_f150", "1f"),
        ("pa5_f090", "1f"),
        ("pa6_f150", "1f"),
        ("pa6_f090", "1f"),
        ("pa7_f040", "1f"),
        ("pa7_f030", "1f"),
    ]
    ocat = np.zeros(len(new_cat), ocat_dtype).view(np.recarray)
    ocat.pa4_f220 = new_cat[:, 0]
    ocat.pa4_f150 = new_cat[:, 1]
    ocat.pa5_f150 = new_cat[:, 2]
    ocat.pa5_f090 = new_cat[:, 3]
    ocat.pa6_f150 = new_cat[:, 4]
    ocat.pa6_f090 = new_cat[:, 5]
    ocat.pa7_f040 = new_cat[:, 6]
    ocat.pa7_f030 = new_cat[:, 7]
    return ocat


def crossmatch_col(cats, cross_idx):
    new_cats = []
    keys = [
        "pa4_f220",
        "pa4_f150",
        "pa5_f150",
        "pa5_f090",
        "pa6_f150",
        "pa6_f090",
        "pa7_f040",
        "pa7_f030",
    ]
    for i, df in enumerate(cats):
        indices = cross_idx[keys[i]]
        indices = [x for x in indices if str(x) != "nan"]
        df["crossmatch"] = False
        df.loc[indices, "crossmatch"] = True
        new_cats.append(df)
    return new_cats


def merge_cats(full_cats_in_order, N, radius, saveps=False, type="arr", ctime=None):
    """merge btw 6 catalogs to get one single merged catalog, with on overall ra, dec, position error

    Args:
        full_cats_in_order: fits.open(file)[1].data from fits file, order is pa4_f220,pa4_f150, pa5_f150,pa5f090,pa6f150,pa6f090.
        N: integer that candidate should be detected by at least N arrays
        radius: radius to search for matches in arcmin
        saveps: if True, output point sources
        type: arr if crossmatch by array, freq if crossmatch by freq or arr, cond if crossmatch by array but use freq if only one array's data is available
        ctime: ctime in the file name with string format, needed if type is cond
    """
    cats_pos = []
    for i in range(8):
        cat_data = full_cats_in_order[i]
        cat_array = np.zeros((cat_data.shape[0], 2))
        cat_array[:, 0] = cat_data.ra
        cat_array[:, 1] = cat_data.dec
        cats_pos.append(cat_array)
    if type == "arr":
        matched_inds = crossmatch_array(cats_pos, N, radius)
    elif type == "freq":
        matched_inds = crossmatch_freq_array(cats_pos, N, radius)
    elif type == "cond":
        matched_inds = crossmatch_array_new(cats_pos, N, radius, ctime)
    else:
        raise ValueError("type must be arr, freq, or cond")
    new_cats = crossmatch_col(full_cats_in_order, matched_inds)
    cat_merge = []
    for i in range(matched_inds.shape[0]):
        ra = 0
        dec = 0
        invar = 0
        # count = 0
        ps = 0
        for j, [arr, freq] in enumerate(
            [
                ["pa4", "f220"],
                ["pa4", "f150"],
                ["pa5", "f150"],
                ["pa5", "f090"],
                ["pa6", "f150"],
                ["pa6", "f090"],
                ["pa7", "f040"],
                ["pa7", "f030"],
            ]
        ):
            key = "%s_%s" % (arr, freq)
            if not math.isnan(matched_inds[key][i]):
                inputs = None  # TODO: Fix
                fwhm = inputs.get_fwhm_arcmin(arr, freq)
                fwhm_deg = fwhm / 60.0
                idx_src = int(matched_inds[key][i])
                cat_src = full_cats_in_order[j]
                ps += cat_src.source[idx_src]
                ra_src = cat_src.ra[idx_src]
                dec_src = cat_src.dec[idx_src]
                snr_src = cat_src.snr[idx_src]
                pos_err = fwhm_deg / snr_src
                ra += ra_src / pos_err**2.0
                dec += dec_src / pos_err**2.0
                invar += 1 / pos_err**2.0
        if invar == 0:
            print("ZeroDivisionError: Candidate Removed")
            continue
        ra /= invar
        dec /= invar
        var_w = 0
        if (ps > 0) and not saveps:
            continue
        for j, [arr, freq] in enumerate(
            [
                ["pa4", "f220"],
                ["pa4", "f150"],
                ["pa5", "f150"],
                ["pa5", "f090"],
                ["pa6", "f150"],
                ["pa6", "f090"],
                ["pa7", "f040"],
                ["pa7", "f030"],
            ]
        ):
            key = "%s_%s" % (arr, freq)
            if not math.isnan(matched_inds[key][i]):
                idx_src = int(matched_inds[key][i])
                ra_src = full_cats_in_order[j].ra[idx_src]
                dec_src = full_cats_in_order[j].dec[idx_src]
                dist = pixell_utils.angdist(
                    np.array([ra_src, dec_src]) * pixell_utils.degree,
                    np.array([ra, dec]) * pixell_utils.degree,
                )
                inputs = None  # TODO: FIX
                fwhm = inputs.get_fwhm_arcmin(arr, freq)
                fwhm_deg = fwhm / 60.0
                pos_err = fwhm_deg / snr_src
                var_w += (dist**2.0) * (1 / pos_err**2.0)
        var_w /= invar
        pos_err_arcmin = var_w**0.5 * 60 / pixell_utils.degree
        cat_merge.append(
            np.array(
                [
                    ra,
                    dec,
                    pos_err_arcmin,
                    matched_inds["pa4_f220"][i],
                    matched_inds["pa4_f150"][i],
                    matched_inds["pa5_f150"][i],
                    matched_inds["pa5_f090"][i],
                    matched_inds["pa6_f150"][i],
                    matched_inds["pa6_f090"][i],
                    matched_inds["pa7_f040"][i],
                    matched_inds["pa7_f030"][i],
                ]
            )
        )
    if len(cat_merge) == 0:
        cat_merge = np.zeros((0, 11))
    else:
        cat_merge = np.vstack(cat_merge)
    ocat_dtype = [
        ("ra", "1f"),
        ("dec", "1f"),
        ("pos_err_arcmin", "1f"),
        ("pa4_f220", "1f"),
        ("pa4_f150", "1f"),
        ("pa5_f150", "1f"),
        ("pa5_f090", "1f"),
        ("pa6_f150", "1f"),
        ("pa6_f090", "1f"),
        ("pa7_f040", "1f"),
        ("pa7_f030", "1f"),
    ]
    ocat = np.zeros(len(cat_merge), ocat_dtype).view(np.recarray)
    ocat.ra = cat_merge[:, 0]
    ocat.dec = cat_merge[:, 1]
    ocat.pos_err_arcmin = cat_merge[:, 2]
    ocat.pa4_f220 = cat_merge[:, 3]
    ocat.pa4_f150 = cat_merge[:, 4]
    ocat.pa5_f150 = cat_merge[:, 5]
    ocat.pa5_f090 = cat_merge[:, 6]
    ocat.pa6_f150 = cat_merge[:, 7]
    ocat.pa6_f090 = cat_merge[:, 8]
    ocat.pa7_f040 = cat_merge[:, 9]
    ocat.pa7_f030 = cat_merge[:, 10]
    return ocat, new_cats


def candidate_output(
    cross_ocat: pd.DataFrame,
    cats: dict,
    odir: str,
    ctime: float,
    verbosity=0,
    overwrite=True,
):
    # for each candidate, write output
    keys = list(cats.keys())
    for i in range(len(cross_ocat)):
        ocat = {
            "idx": [],
            "band": [],
            "ra": [],
            "dec": [],
            "flux": [],
            "dflux": [],
            "snr": [],
            "ctime": [],
        }

        # create output dir
        can_odir = op.join(odir, f"{ctime}_{str(i)}")
        if not op.exists(can_odir):
            os.makedirs(can_odir)

        # get ra, dec, flux, dflux, snr, ctime for each candidate
        for k in keys:
            # check if detection exists
            if np.isnan(cross_ocat[k][i]):
                continue

            # get ra, dec, flux, dflux, snr, ctime
            ocat["idx"].append(cross_ocat[k][i])
            ocat["band"].append(k)
            ocat["ra"].append(cats[k]["ra"][int(cross_ocat[k][i])])
            ocat["dec"].append(cats[k]["dec"][int(cross_ocat[k][i])])
            ocat["flux"].append(cats[k]["flux"][int(cross_ocat[k][i])])
            ocat["dflux"].append(cats[k]["dflux"][int(cross_ocat[k][i])])
            ocat["snr"].append(cats[k]["snr"][int(cross_ocat[k][i])])
            ocat["ctime"].append(cats[k]["ctime"][int(cross_ocat[k][i])])

        # Convert to table
        output = pd.DataFrame(ocat)
        output = Table.from_pandas(output)

        # metadata
        output.meta["ra_deg"] = cross_ocat["ra"][i]
        output.meta["dec_deg"] = cross_ocat["dec"][i]
        output.meta["err_am"] = cross_ocat["pos_err_arcmin"][i]
        output.meta["maptime"] = ctime

        # write output
        output.write(
            op.join(can_odir, f"{ctime}_{str(i)}_ocat.fits"), overwrite=overwrite
        )

        if verbosity > 0:
            print(f"Wrote {op.join(can_odir, f'{ctime}_{str(i)}_ocat.fits')}")

    return


def blazar_crossmatch(pos, tol=0.5):
    """
    Mask blazar 3C 454.3 using icrs position from SIMBAD

    Args:
        pos:  np.array of positions [[dec, ra]] in deg
        tol: tolerance in deg

    Returns:
        1 in blazar, 0 if not in blazar
    """
    blazar_pos = SkyCoord("22h53m57.7480438728s", "+16d08m53.561508864s", frame="icrs")
    can_pos = SkyCoord(pos[:, 1], pos[:, 0], frame="icrs", unit="deg")
    sep = blazar_pos.separation(can_pos).to(u.deg)
    match = []
    for s in sep:
        if s.value < tol:
            match.append(1)
        else:
            match.append(0)
    return np.array(match).astype(bool)
