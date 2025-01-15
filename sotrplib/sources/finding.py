import numpy as np
from pixell.enmap import enmap
'''
Source finding routines from spt3g_software.
Photutils version implemented by Melanie Archipley

'''

def get_source_sky_positions(extracted_sources,
                             skymap
                             ):
    '''
    from output of extract_sources, use xpeak and ypeak to 
    convert to sky coordinates.
    '''
    from pixell.utils import degree

    for f in extracted_sources:
        x,y = extracted_sources[f]['xpeak'],extracted_sources[f]['ypeak']
        dec,ra = skymap.pix2sky(np.asarray([[y],[x]]))
        extracted_sources[f]['ra'] = ra[0]%(360*degree)
        extracted_sources[f]['dec'] = dec[0]
        
    return extracted_sources


def get_source_observation_time(extracted_sources,
                                timemap:np.ndarray
                                ):
    '''
    from output of extract_sources, use xpeak and ypeak to 
    get the observed time given the map `timemap`.
    '''
    for f in extracted_sources:
        x,y = int(extracted_sources[f]['xpeak']),int(extracted_sources[f]['ypeak'])
        extracted_sources[f]['time'] = timemap[y,x]
        
    return extracted_sources


def extract_sources(inmap:enmap,
                    timemap:enmap=None,
                    maprms:float=None,
                    nsigma:float=5.0,
                    minrad:list=[0.5],
                    sigma_thresh_for_minrad:list=[0.0],
                    res:float=None,
                    pixel_mask:np.ndarray=None,
                    ):
    """
    Quick 'n' dirty source finding

    Arguments
    ---------
    inmap : 2d array or enmap
        2d-array representing an unweighted flux map.
        Must be an enmap to get sky coordinates.
    timemap : enmap (or 2D array)
        time at each pixel, used to get observed time if provided
    maprms : float
        The 1-sigma noise level in the map. If not provided, will be calculated.
    nsigma :
        Required signal-to-noise to detect a source.
    minrad : array-like
        The required separation between detected sources in arcmin. If given as
        a list, provides different radii for different-sigma sources.
    sigma_thresh_for_minrad : array-like
        The source detection strengths corresponding to different exclusion
        radii. Only used if more than one element in ``minrad``.
    res :
        Resolution of map, in arcmin.  Required if ``inmap`` is an array.
    pixel_mask : 2d array or enmap
        Optional mask applied to map before source finding.
    
    Returns
    -------
    output_struct: dict
        Contains a dictionary for each source, labelled by sequential integers,
        with keys 'xpeak', 'ypeak','peakval','peaksig'.
        Also the various photutils output which may be useful:
            "area", "ellipticity", "elongation", "fwhm", "kron_aperture",
            "kron_flux", "kron_fluxerr", "kron_radius"
            
    Notes
    -----
    Jan 2019: DPD ported from spt_analysis/sources/find_sources_quick.pro
    Jan 2025: AF porting to sotrplib
    """

    if res is None:
        try:
            res = np.abs(inmap.wcs.wcs.cdelt[0])
        except AttributeError:
            raise ValueError("Argument `res` required if inmap is an array")

    minrad = np.atleast_1d(minrad)
    sigma_thresh_for_minrad = np.atleast_1d(sigma_thresh_for_minrad)

    
    if len(sigma_thresh_for_minrad) != len(minrad):
        raise ValueError(
            "If you are specifying multiple avoidance radii,"
            + "please supply a threshold level for each one."
        )

    if pixel_mask is not None:
        imap*=pixel_mask 

    # get rms in map if not supplied
    if maprms is None:
        whn0 = np.where(np.abs(inmap) > 1.0e-8)
        if len(whn0[0]) == 0:
            maprms = np.nanstd(inmap)
        else:
            maprms = np.nanstd(np.asarray(inmap)[whn0])
            
    
    
    peaks = find_using_photutils(np.asarray(inmap),
                                 maprms,
                                 nsigma=nsigma,
                                 minnum=1,
                                )
    
    npeaks = peaks["n_detected"]
    if npeaks == 0:
        print("No sources found")
        return

    # gather detected peaks into output structure, ignoring repeat
    # detections of same object
    xpeaks = peaks["xcen"]
    ypeaks = peaks["ycen"]
    peakvals = peaks["maxvals"]
    peaksigs = peaks["sigvals"]
    areas = peaks["area"]
    ellipticities = peaks["ellipticity"]
    elongations = peaks["elongation"]
    fwhms = peaks["fwhm"]
    kron_apertures = peaks["kron_aperture"]
    kron_fluxes = peaks["kron_flux"]
    kron_fluxerrs = peaks["kron_fluxerr"]
    kron_radii = peaks["kron_radius"]
    peak_assoc = np.zeros(npeaks)

    output_struct = dict()
    for i in np.arange(npeaks):
        output_struct[i] = {"xpeak": xpeaks[0],
                            "ypeak": ypeaks[0],
                            "peakval": peakvals[0],
                            "peaksig": peaksigs[0],
                            "area": areas[0].value*res**2,
                            "ellipticity": ellipticities[0],
                            "elongation": elongations[0],
                            "fwhm": fwhms[0].value*res,
                            "kron_aperture": kron_apertures[0],
                            "kron_flux": kron_fluxes[0],
                            "kron_fluxerr": kron_fluxerrs[0],
                            "kron_radius": kron_radii[0].value*res
                           }
            
    minrad_pix = minrad / res

    ksource = 1

    # different accounting if exclusion radius is specified as a function
    # of significance
    if len(minrad) > 1:
        minrad_pix_all = np.zeros(npeaks)
        sthresh = np.argsort(sigma_thresh_for_minrad)
        for j in np.arange(len(minrad)):
            i = sthresh[j]
            whgthresh = np.where(peaksigs >= sigma_thresh_for_minrad[i])[0]
            if len(whgthresh) > 0:
                minrad_pix_all[whgthresh] = minrad_pix[i]
        minrad_pix_os = np.zeros(npeaks)
        minrad_pix_os[0] = minrad_pix_all[0]
        for j in np.arange(npeaks):
            prev_x = np.array(
                [output_struct[n]["xpeak"] for n in np.arange(0, ksource)]
            )
            prev_y = np.array(
                [output_struct[n]["ypeak"] for n in np.arange(0, ksource)]
            )
            distpix = np.sqrt((prev_x - xpeaks[j]) ** 2 + (prev_y - ypeaks[j]) ** 2)
            whclose = np.where(distpix <= minrad_pix_os[0:ksource])[0]
            if len(whclose) == 0:
                output_struct[ksource] = {"xpeak": xpeaks[j],
                                          "ypeak": ypeaks[j],
                                          "peakval": peakvals[j],
                                          "peaksig": peaksigs[j],
                                          "area": areas[j].value*res**2,
                                          "ellipticity": ellipticities[j],
                                          "elongation": elongations[j],
                                          "fwhm": fwhms[j].value*res,
                                          "kron_aperture": kron_apertures[j],
                                          "kron_flux": kron_fluxes[j],
                                          "kron_fluxerr": kron_fluxerrs[j],
                                          "kron_radius": kron_radii[j].value*res
                                         }
                    
                peak_assoc[j] = ksource
                minrad_pix_os[ksource] = minrad_pix_all[j]
                ksource += 1
            else:
                mindist = min(distpix)
                peak_assoc[j] = distpix.argmin()
    else:
        for j in range(npeaks):
            prev_x = np.array(
                [output_struct[n]["xpeak"] for n in np.arange(0, ksource)]
            )
            prev_y = np.array(
                [output_struct[n]["ypeak"] for n in np.arange(0, ksource)]
            )
            distpix = np.sqrt((prev_x - xpeaks[j]) ** 2 + (prev_y - ypeaks[j]) ** 2)
            mindist = min(distpix)
            if mindist > minrad_pix:
                output_struct[ksource] = {"xpeak": xpeaks[j],
                                          "ypeak": ypeaks[j],
                                          "peakval": peakvals[j],
                                          "peaksig": peaksigs[j],
                                          "area": areas[j].value*res**2,
                                          "ellipticity": ellipticities[j],
                                          "elongation": elongations[j],
                                          "fwhm": fwhms[j].value*res,
                                          "kron_aperture": kron_apertures[j],
                                          "kron_flux": kron_fluxes[j],
                                          "kron_fluxerr": kron_fluxerrs[j],
                                          "kron_radius": kron_radii[j].value*res
                                         }
                peak_assoc[j] = ksource
                ksource += 1
            else:
                peak_assoc[j] = distpix.argmin()
    for src in list(output_struct.keys()):
        if src >= ksource:
            del output_struct[src]

    output_struct = get_source_sky_positions(output_struct,
                                             inmap
                                            )
    if not isinstance(timemap,type(None)):
        output_struct = get_source_observation_time(output_struct,
                                                    timemap
                                                   )
    return output_struct


def find_using_photutils(Tmap:np.ndarray, 
                         signoise:float=None,
                         minnum:int=2, 
                         nsigma:float=5.0
                         ):
    """
    Written to take same inputs and return outputs in the same format as the
    function ``find_groups``.  Utilizing astropy.photutils, one can deblend
    sources close to each other.  Exactly how the deblending is done uses the
    default arguments of the package, influenced by SExtractor.

    From ``find_groups``: given a 2d array (a map), will find groups of elements
    (pixels) that are spatially associated (sources).

    Arguments
    ---------
    Tmap: enamp,ndarray
        enmap,ndarray, e.g. representing a flux map.
    signoise : float
        global rms of the map.
        ## need to update this to use tiled rms map - it's all set up, just need to do it.
    offset : float
        Zero point of map.
    minnum : int
        Minimum number of pixels needed to form a group.
    nsigma : float
        Required detection threshold for a group.

    Returns
    -------
    Dictionary with following keys:
        * maxvals - array of heights (in map units) of found objects.
        * sigvals - array of heights (in significance units) of found objects.
        * xcen - array of x-location of group centers. From the documentation,
          the "centroid is computed as the center of mass of the unmasked pixels
          within the source segment."
        * ycen - array of Y-location of group centers.
        * n_detected - Number of detected sources.

    The keys below are outputs of the astropy source finding functionality,
    being used to test if any of them indicate extendedness of a source.
        * area - units of pixels**2
        * ellipticity
        * elongation
        * fwhm - units of pixels. From the documentation, "circularized FWHM of
          2D Gaussian function with same second order moments as the source."
        * kron_aperture
        * kron_flux - Parameter requires that Tmap already be in units of mJy.
        * kron_fluxerr - Parameter requires that Tmap already be in units of mJy.
        * kron_radius - units of pixels.

    Function written by Melanie Archipley, adapted by AF Jan 2025
    """
    # this import is here instead of at the beginning because it has strange
    # dependencies that I (Melanie) do not want to cause problems for other people
    from photutils import segmentation as pseg

    default_keys = {
        "maxvals": "max_value",
        "sigvals": None,
        "xcen": "xcentroid",
        "ycen": "ycentroid",
        "n_detected": None,
    }

    extra_keys = [
        "area",
        "ellipticity",
        "elongation",
        "fwhm",
        "kron_aperture",
        "kron_flux",
        "kron_fluxerr",
        "kron_radius",
    ]

    groups = {k: 0 for k in list(default_keys) + extra_keys}

    if not isinstance(Tmap, np.ndarray):
        Tmap = np.asarray(Tmap)
    assert len(Tmap.shape) == 2
    if not isinstance(signoise, np.ndarray):
        signoise = np.asarray(signoise)
    assert signoise.shape == Tmap.shape

    img = pseg.detect_sources(Tmap, 
                              threshold=nsigma * signoise, 
                              npixels=minnum
                              )
    if img is None:
        return groups

    img = pseg.deblend_sources(Tmap, 
                               img, 
                               npixels=minnum
                               )
    if img is None:
        return groups

    cat = pseg.SourceCatalog(Tmap,
                             img,
                             error=signoise,
                             apermask_method="correct",
                             kron_params=(2.5, 1.0),
                            )

    # convert catalog to table
    columns = [v for _, v in default_keys.items() if v] + extra_keys
    tbl = cat.to_table(columns=columns)
    # rename keys to match find_groups output
    for k, v in default_keys.items():
        if v is not None:
            tbl.rename_column(v, k)
    tbl.sort("maxvals", reverse=True)

    # store signal at each source location
    ix, iy = [np.floor(tbl[k] + 0.5).astype(int) for k in ("xcen", "ycen")]
    tbl["sigvals"] = Tmap[iy, ix] / signoise[iy, ix]

    # populate output dictionary
    groups["n_detected"] = len(tbl)
    for k in groups:
        if k in tbl.columns:
            groups[k] = tbl[k]

    return groups


def source_in_mask(sources, 
                   maskmap
                   ):
    """Determines if source is inside mask

    Args:
        sources: np.array of sources [[dec, ra]] in deg
        maskmap: ndmap of mask (zero masked, one not masked)

    Returns:
        mask column for sources, 1 masked, 0 not masked
    """
    from pixell.enmap import sky2pix
    # Check if sources are in mask
    dec = sources[:, 0]
    ra = sources[:, 1]
    coords_rad = np.deg2rad(np.array([dec, ra]))
    ypix, xpix = sky2pix(maskmap.shape, maskmap.wcs, coords_rad)
    mask = maskmap[ypix.astype(int), xpix.astype(int)]  # lookup mask value

    # Convert to binary
    mask[np.abs(mask) > 0] = 1

    # switch 0 and 1
    mask = mask.astype("bool")
    mask = ~mask
    mask = mask.astype("int")

    return mask


def get_ps_inmap(imap:np.ndarray, 
                 sourcecat, 
                 fluxlim:bool=None
                 ):
    """
    get point sources in map

    Args:
        imap:ndmap to get point sources in
        sourcecat:source catalog
        fluxlim:flux limit in mJy

    Returns:
        sourcecat with sources in map
    """
    from pixell.enmap import sky2pix

    if fluxlim:
        sourcecat = sourcecat[sourcecat["fluxJy"] > fluxlim / 1000.0]
    # convert ra to -180 to 180
    sourcecat["RADeg"] = np.where(
        sourcecat["RADeg"] > 180, sourcecat["RADeg"] - 360, sourcecat["RADeg"]
    )

    sourcecat_coords = sky2pix(imap.shape,
                               imap.wcs,
                               np.array([np.deg2rad(sourcecat["decDeg"]), np.deg2rad(sourcecat["RADeg"])]),
                              )
    sourcecat = sourcecat[
        (sourcecat_coords[0] > 0)
        & (sourcecat_coords[0] < imap.shape[0])
        & (sourcecat_coords[1] > 0)
        & (sourcecat_coords[1] < imap.shape[1])
    ]

    return sourcecat