import numpy as np
from pixell import enmap


def mask_dustgal(imap: enmap.ndmap, galmask: enmap.ndmap):
    return enmap.extract(galmask, imap.shape, imap.wcs)


def mask_planets(
    tmap: enmap.ndmap,
    ctime: float,
    planets=["Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
) -> enmap.ndmap:
    from pixell.utils import arcmin, ctime2mjd

    try:
        from enlib import planet9
    except ModuleNotFoundError:
        print('enlib not found, cannot mask planets')
        return np.ones(tmap.shape)
    
    mjd_min = ctime2mjd(ctime)
    mjd_max = ctime2mjd(ctime + np.amax(tmap))
    mjds = np.linspace(mjd_min, mjd_max, 10)
    try:
        mask = planet9.build_planet_mask(
            tmap.shape, tmap.wcs, planets, mjds, r=50 * arcmin
        )
        mask_planet = mask.copy()
        mask_planet[np.where(not mask)] = 1.0
        mask_planet[np.where(mask)] = 0.0
    except Exception:
        mask_planet = np.ones(tmap.shape)

    return mask_planet


def mask_edge(imap: enmap.ndmap, pix_num: int):
    """Get a mask that masking off edge pixels

    Args:
        imap:ndmap to create mask for,usually kappa map
        pix_num: pixels within this number from edge will be cutoff

    Returns:
        binary ndmap with 1 == unmasked, 0 == masked
    """
    from scipy.ndimage import morphology as morph

    from .maps import edge_map

    edge = edge_map(imap)
    dmap = enmap.enmap(morph.distance_transform_edt(edge), edge.wcs)
    del edge
    mask = enmap.zeros(imap.shape, imap.wcs)
    mask[np.where(dmap > pix_num)] = 1
    return mask


def get_masked_map(
    imap: enmap.ndmap,
    edgemask: enmap.ndmap,
    galmask: enmap.ndmap,
    planet_mask: enmap.ndmap,
):
    """returns map with edge, dustygalaxy, and planets masked

    Args:
        imap: input map
        edgemask: mask of map edges
        galmask: enmap of galaxy mask
        planet_mask: enmap of planet mask

    Returns:
        imap with masks applied
    """
    if galmask is None:
        return imap * edgemask * planet_mask
    else:
        galmask = mask_dustgal(imap, galmask)
        return imap * edgemask * galmask * planet_mask


def make_src_mask(imap:enmap.ndmap, 
                  srcs_pix:list=None, 
                  fwhm:float=None, 
                  mask_radius:list=[], 
                  arr:str=None, 
                  freq:str=None
                  ):
    '''
    mask_radius in pixels, can be gotten using
    sotrplib.utils.utils.get_pix_from_peak_to_noise
    
    '''
    from ..utils.utils import get_cut_radius
    from pixell.utils import degree,arcmin

    mask = enmap.ones(imap.shape, wcs=imap.wcs, dtype=None)
    map_res = np.abs(imap.wcs.wcs.cdelt[0])*degree
    if arr and freq and len(mask_radius)==0:
        r_pix = get_cut_radius(map_res/arcmin, arr, freq, fwhm)
    else:
        r_pix = np.asarray(mask_radius) 

    if len(srcs_pix) > 0:
        for i,cut_pix in enumerate(srcs_pix):
            r = round(r_pix[i])+1
            dec_pix = round(cut_pix[0])
            ra_pix = round(cut_pix[1])
            dec_min = max(dec_pix - r, 0)
            dec_max = min(dec_pix + r, imap.shape[0] - 1)
            dec, _ = imap.pix2sky([dec_pix, ra_pix])
            dec_factor = np.cos(dec)
            ra_min = max(round(ra_pix - r / dec_factor), 0)
            ra_max = min(round(ra_pix + r / dec_factor), imap.shape[1] - 1)
            mask[dec_min:dec_max, ra_min:ra_max] = 0

    return mask
