import numpy as np


def mask_dustgal(imap: enmap.ndmap, 
                 galmask: enmap.ndmap
                ):
    return enmap.extract(galmask, imap.shape, imap.wcs)


def mask_planets(
    ctime,
    tmap: enmap.ndmap,
    planets=["Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
):
    from pixell.utils import ctime2mjd, arcmin
    try:
        from enlib import planet9
    except ModuleNotFoundError:
        print('enlib not found, cannot mask planets')

    mjd_min = ctime2mjd(ctime)
    mjd_max = ctime2mjd(ctime + np.amax(tmap))
    mjds = np.linspace(mjd_min, mjd_max, 10)
    try:
        mask = planet9.build_planet_mask(
            tmap.shape, tmap.wcs, planets, mjds, r=50 * arcmin
        )
        mask_planet = mask.copy()
        mask_planet[np.where(mask == False)] = 1.0
        mask_planet[np.where(mask == True)] = 0.0
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
    
def make_circle_mask(thumb, 
                     ra, 
                     dec, 
                     fwhm=None,
                     mask_radius=None,
                     arr=None, 
                     freq=None, 
                     ):
    """
    Mask of center of thumbnail map

    Args:e
        thumb: thumbnail map
        arr: array name
        freq: frequency
        fwhm: freq+arr fwhm
        mask_radius [arcmin]: make mask of this radius, ignore freq,arr,fwhm
        ra: ra of source in degree
        dec: dec of source in degree

    Returns:
        mask center of thumbnail map
    """
    from pixell.utils import degree
    from .tools import get_cut_radius
    mask = enmap.zeros(thumb.shape, wcs=thumb.wcs, dtype=None)
    mask[np.where(thumb > 0)] = 1
    map_res = np.abs(thumb.wcs.wcs.cdelt[0])
    if arr and freq and not mask_radius:
        r_pix = get_cut_radius(thumb, arr, freq, fwhm)
    else:
        r_pix = mask_radius/(60*map_res)
    center_pix = thumb.sky2pix([dec * degree, ra * degree])
    mask[
        round(center_pix[0]) - r_pix : round(center_pix[0]) + r_pix,
        round(center_pix[1]) - r_pix : round(center_pix[1]) + r_pix,
    ] = 0
    return mask


def make_src_mask(thumb, 
                  srcs_pix,
                  fwhm=None,
                  mask_radius=None,
                  arr=None, 
                  freq=None
                 ):
    from .tools import get_cut_radius
    mask = enmap.zeros(thumb.shape, wcs=thumb.wcs, dtype=None)
    mask[np.where(thumb > 0)] = 1
    map_res = np.abs(thumb.wcs.wcs.cdelt[0])
    if arr and freq and not mask_radius:
        r_pix = get_cut_radius(thumb, arr, freq, fwhm)
    else:
        r_pix = mask_radius/(60*map_res)
    if len(srcs_pix) > 0:
        for cut_pix in srcs_pix:
            dec_pix = int(cut_pix[0])
            ra_pix = int(cut_pix[1])
            dec_min = max(dec_pix - r_pix, 0)
            dec_max = min(dec_pix + r_pix, thumb.shape[0] - 1)
            ra_min = max(ra_pix - r_pix, 0)
            ra_max = min(ra_pix + r_pix, thumb.shape[0] - 1)
            mask[dec_min:dec_max, ra_min:ra_max] = 0
    return mask