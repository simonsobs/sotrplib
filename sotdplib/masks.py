import numpy as np

try:
    from enlib import planet9
except ModuleNotFoundError:
    print('enlib not found, cannot mask planets')

from pixell import enmap,utils


def mask_dustgal(imap: enmap.ndmap, 
                 galmask: enmap.ndmap
                ):
    return enmap.extract(galmask, imap.shape, imap.wcs)


def mask_planets(
    ctime,
    tmap: enmap.ndmap,
    planets=["Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
):
    mjd_min = utils.ctime2mjd(ctime)
    mjd_max = utils.ctime2mjd(ctime + np.amax(tmap))
    mjds = np.linspace(mjd_min, mjd_max, 10)
    try:
        mask = planet9.build_planet_mask(
            tmap.shape, tmap.wcs, planets, mjds, r=50 * utils.arcmin
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
