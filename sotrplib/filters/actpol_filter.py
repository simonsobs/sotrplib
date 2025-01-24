import numpy as np
from pixell import enmap


def matched_filter(
    thumb_temp,
    thumb_ivar,
    arr,
    freq,
    ra_deg,
    dec_deg,
    source_cat=None,
    apod_size_arcmin=10,
):
    """
    Make matched filter map around source thumbnail

    Args:
        thumb_temp: thumbnail of temperature map
        thumb_ivar: thumbnail of inverse variance map
        arr: array name
        freq: frequency
        ra_deg: ra of source in degree
        dec_deg: dec of source in degree
        source_cat: catalog of point sources, if None do not cut point sources

    """
    import warnings
    from pixell.utils import arcmin
    from pixell.analysis import matched_filter_constcorr_dual

    from ..utils.utils import get_ps_inmap
    from ..utils.actpol_utils import get_sourceflux_threshold
    from ..maps.masks import make_circle_mask

    mask_base = enmap.zeros(thumb_temp.shape, wcs=thumb_temp.wcs, dtype=None)
    mask_base[np.where(thumb_ivar > 0)] = 1
    mask_center = make_circle_mask(thumb_ivar, ra_deg, dec_deg, arr=arr, freq=freq)
    apod = enmap.apod_mask(mask_base, apod_size_arcmin * arcmin)
    flux_thresh = get_sourceflux_threshold(freq)
    if source_cat:
        source_cat = get_ps_inmap(thumb_temp, source_cat, fluxlim=flux_thresh)
        mask_srcs = mask_base
        for i in range(len(source_cat)):
            mask_srcs *= make_circle_mask(
                thumb_ivar,
                source_cat["RADeg"][i],
                source_cat["decDeg"][i],
                mask_radius=4,
            )
        apod_noise = enmap.apod_mask(
            mask_center * mask_srcs * mask_base,
            apod_size_arcmin * arcmin,
        )
    else:
        apod_noise = enmap.apod_mask(
            mask_center * mask_base,
            apod_size_arcmin * arcmin,
        )
    if arr == "pa7":
        datapath = (
            "/home/eb8912/transients/depth1/ACT_depth1-transient-pipeline/data/inputs/beam_transform_%s_%s_201203.txt"
            % (arr, freq)
        )
    else:
        datapath = (
            "/home/eb8912/transients/depth1/ACT_depth1-transient-pipeline/data/inputs/s19_%s_%s_night_beam_tform_instant.txt"
            % (arr, freq)
        )
    bl = np.loadtxt(datapath).T[1]
    ell = thumb_temp.modlmap()
    B = enmap.samewcs(np.interp(ell, np.arange(len(bl)), bl), ell)
    ps2d = enmap.upgrade(
        enmap.downgrade(
            np.abs(enmap.fft(thumb_temp * thumb_ivar**0.5 * apod_noise)) ** 2,
            5,
            ref=[0, 0],
            inclusive=True,
        ),
        5,
        oshape=thumb_temp.shape,
        inclusive=True,
    ) / np.mean(apod_noise**2)
    # Force shape if it does not output correct shape Todo: Data loss for some maps. This is a quick fix
    if ps2d.shape != thumb_temp.shape:
        ps2d = ps2d[: thumb_temp.shape[0], : thumb_temp.shape[1]]
        warnings.warn("Forcing ps2d shape to match thumb_temp shape. Data may be lost.")

    iN_emp = 1.0 / ps2d
    rho, kappa = matched_filter_constcorr_dual(
        thumb_temp * apod, B, thumb_ivar * apod, iN_emp
    )
    return rho, kappa