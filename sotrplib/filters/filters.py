import numpy as np
from pixell import utils, enmap


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
    from pixell.analysis import matched_filter_constcorr_dual
    from .masks import make_circle_mask
    from ..utils.utils import get_ps_inmap
    from .inputs import get_sourceflux_threshold

    mask_base = enmap.zeros(thumb_temp.shape, wcs=thumb_temp.wcs, dtype=None)
    mask_base[np.where(thumb_ivar > 0)] = 1
    mask_center = make_circle_mask(thumb_ivar, ra_deg, dec_deg,arr=arr, freq=freq)
    apod = enmap.apod_mask(mask_base, apod_size_arcmin * utils.arcmin)
    flux_thresh = get_sourceflux_threshold(freq)
    if source_cat:
        source_cat = get_ps_inmap(thumb_temp,
                                  source_cat,
                                  fluxlim=flux_thresh
                                  )
        mask_srcs = mask_base
        for i in range(len(source_cat)):
            mask_srcs *= make_circle_mask(thumb_ivar, 
                                          source_cat['RADeg'][i],
                                          source_cat['decDeg'][i],
                                          mask_radius=4
                                          )
        apod_noise = enmap.apod_mask(mask_center * mask_srcs * mask_base, 
                                     apod_size_arcmin * utils.arcmin,
                                     )
    else:
        apod_noise = enmap.apod_mask(mask_center * mask_base, 
                                     apod_size_arcmin * utils.arcmin,
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
    l = thumb_temp.modlmap()
    B = enmap.samewcs(np.interp(l, np.arange(len(bl)), bl), l)
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


def renorm_ms(
    thumb_snr, arr, freq, ra_deg, dec_deg, source_cat=None, apod_size_arcmin=10
):
    """
    renormalize the matched filtered maps
    Args:
        thumb_snr:thumbnail of matched filtered map, it should be a snr map
        arr: array name
        freq: frequency
        ra_deg:ra of source in degree
        dec_deg:dec of source in degree
        source_cat: catalog of point sources
    """
    from .masks import make_circle_mask
    from ..utils.utils import get_ps_inmap
    from .inputs import get_sourceflux_threshold

    resolution = np.abs(thumb_snr.wcs.wcs.cdelt[0])
    apod_pix = int(apod_size_arcmin / (resolution * 60) + 2)
    mask = make_circle_mask(thumb_snr, 
                            ra_deg, 
                            dec_deg,
                            arr=arr,
                            freq=freq
                            ) 
    flux_thresh = get_sourceflux_threshold(freq)
    mask_apod = enmap.zeros(thumb_snr.shape, wcs=thumb_snr.wcs, dtype=None)
    mask_apod[
        apod_pix : thumb_snr.shape[0] - apod_pix,
        apod_pix : thumb_snr.shape[1] - apod_pix,
    ] = 1
    if source_cat:
        source_cat = get_ps_inmap(thumb_snr,
                                  source_cat,
                                  fluxlim=flux_thresh
                                  )
        for i in range(len(source_cat)):
            mask *= make_circle_mask(thumb_ivar, 
                                    source_cat['RADeg'][i],
                                    source_cat['decDeg'][i],
                                    mask_radius=4
                                    )
        
    mask *= mask_apod
    thumb_snr_sel_sq = (thumb_snr * mask) ** 2
    mean_snr_sq = thumb_snr_sel_sq[np.nonzero(thumb_snr_sel_sq)].mean()
    thumb_snr_renorm = thumb_snr / (mean_snr_sq) ** 0.5
    return thumb_snr_renorm


def matched_filter_1overf(
    thumb_temp, thumb_ivar, freq, source_cat=None, size_deg=0.5, apod_size_arcmin=10
):
    """
    Make matched filter map with noised modeled by 1/f noise

    Args:
        thumb_temp: thumbnail of temperature map
        thumb_ivar: thumbnail of inverse variance map
        freq: frequency
        source_cat: catalog of point sources, if None do not cut point sources

    """
    from pixell.analysis import matched_filter_constcorr_dual
    from pixell.uharm import UHT
    wcs = thumb_temp.wcs
    shape = thumb_temp.shape
    bsigma = 1.4 * utils.fwhm * utils.arcmin
    fconv = utils.dplanck(150e9, utils.T_cmb) / 1e3
    alpha = -3.5
    uht = UHT(shape, wcs)
    iN = (1 + ((uht.l + 0.5) / 3000) ** alpha) ** (-1)
    if freq == "f220":
        bsigma = 1.4 * utils.fwhm * utils.arcmin * 150.0 / 220.0
        fconv = utils.dplanck(220e9, utils.T_cmb) / 1e3
        iN = (1 + ((uht.l + 0.5) / 4000) ** alpha) ** (-1)
        # iN   =  10**-2/utils.arcmin**2 / (1 + ((uht.l+0.5)/4000)**-3.5)
    elif freq == "f090":
        bsigma = 1.4 * utils.fwhm * utils.arcmin * 150.0 / 90.0
        fconv = utils.dplanck(90e9, utils.T_cmb) / 1e3
        iN = (1 + ((uht.l + 0.5) / 2000) ** alpha) ** (-1)
        # iN   =  10**-2/utils.arcmin**2 / (1 + ((uht.l+0.5)/2000)**-3.5)
    beam = np.exp(-0.5 * uht.l**2 * bsigma**2)
    rho, kappa = matched_filter_constcorr_dual(
        thumb_temp * fconv, beam, thumb_ivar / fconv**2, iN, uht=uht
    )
    return rho, kappa
