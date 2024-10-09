import numpy as np
from pixell import utils, uharm, enmap, reproject, analysis, coordinates
from sotdplib import inputs
import warnings


def get_thumbnail(imap, ra_deg, dec_deg, size_deg=0.5):
    ra = ra_deg * utils.degree
    dec = dec_deg * utils.degree
    radius = size_deg * utils.degree
    omap = reproject.thumbnails(imap, [dec, ra], radius)
    return omap


def get_submap(imap, ra_deg, dec_deg, size_deg=0.5):
    ra = ra_deg * utils.degree
    dec = dec_deg * utils.degree
    radius = size_deg * utils.degree
    omap = imap.submap([[dec - radius, ra - radius], [dec + radius, ra + radius]])
    return omap


def get_thumbnail_ivar(imap, ra_deg, dec_deg, size_deg=0.5):
    ra = ra_deg * utils.degree
    dec = dec_deg * utils.degree
    radius = size_deg * utils.degree
    omap = reproject.thumbnails_ivar(imap, [dec, ra], radius)
    return omap


def find_close_ps(ra, dec, catalog, thumb, flux_thres, size_deg=0.5):
    """
    Given position and map find nearby point sources in catalog

    Args:
        ra: ra of the center of the map [rad]
        dec: dec of the center of the map [rad]
        catalog: catalog of point sources
        thumb: map
        flux_thres: flux threshold [mJy]
        size_deg: size of the map [deg]

    Returns:
        list of pixel positions of point sources
    """
    pixs = []
    for i in range(len(catalog)):
        ra_src = catalog["RADeg"][i] * utils.degree
        dec_src = catalog["decDeg"][i] * utils.degree
        if ra_src > np.pi:
            ra_src -= 2 * np.pi
        if (
            np.abs(ra_src - ra) < size_deg * 2 * utils.degree
            and np.abs(dec_src - dec) < size_deg * utils.degree
            and catalog["fluxJy"][i] * 1000.0 > flux_thres
        ):
            ps_cor = coordinates.recenter(
                [ra_src, dec_src], [ra, dec, 0, 0], restore=False
            )
            ps_pix = thumb.sky2pix([ps_cor[1], ps_cor[0]])
            if (
                -10 < ps_pix[0] < thumb.shape[0] + 10
                and -10 < ps_pix[1] < thumb.shape[0] + 10
            ):
                pixs.append(ps_pix)
    return pixs


def find_close_ps_sub(ra, dec, catalog, thumb, flux_thres, size_deg=0.5):
    """
    Given position and map(not thumbnail maps) find nearby point sources in catalog

    Args:
        ra: ra of the center of the map [rad]
        dec: dec of the center of the map [rad]
        catalog: catalog of point sources
        thumb: map
        flux_thres: flux threshold [mJy]
        size_deg: size of the map [deg]

    Returns:
        list of pixel positions of point sources
    """
    pixs = []
    for i in range(len(catalog)):
        ra_src = catalog.RADeg[i] * utils.degree
        dec_src = catalog.decDeg[i] * utils.degree
        if ra_src > np.pi:
            ra_src -= 2 * np.pi
        if (
            np.abs(ra_src - ra) < size_deg * 2 * utils.degree
            and np.abs(dec_src - dec) < size_deg * utils.degree
            and catalog.fluxJy[i] * 1000.0 > flux_thres
        ):
            ps_pix = thumb.sky2pix([dec_src, ra_src])
            if (
                -10 < ps_pix[0] < thumb.shape[0] + 10
                and -10 < ps_pix[1] < thumb.shape[0] + 10
            ):
                pixs.append(ps_pix)
    return pixs


def get_cut_radius(thumb, arr, freq, fwhm=None):
    """
    get desired radius that we want to cut on point sources or signal at the center, radius~2*fwhm

    Args:
        thumb: thumbnail map
        arr: array name
        freq: frequency

    Returns:
        radius in pixel
    """
    if fwhm is None:
        fwhm = inputs.get_fwhm_arcmin(arr, freq)
    resolution = np.abs(thumb.wcs.wcs.cdelt[0])
    radius_pix = round(2 * fwhm / (60 * resolution))
    return radius_pix


def get_cut_radius_mf(thumb, arr, freq):
    """
    get desired radius that we want to cut on point sources or signal at the center for matched filtered maps, radius~2*fwhm*sqr(2)

    Args:
        thumb: thumbnail map
        arr: array name
        freq: frequency

    Returns:
        radius in pixel
    """
    fwhm = inputs.get_fwhm_arcmin(arr, freq)
    resolution = np.abs(thumb.wcs.wcs.cdelt[0])
    radius_pix = round(2**0.5 * 2 * fwhm / (60 * resolution))
    return radius_pix


def cut_center_mask(thumb, arr, freq, ra, dec, fwhm=None):
    """
    Mask of center of thumbnail map

    Args:
        thumb: thumbnail map
        arr: array name
        freq: frequency
        ra: ra of source in degree
        dec: dec of source in degree

    Returns:
        mask center of thumbnail map
    """
    mask = enmap.zeros(thumb.shape, wcs=thumb.wcs, dtype=None)
    mask[np.where(thumb > 0)] = 1
    r_pix = get_cut_radius(thumb, arr, freq, fwhm)
    center_pix = thumb.sky2pix([dec * utils.degree, ra * utils.degree])
    mask[
        round(center_pix[0]) - r_pix : round(center_pix[0]) + r_pix,
        round(center_pix[1]) - r_pix : round(center_pix[1]) + r_pix,
    ] = 0
    return mask


def cut_center_mask_mf(thumb, arr, freq, ra, dec):
    """
    Mask of center of thumbnail map, which are matched filtered

    Args:
        thumb: matched filtered thumbnail map
        arr: array name
        freq: frequency
        ra: ra of source in degree
        dec: dec of source in degree

    Returns:
        mask center of thumbnail map
    """
    mask = enmap.zeros(thumb.shape, wcs=thumb.wcs, dtype=None)
    mask[np.where(thumb > 0)] = 1
    r_pix = get_cut_radius_mf(thumb, arr, freq)
    center_pix = thumb.sky2pix([dec * utils.degree, ra * utils.degree])
    mask[
        round(center_pix[0]) - r_pix : round(center_pix[0]) + r_pix,
        round(center_pix[1]) - r_pix : round(center_pix[1]) + r_pix,
    ] = 0
    return mask


def cut_src_mask(thumb, arr, freq, srcs_pix):
    mask = enmap.zeros(thumb.shape, wcs=thumb.wcs, dtype=None)
    mask[np.where(thumb > 0)] = 1
    r_pix = get_cut_radius(thumb, arr, freq)
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


def cut_src_mask_mf(thumb, arr, freq, srcs_pix):
    mask = enmap.zeros(thumb.shape, wcs=thumb.wcs, dtype=None)
    mask[np.where(thumb > 0)] = 1
    r_pix = get_cut_radius_mf(thumb, arr, freq)
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


def matched_filter(
    thumb_temp,
    thumb_ivar,
    arr,
    freq,
    ra_deg,
    dec_deg,
    source_cat=None,
    size_deg=0.5,
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
    ra = ra_deg * utils.degree
    dec = dec_deg * utils.degree
    mask_base = enmap.zeros(thumb_temp.shape, wcs=thumb_temp.wcs, dtype=None)
    mask_base[np.where(thumb_ivar > 0)] = 1
    mask_center = cut_center_mask(thumb_ivar, arr, freq, ra_deg, dec_deg)
    apod = enmap.apod_mask(mask_base, 10 * utils.arcmin)
    flux_thres = inputs.get_sourceflux_threshold(freq)
    if source_cat:
        cut_pixs = find_close_ps(
            ra, dec, source_cat, thumb_temp, flux_thres, size_deg=0.5
        )
        mask_srcs = cut_src_mask(thumb_ivar, arr, freq, cut_pixs)
        apod_noise = enmap.apod_mask(
            mask_center * mask_srcs * mask_base, apod_size_arcmin * utils.arcmin
        )
    else:
        apod_noise = enmap.apod_mask(
            mask_center * mask_base, apod_size_arcmin * utils.arcmin
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
    rho, kappa = analysis.matched_filter_constcorr_dual(
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
    ra = ra_deg * utils.degree
    dec = dec_deg * utils.degree
    resolution = np.abs(thumb_snr.wcs.wcs.cdelt[0])
    apod_pix = int(apod_size_arcmin / (resolution * 60) + 2)
    mask_center = cut_center_mask_mf(thumb_snr, arr, freq, ra_deg, dec_deg)
    flux_thres = inputs.get_sourceflux_threshold(freq)
    mask_apod = enmap.zeros(thumb_snr.shape, wcs=thumb_snr.wcs, dtype=None)
    mask_apod[
        apod_pix : thumb_snr.shape[0] - apod_pix,
        apod_pix : thumb_snr.shape[1] - apod_pix,
    ] = 1
    if source_cat:
        cut_pixs = find_close_ps(
            ra, dec, source_cat, thumb_snr, flux_thres, size_deg=0.5
        )
        mask_srcs = cut_src_mask_mf(thumb_snr, arr, freq, cut_pixs)
        mask_all = mask_center * mask_srcs * mask_apod
    else:
        mask_all = mask_center * mask_apod
    thumb_snr_sel_sq = (thumb_snr * mask_all) ** 2
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
    wcs = thumb_temp.wcs
    shape = thumb_temp.shape
    bsigma = 1.4 * utils.fwhm * utils.arcmin
    fconv = utils.dplanck(150e9, utils.T_cmb) / 1e3
    alpha = -3.5
    uht = uharm.UHT(shape, wcs)
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
    pixarea = enmap.pixsizemap(shape, wcs)
    rho, kappa = analysis.matched_filter_constcorr_dual(
        thumb_temp * fconv, beam, thumb_ivar / fconv**2, iN, uht=uht
    )
    return rho, kappa
