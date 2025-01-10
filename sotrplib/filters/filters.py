import numpy as np
from pixell import enmap, utils


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

    from ..utils.utils import get_ps_inmap
    from .inputs import get_sourceflux_threshold
    from .masks import make_circle_mask

    mask_base = enmap.zeros(thumb_temp.shape, wcs=thumb_temp.wcs, dtype=None)
    mask_base[np.where(thumb_ivar > 0)] = 1
    mask_center = make_circle_mask(thumb_ivar, ra_deg, dec_deg, arr=arr, freq=freq)
    apod = enmap.apod_mask(mask_base, apod_size_arcmin * utils.arcmin)
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
            apod_size_arcmin * utils.arcmin,
        )
    else:
        apod_noise = enmap.apod_mask(
            mask_center * mask_base,
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
    from ..utils.utils import get_ps_inmap
    from .inputs import get_sourceflux_threshold
    from .masks import make_circle_mask

    resolution = np.abs(thumb_snr.wcs.wcs.cdelt[0])
    apod_pix = int(apod_size_arcmin / (resolution * 60) + 2)
    mask = make_circle_mask(thumb_snr, ra_deg, dec_deg, arr=arr, freq=freq)
    flux_thresh = get_sourceflux_threshold(freq)
    mask_apod = enmap.zeros(thumb_snr.shape, wcs=thumb_snr.wcs, dtype=None)
    mask_apod[
        apod_pix : thumb_snr.shape[0] - apod_pix,
        apod_pix : thumb_snr.shape[1] - apod_pix,
    ] = 1
    if source_cat:
        source_cat = get_ps_inmap(thumb_snr, source_cat, fluxlim=flux_thresh)
        for i in range(len(source_cat)):
            mask *= make_circle_mask(
                thumb_ivar,  # noqa
                source_cat["RADeg"][i],
                source_cat["decDeg"][i],
                mask_radius=4,
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



def matched_filter_full_map(file,imap, ivar, arr, freq):
    """
    Make matched filter map using Sigurd's way
    basically copied from this script:/home/yaqiongl/code/depth1_transients/filter_depth1.py on tiger

    Args:
        mapfile: file path of the original depth1 map file
        imap: temp map
        ivar: inverse variance map
        arr: array
        freq: frequency
    """
    frequency = float(freq[1:])*1e9
    fconv = utils.dplanck(frequency)/1e3
    imap *= fconv
    ivar /= fconv**2
    path = mapfile[:-8]
    infofile = path + 'info.hdf'
    info = bunch.read(infofile)
    dtype  = imap.dtype
    ny, nx = imap.shape[-2:]
    beam_file = '/projects/ACT/adriaand/beams/20220817_beams/coadd_%s_%s_night_beam_tform_jitter_cmb.txt'%(arr,freq) #this path is on tiger
    beam1d = np.loadtxt(beam_file).T[1]
    lfwhm  = np.where(beam1d<0.5)[0][0]
    fwhm   = 1/(lfwhm*utils.fwhm)/utils.fwhm
    mask_file = '/scratch/gpfs/snaess/actpol/masks/srcfind/srcfind_mask_%s.fits'%(freq)   #this path is on tiger
    #Parameters I used defult from Sigurd's script
    shrink_holes=1.0
    apod_edge = 10
    apod_holes = 10
    highpass = 0
    band_height = 2
    shift = 0
    lres = np.array([70,100])
    simple = False
    simple_alpha = -3.5
    simple_lknee = 1000
    apod_edge   = apod_edge *utils.arcmin
    apod_holes  = apod_holes*utils.arcmin
    shrink_holes= shrink_holes*fwhm
    # Build our shift matrix
    if shift > 0: S = ShiftMatrix(imap.shape, imap.wcs, info.profile)
    else:              S = ShiftDummy()
    # Set up apodization. A bit messy due to handling two types of apodizion
    # depending on whether it's based on the extrnal mask or not
    hit  = ivar > 0
    if shrink_holes > 0:
        hit = enmap.shrink_mask(enmap.grow_mask(hit, shrink_holes), shrink_holes)
    # Apodize the edge by decreasing the significance in ivar
    noise_apod = enmap.apod_mask(hit, apod_edge)
    apod_holes  = apod_holes*utils.arcmin
    shrink_holes= shrink_holes*fwhm
    # Build our shift matrix
    if shift > 0: S = ShiftMatrix(imap.shape, imap.wcs, info.profile)
    else:              S = ShiftDummy()
    # Set up apodization. A bit messy due to handling two types of apodizion
    # depending on whether it's based on the extrnal mask or not
    hit  = ivar > 0
    if shrink_holes > 0:
        hit = enmap.shrink_mask(enmap.grow_mask(hit, shrink_holes), shrink_holes)
    # Apodize the edge by decreasing the significance in ivar
    noise_apod = enmap.apod_mask(hit, apod_edge)
    # Check if we have a noise model mask too
    mask = enmap.read_map(mask_file, geometry=imap.geometry)
    mask = np.asanyarray(mask)
    noise_apod *= enmap.apod_mask(1-mask, apod_holes)
    del mask
    # Build the noise model
    iC  = build_ips2d_udgrade(S.forward(imap), S.forward(ivar*noise_apod), apod_corr=np.mean(noise_apod**2), lres=lres)
    del noise_apod
    if highpass > 0:
        iC = highpass_ips2d(iC, highpass)
    # Set up output map buffers
    rho   = imap*0
    kappa = imap*0
    tot_weight = np.zeros(ny, imap.dtype)
    # Bands. At least band_height in height and with at least
    # 2*apod_edge of overlapping padding at top and bottom
    nband    = utils.ceil(imap.extent()[0]/band_height) if band_height > 0 else 1
    bedge    = utils.ceil(apod_edge/imap.pixshape()[0]) * 2
    boverlap = bedge
    for bi, r in enumerate(overlapping_range_iterator(ny, nband, boverlap, padding=bedge)):
        # Get band-local versions of the map etc.
        bmap, bivar, bhit = [a[...,r.i1:r.i2,:] for a in [imap, ivar, hit]]
        bny, bnx = bmap.shape[-2:]
        if shift > 0: bS = ShiftMatrix(bmap.shape, bmap.wcs, info.profile)
        else:              bS = ShiftDummy()
        # Translate iC to smaller fourier space
        if simple:
            bl  = bmap.modlmap()
            biC = (1 + (np.maximum(bl,0.5)/simple_lknee)**simple_alpha)**-1
        else:
            biC     = enmap.ifftshift(enmap.resample(enmap.fftshift(iC), bmap.shape, method="spline", order=1))
            biC.wcs = bmap.wcs.deepcopy()
        # 2d beam
        beam2d  = enmap.samewcs(utils.interp(bmap.modlmap(), np.arange(len(beam1d)), beam1d), bmap)
        # Set up apodization
        filter_apod = enmap.apod_mask(bhit, apod_edge)
        bivar       = bivar*filter_apod
        del filter_apod
        # Phew! Actually perform the filtering
        uht  = uharm.UHT(bmap.shape, bmap.wcs, mode="flat")
        brho, bkappa = analysis.matched_filter_constcorr_dual(bmap, beam2d, bivar, biC, uht, S=bS.forward, iS=bS.backward)
        del uht
        # Restrict to exposed area
        brho   *= bhit
        bkappa *= bhit
        # Merge into full output
        weight = r.weight.astype(imap.dtype)
        rho  [...,r.i1+r.p1:r.i2-r.p2,:] += brho  [...,r.p1:bny-r.p2,:]*weight[:,None]
        kappa[...,r.i1+r.p1:r.i2-r.p2,:] += bkappa[...,r.p1:bny-r.p2,:]*weight[:,None]
        tot_weight[r.i1+r.p1:r.i2-r.p2]  += weight
        del bmap, bivar, bhit, bS, biC, beam2d, brho, bkappa

    del imap, ivar, iC
    return rho, kappa
