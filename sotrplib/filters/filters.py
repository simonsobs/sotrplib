import numpy as np
from pixell import enmap, utils

"""
Lots of functions copied from https://github.com/amaurea/tenki/blob/master/filter_depth1.py

"""


def renorm_thumbnail_snr(
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


def fix_profile(profile):
    """profile has a bug where it was generated over too big an
    az range. This is probably harmless, but could in theory lead to it moving
    past north, which makes the profile non-monotonous in dec. This function
    chops off those parts.
    """
    # First fix any wrapping issues
    profile[1] = utils.unwind(profile[1])
    # Then handle any north/south-crossing
    dec = profile[0]
    i0 = len(dec) // 2
    # Starting from the reference point i0, run left and right until
    # the slope switches sign. That's equivalent to the code below.
    # The point i0 is double-counted here, but that compensates for
    # the ddec calculation shortening the array.
    ddec = dec[1:] - dec[:-1]
    good1 = ddec[: i0 + 1] * ddec[i0] > 0
    good2 = ddec[i0:] * ddec[i0] > 0
    good = np.concatenate([good1, good2])
    return profile[:, good]


class ShiftMatrix:
    def __init__(self, shape, wcs, profile):
        """Given a map geometry and a scanning profile [{dec,ra},:]
        create an operator that can be used to transform a map to/from
        a coordinate system where the scans are straight vertically.
        """
        map_decs, map_ras = enmap.posaxes(shape, wcs)
        # make sure it's sorted, otherwise interp won't work
        profile = fix_profile(profile)
        profile = profile[:, np.argsort(profile[0])]
        # get the profile ra for each dec in the map,
        # and since its position is arbitrary, put it in the middle ofthe
        # map to be safe
        ras = np.interp(map_decs, profile[0], profile[1])
        ras += np.mean(map_ras) - np.mean(ras)
        # Transform to x pixel positions
        xs = enmap.sky2pix(shape, wcs, [map_decs, ras])[1]
        # We want to turn this into pixel *offsets* rather than
        # direct pixel values.
        dxs = xs - shape[-1] / 2
        # This operator just shifts whole pixels, it doesn't interpolate
        dxs = utils.nint(dxs)
        self.dxs = dxs

    def forward(self, map):
        from pixell import array_ops

        # Numpy can't do this efficiently
        return array_ops.roll_rows(map, -self.dxs)

    def backward(self, map):
        from pixell import array_ops

        return array_ops.roll_rows(map, self.dxs)


class ShiftDummy:
    def forward(self, map):
        return map.copy()

    def backward(self, map):
        return map.copy()


class NmatShiftConstCorr:
    def __init__(self, S, ivar, iC):
        self.S = S
        self.H = ivar**0.5
        self.iC = iC

    def apply(self, map, omap=None):
        sub = map.extract(self.H.shape, self.H.wcs)
        sub = self.H * self.S.backward(
            enmap.ifft(self.iC * enmap.fft(self.S.forward(self.H * sub))).real
        )
        if omap is None:
            return sub.extract(map.shape, map.wcs)
        else:
            return omap.insert(sub, op=np.add)


def build_ips2d_udgrade(map, ivar, lres=(70, 100), apod_corr=1):
    # Apodize
    white = map * ivar**0.5
    # Compute the 2d spectrun
    ps2d = np.abs(enmap.fft(white)) ** 2 / apod_corr
    del white
    # Smooth it. Using downgrade/upgrade is crude, but
    # has the advantage of being local, so that stong values
    # don't leak far
    down = np.maximum(1, utils.nint(lres / ps2d.lpixshape()))
    down[1] = max(down[1], 4)
    ps2d = enmap.upgrade(
        enmap.downgrade(ps2d, down, inclusive=True),
        down,
        inclusive=True,
        oshape=ps2d.shape,
    )
    return 1 / ps2d


def overlapping_range_iterator(n, nblock, overlap, padding=0):
    from pixell.bunch import Bunch

    if nblock == 0:
        return
    # We don't handle overlap > half our block size
    min_bsize = n // nblock
    if min_bsize == 0:
        raise ValueError(
            "Empty blocks in overlapping_range_iterator. Too low n or too high nblock? n = %d nblock = %d"
            % (n, nblock)
        )
    overlap = min(overlap, min_bsize // 2)
    for bi in range(nblock):
        # Range we would have had without any overlap etc
        i1 = bi * n // nblock
        i2 = (bi + 1) * n // nblock
        # Full range including overlap and padding
        ifull1 = max(i1 - overlap - padding, 0)
        ifull2 = min(i2 + overlap + padding, n)
        # Range that crops away padding
        iuse1 = max(i1 - overlap, 0)
        iuse2 = min(i2 + overlap, n)
        ionly1 = i1 + overlap if bi > 0 else 0
        ionly2 = i2 - overlap if bi < nblock - 1 else n
        nover1 = ionly1 - iuse1
        nover2 = iuse2 - ionly2
        left = (np.arange(nover1) + 1) / (nover1 + 2)
        right = (np.arange(nover2) + 2)[::-1] / (nover2 + 2)
        middle = np.full(ionly2 - ionly1, 1.0)
        weight = np.concatenate([left, middle, right])
        yield Bunch(
            i1=ifull1, i2=ifull2, p1=iuse1 - ifull1, p2=ifull2 - iuse2, weight=weight
        )


def highpass_ips2d(ips2d, lknee, alpha=-20):
    l = np.maximum(ips2d.modlmap(), 0.5)  # noqa: E741
    return ips2d * (1 + (l / lknee) ** alpha) ** -1


def matched_filter_depth1_map(
    imap: enmap,
    ivarmap: enmap,
    freq_ghz: float,
    infofile: str = None,
    maskfile: str = None,
    beam_fwhm: float = None,
    beam1d: str = None,
    shrink_holes: float = 0 * utils.arcmin,
    apod_edge: float = 10 * utils.arcmin,
    apod_holes: float = 5 * utils.arcmin,
    noisemask_lim: float = 0.01,
    highpass: bool = False,
    band_height: float = 0 * utils.degree,
    shift: float = 0,
    simple_lknee: float = 1000,
    simple_alpha: float = -3.5,
    lres=(70, 100),
    pixwin: str = "nn",
    simple: bool = True,
):
    """
    copy of the script https://github.com/amaurea/tenki/blob/master/filter_depth1.py converted to a function.

    Arguments:
    -----------
    imap:enmap
        intensity map, uK

    ivarmap:enmap
        inverse variance map, 1./uK^2

    freq_ghz:float
        observing band, in GHz

    infofile:str [None]
        the .info file for the observation in case shifting is nonzero.

    maskfile:str [None]
        a mask file to apply to the observation

    beam_fwhm:float [None]
        fwhm of the beam, in radian.

    beam1d:str=None
        path to 1D b_l file (i think...)

    shrink_holes:float [0]
        hole size under which to ignore, radians

    apod_edge:float [10*utils.arcmin]
        apodize this far from the map edge, radians

    apod_holes:float [5*utils.arcmin]
        apodize this far around holes, radians

    noisemask_lim:float [0.01]
        an upper limit to the noise, above which you mask. mJy/sr

    highpass:bool [False]
        perform highpass filtering

    band_height [0*utils.degree]
        do filtering in dec bands of this height if >0, else one filtering for entire map, in radians.

    shift = 0
    simple_lknee = 1000,
    simple_alpha = -3.5,
    lres = (70,100),
    pixwin = 'nn',
    simple=True


    Returns:
    -----------
    rho,kappa : enmap
        the matched filtered maps rho and kappa.
    """
    from pixell import analysis, bunch, enmap, uharm, utils

    freq = freq_ghz * 1e9
    ## uK-> mJy/sr
    fconv = utils.dplanck(freq) / 1e3

    if beam1d:
        beam = np.loadtxt(beam1d).T[1]
        beam /= np.max(beam)
        beamis2d = False
    elif beam_fwhm:
        bsigma = beam_fwhm * utils.fwhm
        uht = uharm.UHT(imap.shape, imap.wcs)
        beam = np.exp(-0.5 * uht.l**2 * bsigma**2)
        beamis2d = True
    else:
        raise "Need one of beam1d or beam_fwhm"

    ## convert map in T_cmb to mJy/sr
    imap *= fconv
    ivarmap /= fconv**2

    if shift > 0:
        info = bunch.read(infofile)
    ny, _ = imap.shape[-2:]

    # Build our shift matrix
    if shift > 0:
        S = ShiftMatrix(imap.shape, map.wcs, info.profile)
    else:
        S = ShiftDummy()
    # Set up apodization. A bit messy due to handling two types of apodizion
    # depending on whether it's based on the extrnal mask or not
    hit = ivarmap > 0
    if shrink_holes > 0:
        hit = enmap.shrink_mask(enmap.grow_mask(hit, shrink_holes), shrink_holes)
    # Apodize the edge by decreasing the significance in ivar
    noise_apod = enmap.apod_mask(hit, apod_edge)
    # Check if we have a noise model mask too
    mask = hit
    if maskfile:
        mask |= enmap.read_map(maskfile, geometry=imap.geometry)
    # Optionally mask very bright regions
    if noisemask_lim:
        bright = np.abs(imap.preflat[0] < noisemask_lim * fconv)
        rmask = 5 * utils.arcmin
        mask |= bright.distance_transform(rmax=rmask) < rmask
        del bright
    mask = np.asanyarray(mask)
    if mask.size > 0 and mask.ndim > 0:
        noise_apod *= enmap.apod_mask(1 - mask, apod_holes)
    del mask
    # Build the noise model
    iC = build_ips2d_udgrade(
        S.forward(imap),
        S.forward(ivarmap * noise_apod),
        apod_corr=np.mean(noise_apod**2),
        lres=lres,
    )
    del noise_apod
    if highpass > 0:
        iC = highpass_ips2d(iC, simple_lknee)

    # Set up output map buffers
    rho = imap * 0
    kappa = imap * 0
    tot_weight = np.zeros(ny, imap.dtype)
    # Bands. At least band_height in height and with at least
    # 2*apod_edge of overlapping padding at top and bottom. Using narrow bands
    # make the flat sky approximation a good approximation
    nband = utils.ceil(imap.extent()[0] / band_height) if band_height > 0 else 1
    bedge = utils.ceil(apod_edge / imap.pixshape()[0]) * 2
    boverlap = bedge
    for _, r in enumerate(
        overlapping_range_iterator(ny, nband, boverlap, padding=bedge)
    ):
        # Get band-local versions of the map etc.
        bmap, bivar, bhit = [a[..., r.i1 : r.i2, :] for a in [imap, ivarmap, hit]]
        bny, _ = bmap.shape[-2:]
        if shift > 0:
            bS = ShiftMatrix(bmap.shape, bmap.wcs, info.profile)
        else:
            bS = ShiftDummy()
        # Translate iC to smaller fourier space
        if simple:
            bl = bmap.modlmap()
            biC = (1 + (np.maximum(bl, 0.5) / simple_lknee) ** simple_alpha) ** -1
        else:
            biC = enmap.ifftshift(
                enmap.resample(enmap.fftshift(iC), bmap.shape, method="spline", order=1)
            )
            biC.wcs = bmap.wcs.deepcopy()
        # 2d beam
        if beamis2d:
            bsigma = beam_fwhm * utils.fwhm
            uht = uharm.UHT(bmap.shape, bmap.wcs)
            beam2d = np.exp(-0.5 * uht.l**2 * bsigma**2)
        else:
            beam2d = enmap.samewcs(
                utils.interp(bmap.modlmap(), np.arange(len(beam)), beam), bmap
            )
        # Pixel window. We include it as part of the 2d beam, which is valid in the flat sky
        # approximation we use here.
        if pixwin in ["nn", "0"]:
            beam2d = enmap.apply_window(beam2d, order=0, nofft=True)
        elif pixwin in ["lin", "bilin", "1"]:
            beam2d = enmap.apply_window(beam2d, order=1, nofft=True)
        elif pixwin in ["none"]:
            pass
        else:
            raise ValueError("Invalid pixel window '%s'" % str(pixwin))
        # Set up apodization
        filter_apod = enmap.apod_mask(bhit, apod_edge)
        bivar = bivar * filter_apod
        del filter_apod
        # Phew! Actually perform the filtering
        uht = uharm.UHT(bmap.shape, bmap.wcs, mode="flat")
        brho, bkappa = analysis.matched_filter_constcorr_dual(
            bmap, beam2d, bivar, biC, uht, S=bS.forward, iS=bS.backward
        )
        del uht
        # Restrict to exposed area
        brho *= bhit
        bkappa *= bhit
        # Merge into full output
        weight = r.weight.astype(imap.dtype)
        rho[..., r.i1 + r.p1 : r.i2 - r.p2, :] += (
            brho[..., r.p1 : bny - r.p2, :] * weight[:, None]
        )
        kappa[..., r.i1 + r.p1 : r.i2 - r.p2, :] += (
            bkappa[..., r.p1 : bny - r.p2, :] * weight[:, None]
        )
        tot_weight[r.i1 + r.p1 : r.i2 - r.p2] += weight
        del bmap, bivar, bhit, bS, biC, beam2d, brho, bkappa

    del imap, ivarmap, iC

    return rho, kappa
