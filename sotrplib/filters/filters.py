import numpy as np
from astropy import units as u
from pixell import analysis, bunch, enmap, uharm, utils

"""
Lots of functions copied from https://github.com/amaurea/tenki/blob/master/filter_depth1.py

"""


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
        yield bunch.Bunch(
            i1=ifull1, i2=ifull2, p1=iuse1 - ifull1, p2=ifull2 - iuse2, weight=weight
        )


def highpass_ips2d(ips2d, lknee, alpha=-20):
    l = np.maximum(ips2d.modlmap(), 0.5)  # noqa: E741
    return ips2d * (1 + (l / lknee) ** alpha) ** -1


def matched_filter_depth1_map(
    imap: enmap,
    ivarmap: enmap,
    band_center: u.Quantity,
    infofile: str | None = None,
    maskfile: str | None = None,
    source_mask: enmap.ndmap | None = None,
    beam_fwhm: u.Quantity | None = None,
    beam1d: str | None = None,
    shrink_holes: u.Quantity = 5 * u.arcmin,
    apod_edge: u.Quantity = 10 * u.arcmin,
    apod_holes: u.Quantity = 5 * u.arcmin,
    noisemask_lim: float | None = None,
    noisemask_radius: u.Quantity = 10 * u.arcmin,
    highpass: bool = False,
    band_height: u.Quantity = 1 * u.degree,
    shift: float = 0,
    simple: bool = False,
    simple_lknee: float = 1000,
    simple_alpha: float = -3.5,
    lres=(70, 100),
    pixwin: str = "nn",
):
    """
    copy of the script https://github.com/amaurea/tenki/blob/master/filter_depth1.py converted to a function.

    Arguments:
    -----------
    imap:enmap
        intensity map, uK

    ivarmap:enmap
        inverse variance map, 1./uK^2

    band_center: u.Quantity
        observing band center frequency

    infofile: str | None = None
        the .info file for the observation in case shifting is nonzero.

    maskfile: str | None = None
        a mask file to apply to the observation

    source_mask: enmap.ndmap | None = None
        a source mask to apply to the observation, with 1==unmasked, 0==masked

    beam_fwhm: u.Quantity | None = None
        fwhm of the Gaussian beam, in radian. e.g. 2.2*utils.arcmin for f090

    beam1d: str | None = None
        beam profile file, B(r), where col[0]=r, col[1]=B(r).
        currently, r assumed to be in arcseconds

    shrink_holes: u.Quantity = 20 * utils.arcmin
        hole size under which to ignore

    apod_edge: u.Quantity = 10 * utils.arcmin
        apodize this far from the map edge

    apod_holes: u.Quantity = 5 * utils.arcmin
        apodize this far around holes

    noisemask_lim: float | None = None
        an upper limit to the noise, above which you mask. mJy/sr

    noisemask_radius: u.Quantity = 10 * utils.arcmin
        radius around bright sources to mask

    highpass: bool = False
        perform highpass filtering

    band_height: u.Quantity = 1 * utils.degree
        do filtering in dec bands of this height if >0, else one filtering for entire map, in radians.

    shift: int = 0
        apply a shift matrix to the data according to the infofile. 0 means no shift

    simple: bool = False
        do a simple filtering using a noise model with lknee and alpha. If False it makes the noise model
        from the data itself

    simple_lknee: float = 1000
        simple filtering lknee, values vary significant among frequencies e.g. ACT full maps had lknee
        intensity values of 2100,3000,3800 for f090, f150, f220, respectively (See Naess 2025 2503.14451)

    simple_alpha: float = -3.5
        simple filtering alpha, values are usually around -3 and -4, depending on the experiment

    lres: tuple = (70,100)
        y,x block size to use when smoothing the noise spectrum

    pixwin: str = "nn"
        apply pixel window function. This is included as part of the beam, which is valid in the flat sky
        posible values are nearest neighbor "nn" "0" bilinear "lin" "bilin" "1" or "none"

    Returns:
    -----------
    rho,kappa : enmap
        the matched filtered maps rho and kappa.
    """

    freq = band_center.to(u.Hz).value
    ## uK-> mJy/sr
    fconv = utils.dplanck(freq) * 1e3

    if beam1d:
        beamdata = np.loadtxt(beam1d).T
        beam_response = beamdata[1]
        beam_response_radius = (beamdata[0] * u.arcsec).to_value(u.rad)
        uht = uharm.UHT(imap.shape, imap.wcs)
        beam = uht.rprof2hprof(beam_response, beam_response_radius)
        beam /= np.max(beam)
    elif beam_fwhm is not None:
        bsigma = beam_fwhm.to(u.radian).value * utils.fwhm
        uht = uharm.UHT(imap.shape, imap.wcs)
        beam = np.exp(-0.5 * uht.l**2 * bsigma**2)
        beam_response = None
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
        hit = enmap.shrink_mask(
            enmap.grow_mask(hit, shrink_holes.to(u.radian).value),
            shrink_holes.to(u.radian).value,
        )
    # Apodize the edge by decreasing the significance in ivar
    noise_apod = enmap.apod_mask(hit, apod_edge.to(u.radian).value)
    # Check if we have a noise model mask too
    mask = 0
    if maskfile:
        mask |= enmap.read_map(maskfile, geometry=imap.geometry)
    ## Apply source mask if given
    if source_mask is not None:
        mask |= 1 - source_mask

    # Optionally mask very bright regions
    if noisemask_lim:
        bright = np.abs(imap.preflat[0]) < noisemask_lim * fconv
        mask |= (
            bright.distance_transform(rmax=noisemask_radius.to(u.radian).value)
            < noisemask_radius.to(u.radian).value
        )
        del bright
    mask = np.asanyarray(mask)
    if mask.size > 0 and mask.ndim > 0:
        noise_apod *= enmap.apod_mask(1 - mask, apod_holes.to(u.radian).value)
    del mask
    imap *= noise_apod

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
    nband = (
        utils.ceil(imap.extent()[0] / band_height.to(u.radian).value)
        if band_height > 0
        else 1
    )
    bedge = utils.ceil(apod_edge.to(u.radian).value / imap.pixshape()[0]) * 2
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
        if beam_response is None:
            bsigma = beam_fwhm.to(u.radian).value * utils.fwhm
            uht = uharm.UHT(bmap.shape, bmap.wcs)
            beam2d = np.exp(-0.5 * uht.l**2 * bsigma**2)
        else:
            uht = uharm.UHT(bmap.shape, bmap.wcs)
            beam2d = uht.rprof2hprof(beam_response, beam_response_radius)
            beam2d /= np.max(beam2d)

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
        filter_apod = enmap.apod_mask(bhit, apod_edge.to(u.radian).value)
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
