"""
Matched-filter implementation for depth-1 SO/ACT maps.

Several functions are adapted from
https://github.com/amaurea/tenki/blob/master/filter_depth1.py
"""

import numpy as np
from astropy import units as u
from pixell import analysis, bunch, enmap, uharm, utils


def fix_profile(profile):
    """Trim a scanning profile that extends beyond the valid azimuth range.

    The profile can be generated over too wide an azimuth range.  This is
    usually harmless, but can cause the dec axis to become non-monotonic if
    the scan crosses north.  This function chops off those regions.

    Parameters
    ----------
    profile : np.ndarray, shape (2, N)
        Scanning profile with rows ``[dec, ra]`` in radians.

    Returns
    -------
    np.ndarray
        Trimmed profile with only the monotonic dec portion retained.
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
    """Operator that shears a map so that scan rows become vertical.

    Given a map geometry and a scanning profile, builds an integer pixel-shift
    vector so that ``forward`` rolls each row to align scans vertically, and
    ``backward`` undoes the shift.

    Parameters
    ----------
    shape : tuple of int
        Map shape ``(ny, nx)``.
    wcs : astropy.wcs.WCS
        WCS for the map.
    profile : np.ndarray, shape (2, N)
        Scanning profile ``[dec, ra]`` in radians.
    """

    def __init__(self, shape, wcs, profile):
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
        """Shift rows so that the scan direction is vertical.

        Parameters
        ----------
        map : enmap.ndmap
            Input map.

        Returns
        -------
        enmap.ndmap
            Row-shifted map.
        """
        from pixell import array_ops

        # Numpy can't do this efficiently
        return array_ops.roll_rows(map, -self.dxs)

    def backward(self, map):
        """Undo the forward shift.

        Parameters
        ----------
        map : enmap.ndmap
            Shifted map.

        Returns
        -------
        enmap.ndmap
            Map with rows restored to their original positions.
        """
        from pixell import array_ops

        return array_ops.roll_rows(map, self.dxs)


class ShiftDummy:
    """No-op replacement for ``ShiftMatrix`` when shifting is disabled."""

    def forward(self, map):
        return map.copy()

    def backward(self, map):
        return map.copy()


class NmatShiftConstCorr:
    """Noise model operator combining a shift matrix, ivar weighting, and a 2-D filter.

    Parameters
    ----------
    S : ShiftMatrix or ShiftDummy
        Shift operator.
    ivar : enmap.ndmap
        Inverse-variance map.
    iC : enmap.ndmap
        Inverse noise power spectrum (2-D).
    """

    def __init__(self, S, ivar, iC):
        self.S = S
        self.H = ivar**0.5
        self.iC = iC

    def apply(self, map, omap=None):
        """Apply the noise model operator to a map.

        Parameters
        ----------
        map : enmap.ndmap
            Input map.
        omap : enmap.ndmap, optional
            Output buffer.  If ``None``, a new map is returned.

        Returns
        -------
        enmap.ndmap
            Filtered map.
        """
        sub = map.extract(self.H.shape, self.H.wcs)
        sub = self.H * self.S.backward(
            enmap.ifft(self.iC * enmap.fft(self.S.forward(self.H * sub))).real
        )
        if omap is None:
            return sub.extract(map.shape, map.wcs)
        else:
            return omap.insert(sub, op=np.add)


def build_ips2d_udgrade(map, ivar, lres=(70, 100), apod_corr=1):
    """Estimate the 2-D inverse noise power spectrum by downgrade/upgrade smoothing.

    Parameters
    ----------
    map : enmap.ndmap
        Whitened signal map (``map * ivar**0.5``).
    ivar : enmap.ndmap
        Inverse-variance map used for whitening.
    lres : tuple of int, optional
        Downgrade factor ``(ly, lx)`` in multipole space.  Default is (70, 100).
    apod_corr : float, optional
        Apodization correction factor applied to the raw power spectrum.
        Default is 1.

    Returns
    -------
    enmap.ndmap
        Inverse of the smoothed 2-D power spectrum.
    """
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
    """Iterate over overlapping index ranges for band-by-band processing.

    Parameters
    ----------
    n : int
        Total length of the axis to split.
    nblock : int
        Number of blocks.
    overlap : int
        Number of pixels of overlap between adjacent blocks.
    padding : int, optional
        Additional padding outside the overlap region.  Default is 0.

    Yields
    ------
    pixell.bunch.Bunch
        Each item has attributes ``i1``, ``i2`` (full range including padding),
        ``p1``, ``p2`` (padding widths at each end), and ``weight`` (taper
        array for the non-padded region).

    Raises
    ------
    ValueError
        If ``nblock`` is so large that individual blocks would be empty.
    """
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
    """Apply a high-pass filter to a 2-D inverse power spectrum.

    Parameters
    ----------
    ips2d : enmap.ndmap
        Inverse 2-D power spectrum to filter.
    lknee : float
        Knee multipole of the high-pass filter.
    alpha : float, optional
        Power-law slope of the filter.  Default is -20 (very sharp).

    Returns
    -------
    enmap.ndmap
        High-pass-filtered inverse power spectrum.
    """
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
    """Apply a matched filter to a depth-1 intensity map.

    Adapted from
    https://github.com/amaurea/tenki/blob/master/filter_depth1.py.
    Converts the input map from CMB temperature units (K) to mJy/sr, builds a
    data-driven (or simple power-law) 2-D noise model, and runs the
    ``pixell.analysis.matched_filter_constcorr_dual`` filter in declination
    bands.

    Parameters
    ----------
    imap : enmap.ndmap
        Intensity map in CMB temperature units (K).
    ivarmap : enmap.ndmap
        Inverse-variance map in K\ :sup:`-2`.
    band_center : Quantity
        Observing band centre frequency (e.g. ``90 * u.GHz``).
    infofile : str, optional
        Path to the ``.info`` HDF file for the observation.  Required when
        ``shift > 0``.
    maskfile : str, optional
        Path to an external mask FITS file (1 = masked).
    source_mask : enmap.ndmap, optional
        Source mask (1 = unmasked, 0 = masked) applied before noise
        estimation.
    beam_fwhm : Quantity, optional
        Gaussian beam FWHM.  Either ``beam_fwhm`` or ``beam1d`` is required.
    beam1d : str, optional
        Path to a 1-D beam profile file with columns ``[r (arcsec), B(r)]``.
        Takes precedence over ``beam_fwhm`` when both are provided.
    shrink_holes : Quantity, optional
        Holes smaller than this radius are filled before apodization.
        Default is 5 arcmin.
    apod_edge : Quantity, optional
        Apodization width at the map edge.  Default is 10 arcmin.
    apod_holes : Quantity, optional
        Apodization width around holes.  Default is 5 arcmin.
    noisemask_lim : float, optional
        Brightness threshold in mJy/sr above which pixels are masked before
        noise estimation.  ``None`` disables this step.
    noisemask_radius : Quantity, optional
        Radius around bright-source pixels to include in the noise mask.
        Default is 10 arcmin.
    highpass : bool, optional
        Apply a high-pass filter to the noise model.  Default is ``False``.
    band_height : Quantity, optional
        Height of each declination processing band.  Set to 0 to process the
        whole map at once.  Default is 1 degree.
    shift : float, optional
        If > 0, apply a scan-direction shift matrix using the info file
        profile.  Default is 0 (no shift).
    simple : bool, optional
        Use a simple power-law noise model instead of estimating it from the
        data.  Default is ``False``.
    simple_lknee : float, optional
        Knee multipole for the simple noise model.  Default is 1000.
    simple_alpha : float, optional
        Power-law index for the simple noise model.  Default is -3.5.
    lres : tuple of int, optional
        ``(ly, lx)`` block size for smoothing the estimated noise spectrum.
        Default is (70, 100).
    pixwin : {"nn", "lin", "none"}, optional
        Pixel window function to include in the beam model.  ``"nn"`` applies
        nearest-neighbour, ``"lin"`` applies bilinear, ``"none"`` skips it.
        Default is ``"nn"``.

    Returns
    -------
    rho : enmap.ndmap
        Matched-filter numerator map in mJy/sr units.
    kappa : enmap.ndmap
        Matched-filter denominator map.
    """

    freq = band_center.to(u.Hz).value
    ## dplanck is Jy/sr / K
    ## to get mJy/sr /K you want 1e3 fconv
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
    imap.fillbad(inplace=True)  ## doesnt like nans
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
