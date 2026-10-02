import numpy as np
import pytest
from astropy import units as u
from pixell import enmap
from pixell import utils as pixell_utils

from sotrplib.filters.filters import matched_filter_depth1_map


def test_matched_filter_requires_beam():
    """
    Without beam1d or beam_fwhm, the matched filter raises a ValueError.
    """
    shape, wcs = enmap.geometry(
        pos=np.array([[-1, 1], [1, -1]]) * pixell_utils.degree,
        res=0.5 * pixell_utils.arcmin,
    )
    imap = enmap.zeros(shape, wcs)
    ivarmap = enmap.ones(shape, wcs)

    with pytest.raises(ValueError, match="beam1d or beam_fwhm"):
        matched_filter_depth1_map(imap, ivarmap, band_center=90 * u.GHz)
