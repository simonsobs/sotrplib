import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta
from pixell import enmap
from pixell import utils as pixell_utils
from structlog import get_logger

from sotrplib.maps.core import IntensityAndInverseVarianceMap, _units_from_bunit

log = get_logger()


def _write_map(path, bunit: str | None, value: float = 1.0):
    shape, wcs = enmap.geometry(
        pos=np.array([[-1, 1], [1, -1]]) * pixell_utils.degree,
        res=0.5 * pixell_utils.arcmin,
    )
    enmap.write_map(str(path), enmap.full(shape, wcs, value))
    if bunit is not None:
        fits.setval(str(path), "BUNIT", value=bunit)
    return path


@pytest.mark.parametrize(
    "bunit, power, default, expected",
    [
        (None, 1, u.K, u.K),
        ("K", 1, u.K, u.K),
        ("uK", 1, u.K, u.uK),
        ("mJy", 1, u.Jy, u.mJy),
        ("mJy-1", -1, u.Jy, u.mJy),
        # Astropy cannot parse uK_CMB, so use the default.
        ("uK_CMB", 1, u.K, u.K),
        # Not a temperature, so use the default.
        ("Jy/sr", 1, u.K, u.K),
    ],
)
def test_units_from_bunit(tmp_path, bunit, power, default, expected):
    path = _write_map(tmp_path / "map.fits", bunit)
    assert _units_from_bunit(path, power=power, default=default, log=log) == expected


def test_intensity_map_reads_bunit(tmp_path):
    intensity = _write_map(tmp_path / "map.fits", "uK", value=100.0)
    ivar = _write_map(tmp_path / "ivar.fits", None, value=1.0)
    start_time = Time("2025-10-01T00:00:00", format="isot", scale="utc")

    imap = IntensityAndInverseVarianceMap(
        intensity_filename=intensity,
        inverse_variance_filename=ivar,
        start_time=start_time,
        end_time=start_time + TimeDelta(3600, format="sec"),
        frequency="f090",
    )
    imap.build()

    assert imap.intensity_units == u.uK
