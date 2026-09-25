"""
Maps written to FITS record their flux unit in BUNIT, and maps read back
from disk take their flux unit from it -- so e.g. a matched-filtered coadd
in mJy isn't read back as Jy (a silent 1000x error in every flux).
"""

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from pixell import enmap

from sotrplib.maps.core import FluxAndSNRMap, RhoAndKappaMap
from sotrplib.outputs.core import MapOutputSerializer

START = Time("2025-09-18T00:00:00")
END = Time("2025-09-25T00:00:00")


def _write_rhokappa_map(directory, flux_units):
    shape, wcs = enmap.geometry(pos=np.deg2rad([[-1, -1], [1, 1]]), res=np.deg2rad(0.5))
    source = RhoAndKappaMap(
        rho_filename=None,
        kappa_filename=None,
        start_time=START,
        end_time=END,
        frequency="f090",
        array="i1i3",
        flux_units=flux_units,
    )
    source.rho = enmap.ones(shape, wcs) * 2.0
    source.kappa = enmap.ones(shape, wcs) * 4.0
    source.flux = source.rho / source.kappa
    source.snr = source.rho / np.sqrt(source.kappa)
    return MapOutputSerializer(
        directory=directory, field_ids=["rho", "kappa", "flux", "snr"]
    ).output(input_map=source)


def test_serializer_writes_bunit(tmp_path):
    paths = _write_rhokappa_map(tmp_path, flux_units=u.mJy)
    assert fits.getheader(paths["flux"])["BUNIT"] == "mJy"
    assert fits.getheader(paths["rho"])["BUNIT"] == "mJy-1"
    assert fits.getheader(paths["kappa"])["BUNIT"] == "mJy-2"
    # SNR is dimensionless: no flux unit recorded.
    assert "BUNIT" not in fits.getheader(paths["snr"])


def test_rhokappa_map_takes_units_from_bunit(tmp_path):
    paths = _write_rhokappa_map(tmp_path, flux_units=u.mJy)
    m = RhoAndKappaMap(
        rho_filename=paths["rho"],
        kappa_filename=paths["kappa"],
        start_time=START,
        end_time=END,
        flux_units=u.Jy,  # configured default is overridden by the file
    )
    m.build()
    assert m.flux_units == u.mJy


def test_flux_map_takes_units_from_bunit(tmp_path):
    paths = _write_rhokappa_map(tmp_path, flux_units=u.mJy)
    m = FluxAndSNRMap(
        flux_filename=paths["flux"],
        snr_filename=paths["snr"],
        start_time=START,
        end_time=END,
        flux_units=u.Jy,
    )
    m.build()
    assert m.flux_units == u.mJy


def test_map_without_bunit_keeps_configured_units(tmp_path):
    shape, wcs = enmap.geometry(pos=np.deg2rad([[-1, -1], [1, 1]]), res=np.deg2rad(0.5))
    enmap.write_map(str(tmp_path / "rho.fits"), enmap.ones(shape, wcs))
    enmap.write_map(str(tmp_path / "kappa.fits"), enmap.ones(shape, wcs))
    m = RhoAndKappaMap(
        rho_filename=tmp_path / "rho.fits",
        kappa_filename=tmp_path / "kappa.fits",
        start_time=START,
        end_time=END,
        flux_units=u.Jy,
    )
    m.build()
    assert m.flux_units == u.Jy
