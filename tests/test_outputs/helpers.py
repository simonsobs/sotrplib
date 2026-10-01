"""Shared sample data for output backend tests."""

import uuid

import astropy.units as u
import numpy as np
from astropy.time import Time

from sotrplib.sources.sources import CrossMatch, MeasuredSource


def make_candidate(
    socat_id: uuid.UUID, flux_mjy: float, with_thumbnail: bool
) -> MeasuredSource:
    return MeasuredSource(
        source_id="TestSource",
        catalog_name="socat",
        measurement_id=str(uuid.uuid4()),
        ra=10.0 * u.deg,
        dec=5.0 * u.deg,
        err_ra=3.6 * u.arcsec,
        err_dec=7.2 * u.arcsec,
        flux=flux_mjy * u.mJy,
        err_flux=1.0 * u.mJy,
        observation_mean_time=Time("2025-09-10T00:00:00"),
        frequency=90.0 * u.GHz,
        array="pa5",
        crossmatches=[
            CrossMatch(
                source_id="TestSource",
                catalog_idx=socat_id,
                catalog_name="socat",
                ra=10.1 * u.deg,
                dec=5.1 * u.deg,
            )
        ],
        thumbnail=np.zeros((4, 4)) if with_thumbnail else None,
        thumbnail_unit=u.mJy if with_thumbnail else None,
    )
