"""
Flag sources that sit on poorly-weighted regions of a map.

The weight map is kappa (inverse variance). Thresholds are set per map from
the distribution of its own valid (finite, positive) weights, so a source is
flagged when it, or the region around it, is among the worst-weighted parts
of that map.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropydantic import AstroPydanticQuantity
from pixell.enmap import ndmap
from pydantic import BaseModel

POOR_WEIGHTS_CENTER = "poor_weights_center"
POOR_WEIGHTS_LOCAL = "poor_weights_local"
POOR_WEIGHTS_EXTENDED = "poor_weights_extended"


class PoorWeightsCriteria(BaseModel):
    """
    Three checks, applied in order; the first that trips gives the flag.

    center: weight at the source pixel is at or below the
        ``center_percentile`` of the map's weights.
    local: at least ``local_fraction`` of the pixels within
        ``local_half_width`` are at or below the ``local_percentile``.
    extended: at least ``extended_fraction`` of the pixels within
        ``extended_half_width`` are at or below the ``extended_percentile``.
    """

    center_percentile: float = 1.0
    local_percentile: float = 2.5
    local_half_width: AstroPydanticQuantity[u.arcmin] = u.Quantity(2.5, "arcmin")
    local_fraction: float = 0.25
    extended_percentile: float = 5.0
    extended_half_width: AstroPydanticQuantity[u.arcmin] = u.Quantity(10.0, "arcmin")
    extended_fraction: float = 0.5


class PoorWeightsThresholds(BaseModel):
    center: float
    local: float
    extended: float


def _as_2d(weights: ndmap) -> ndmap:
    return weights if weights.ndim == 2 else weights.preflat[0]


def get_weight_thresholds(
    weights: ndmap, criteria: PoorWeightsCriteria
) -> PoorWeightsThresholds | None:
    """
    Weight values at the criteria's percentiles over the map's valid pixels.
    Returns None if the map has no valid pixels.
    """
    weights = _as_2d(weights)
    valid = weights[np.isfinite(weights) & (weights > 0)]
    if valid.size == 0:
        return None
    center, local, extended = np.percentile(
        valid,
        [
            criteria.center_percentile,
            criteria.local_percentile,
            criteria.extended_percentile,
        ],
    )
    return PoorWeightsThresholds(
        center=float(center), local=float(local), extended=float(extended)
    )


def _low_fraction(
    weights: ndmap, y: int, x: int, half_width_pix: int, threshold: float
) -> float:
    ny, nx = weights.shape
    box = weights[
        max(y - half_width_pix, 0) : min(y + half_width_pix + 1, ny),
        max(x - half_width_pix, 0) : min(x + half_width_pix + 1, nx),
    ]
    if box.size == 0:
        return 1.0
    ## nan weights count as low
    return float(np.mean(~(box > threshold)))


def poor_weights_flag(
    weights: ndmap,
    position: SkyCoord,
    criteria: PoorWeightsCriteria,
    thresholds: PoorWeightsThresholds,
) -> str | None:
    """
    Return the poor-weights flag for a source at ``position``, or None if it
    passes all checks.

    Boxes are square in pixels, (2 * half_width + 1) on a side, and are
    clipped at the map edge.
    """
    weights = _as_2d(weights)
    ny, nx = weights.shape
    pix = weights.sky2pix([position.dec.to_value(u.rad), position.ra.to_value(u.rad)])
    if not np.all(np.isfinite(pix)):
        return POOR_WEIGHTS_CENTER
    y, x = int(np.round(pix[0])), int(np.round(pix[1]))
    if not (0 <= y < ny and 0 <= x < nx):
        return POOR_WEIGHTS_CENTER

    if not weights[y, x] > thresholds.center:
        return POOR_WEIGHTS_CENTER

    pixel_size = abs(weights.wcs.wcs.cdelt[1]) * u.deg

    local_pix = int(np.round((criteria.local_half_width / pixel_size).to_value("")))
    if (
        _low_fraction(weights, y, x, local_pix, thresholds.local)
        >= criteria.local_fraction
    ):
        return POOR_WEIGHTS_LOCAL

    extended_pix = int(
        np.round((criteria.extended_half_width / pixel_size).to_value(""))
    )
    if (
        _low_fraction(weights, y, x, extended_pix, thresholds.extended)
        >= criteria.extended_fraction
    ):
        return POOR_WEIGHTS_EXTENDED

    return None
