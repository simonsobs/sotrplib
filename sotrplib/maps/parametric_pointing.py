"""
Parametric pointing-residual models with a fixed, small number of parameters.

Each model gives the pointing offset (catalogue - fit, on-sky, in arcsec) as a
function of a map coordinate. The model is stored as a few numbers, so mapcat
can keep it, and it can be evaluated at any position of the map.

Two coordinates are available:

- "tangent": a tensor Legendre basis in a tangent plane at the centre of the
  pointing sources. The order is 0 to ``max_order``.
- "time": a piecewise-linear function of the observation time, with 1 to
  ``max_nodes`` equally spaced nodes (as in ACT DR6). To evaluate it, give the
  time of each position (for example from the depth-1 time map).

The fit:

1. Gate and weight the sources, and clip outliers (see ``robust_pointing``).
2. Fit a ridge-regularised linear model. The constant term is free. The other
   terms have a Gaussian prior of width ``prior``, so that with few sources
   the model goes back to a constant offset.
3. Select the basis size and the prior width with the closed-form
   leave-one-out chi-squared.
4. Clamp the evaluation coordinate to the range that the sources cover. Thus,
   the model does not extrapolate.

The model has two directions:

- ``sky_to_map``: catalogue position -> position in the map (pos - offset).
  Use it for forced photometry.
- ``map_to_sky``: position measured in the map -> sky position (pos + offset).
  Use it for blind detections and transients.

The RA part of the offset is an on-sky angle, so it is divided by cos(Dec)
before it is applied to RA.
"""

from typing import Literal

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from mapcat.pointing.base import PointingModelProtocol, PointingModelStats
from numpy.polynomial import legendre
from structlog.types import FilteringBoundLogger

from sotrplib.maps.robust_pointing import CONDON_FACTOR, _gp_fit, _unit_vectors

ARCSEC_PER_DEG = 3600.0


def _tangent_xy(ra, dec, ra0, dec0):
    """Gnomonic projection about (ra0, dec0). Returns x (east), y (north) in deg."""
    ra, dec = np.deg2rad(np.asarray(ra, float)), np.deg2rad(np.asarray(dec, float))
    a0, d0 = np.deg2rad(ra0), np.deg2rad(dec0)
    cosc = np.sin(d0) * np.sin(dec) + np.cos(d0) * np.cos(dec) * np.cos(ra - a0)
    x = np.cos(dec) * np.sin(ra - a0) / cosc
    y = (np.cos(d0) * np.sin(dec) - np.sin(d0) * np.cos(dec) * np.cos(ra - a0)) / cosc
    return np.rad2deg(x), np.rad2deg(y)


def _legendre_terms(order: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]


def _legendre_design(u_, v_, order):
    cols = []
    for i, j in _legendre_terms(order):
        ci = np.zeros(i + 1)
        ci[i] = 1
        cj = np.zeros(j + 1)
        cj[j] = 1
        cols.append(legendre.legval(u_, ci) * legendre.legval(v_, cj))
    return np.stack(cols, axis=-1)


def _hat_design(t, nodes):
    """Piecewise-linear (hat function) basis at the nodes. One node = constant."""
    t = np.atleast_1d(np.asarray(t, float))
    if len(nodes) == 1:
        return np.ones((len(t), 1))
    t = np.clip(t, nodes[0], nodes[-1])
    a = np.zeros((len(t), len(nodes)))
    k = np.clip(np.searchsorted(nodes, t, side="right") - 1, 0, len(nodes) - 2)
    w = (t - nodes[k]) / (nodes[k + 1] - nodes[k])
    a[np.arange(len(t)), k] = 1 - w
    a[np.arange(len(t)), k + 1] = w
    return a


def _hull(u_, v_) -> list[tuple[float, float]]:
    """Convex hull (counter-clockwise vertices) of the points, monotone chain."""
    pts = sorted(set(zip(np.round(u_, 6).tolist(), np.round(v_, 6).tolist())))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _clamp_to_hull(u_, v_, hull):
    """Move points outside the convex polygon to the nearest point on its edge."""
    h = np.asarray(hull, dtype=float)
    if len(h) < 3:
        return u_, v_
    p = np.stack([u_, v_], axis=-1)
    a, b = h, np.roll(h, -1, axis=0)
    e = b - a
    rel = p[:, None, :] - a[None, :, :]
    inside = np.all(
        e[None, :, 0] * rel[..., 1] - e[None, :, 1] * rel[..., 0] >= 0, axis=1
    )
    t = np.clip(np.sum(rel * e[None], -1) / np.sum(e * e, -1)[None], 0, 1)
    proj = a[None] + t[..., None] * e[None]
    d2 = np.sum((p[:, None, :] - proj) ** 2, -1)
    near = proj[np.arange(len(p)), np.argmin(d2, axis=1)]
    out = np.where(inside[:, None], p, near)
    return out[:, 0], out[:, 1]


def _ridge(a, y, sig, penalty):
    """
    Weighted ridge fit. ``penalty`` is the diagonal of the prior precision.
    Returns the coefficients and the leave-one-out residuals.
    """
    w = 1.0 / sig**2
    aw = a * w[:, None]
    m = a.T @ aw + np.diag(penalty)
    minv = np.linalg.pinv(m)
    c = minv @ (aw.T @ y)
    hdiag = np.einsum("ij,jk,ik->i", a, minv, aw)
    resid = y - a @ c
    loo = resid / np.clip(1.0 - hdiag, 1e-6, None)
    return c, loo


class ParametricPointingModel(PointingModelProtocol):
    """
    A pointing model with a fixed set of parameters. See the module docstring.

    The coefficients give the on-sky offset (catalogue - fit) in arcsec.
    """

    model_type: Literal["parametric"] = "parametric"

    coordinate: Literal["tangent", "time"]
    # tangent: Legendre order and the projection; time: hat-function nodes [s].
    order: int = 0
    ra0: float = 0.0
    dec0: float = 0.0
    scale: float = 1.0
    u_range: tuple[float, float] = (-1.0, 1.0)
    v_range: tuple[float, float] = (-1.0, 1.0)
    # Convex hull of the sources in (u, v). Positions outside it are moved to
    # its edge, so the model does not extrapolate into regions without sources.
    hull: list[tuple[float, float]] | None = None
    nodes: tuple[float, ...] = (0.0,)
    coeffs_ra: list[float]
    coeffs_dec: list[float]
    # Range of the offsets (arcsec) that the model can give. The fit sets it
    # to the range of the fitted offsets at the sources plus a margin, so
    # that the model cannot give a larger offset in a region without sources.
    range_ra: tuple[float, float] | None = None
    range_dec: tuple[float, float] | None = None

    @property
    def n_parameters(self) -> int:
        return 2 * len(self.coeffs_ra)

    def _design(self, pos: SkyCoord | None, times=None):
        if len(self.coeffs_ra) == 1:
            # A constant model needs neither the positions nor the times.
            n = (
                len(np.atleast_1d(times))
                if times is not None
                else len(np.atleast_1d(pos.ra))
            )
            return np.ones((n, 1))
        if self.coordinate == "time":
            if times is None:
                raise ValueError("A time-based pointing model needs the times.")
            return _hat_design(times, np.asarray(self.nodes))
        x, y = _tangent_xy(
            np.atleast_1d(pos.ra.to_value(u.deg)),
            np.atleast_1d(pos.dec.to_value(u.deg)),
            self.ra0,
            self.dec0,
        )
        uu = np.clip(x / self.scale, *self.u_range)
        vv = np.clip(y / self.scale, *self.v_range)
        if self.hull is not None:
            uu, vv = _clamp_to_hull(uu, vv, self.hull)
        return _legendre_design(uu, vv, self.order)

    def offsets_at(self, pos: SkyCoord | None = None, times=None):
        """On-sky offsets (catalogue - fit) in arcsec at the positions or times."""
        a = self._design(pos, times)
        off_ra = a @ np.asarray(self.coeffs_ra)
        off_dec = a @ np.asarray(self.coeffs_dec)
        if self.range_ra is not None:
            off_ra = np.clip(off_ra, *self.range_ra)
        if self.range_dec is not None:
            off_dec = np.clip(off_dec, *self.range_dec)
        return off_ra, off_dec

    def _shift(self, pos: SkyCoord, sign: float, times=None) -> SkyCoord:
        off_ra, off_dec = self.offsets_at(pos, times)
        dec = np.atleast_1d(pos.dec.to_value(u.deg))
        dra = sign * off_ra / ARCSEC_PER_DEG / np.cos(np.deg2rad(dec))
        ddec = sign * off_dec / ARCSEC_PER_DEG
        ra = np.atleast_1d(pos.ra.to_value(u.deg)) + dra
        dec = dec + ddec
        if pos.isscalar:
            ra, dec = ra[0], dec[0]
        return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame=pos.frame)

    def sky_to_map(self, pos: SkyCoord, times=None) -> SkyCoord:
        """Catalogue position -> position of the source in the map."""
        return self._shift(pos, -1.0, times)

    def map_to_sky(self, pos: SkyCoord, times=None) -> SkyCoord:
        """Position measured in the map -> sky position."""
        return self._shift(pos, +1.0, times)

    def predict(self, pos: SkyCoord, times=None) -> SkyCoord:
        # Same direction as ConstantPointingModel.predict (catalogue -> map).
        return self.sky_to_map(pos, times)


class ParametricPointingOffset:
    """
    Fit a ``ParametricPointingModel`` to the pointing sources of one map.

    Parameters
    ----------
    coordinate : {"tangent", "time"}
    max_order : int
        Largest Legendre order for "tangent".
    node_choices : tuple[int, ...]
        Numbers of time nodes to try for "time".
    prior_choices : tuple[float, ...]
        Prior widths (arcsec) to try for the non-constant terms.
    The gate and clip parameters are the same as for ``RobustPointingOffset``.
    """

    def __init__(
        self,
        coordinate: Literal["tangent", "time"] = "tangent",
        max_order: int = 3,
        node_choices: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12),
        prior_choices: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0),
        min_snr: float = 5.0,
        min_num: int = 3,
        max_offset: u.Quantity = 5.0 * u.arcmin,
        max_model_offset: u.Quantity = 5.0 * u.arcmin,
        clip_sigma: float = 4.0,
        sys_floor: u.Quantity = 2.0 * u.arcsec,
        fwhm_snr: float = 10.0,
        log: FilteringBoundLogger | None = None,
    ):
        self.coordinate = coordinate
        self.max_order = max_order
        self.node_choices = node_choices
        self.prior_choices = prior_choices
        self.min_snr = min_snr
        self.min_num = min_num
        self.max_offset = max_offset
        self.max_model_offset = max_model_offset
        self.clip_sigma = clip_sigma
        self.sys_floor = sys_floor
        self.fwhm_snr = fwhm_snr
        self.log = log or structlog.get_logger()

    def _zero(self):
        return ParametricPointingModel(
            coordinate="tangent", coeffs_ra=[0.0], coeffs_dec=[0.0]
        )

    def fit_arrays(self, ra, dec, dra, ddec, snr, fwhm=None, times=None):
        """
        Fit the model. ``dra``/``ddec`` are on-sky offsets in arcsec, ``fwhm``
        in arcsec, ``times`` in seconds (needed for "time").
        Returns the model, the statistics and the mask of the sources used.
        """
        log = self.log.bind(
            func="ParametricPointingOffset.fit_arrays", coordinate=self.coordinate
        )
        ra, dec, dra, ddec, snr = (
            np.asarray(x, dtype=float) for x in (ra, dec, dra, ddec, snr)
        )
        t = None if times is None else np.asarray(times, dtype=float)
        max_off = self.max_offset.to_value(u.arcsec)
        ok = (
            np.isfinite(ra)
            & np.isfinite(dec)
            & np.isfinite(dra)
            & np.isfinite(ddec)
            & np.isfinite(snr)
            & (snr >= self.min_snr)
            & (np.hypot(dra, ddec) <= max_off)
        )
        if self.coordinate == "time":
            if t is None:
                raise ValueError("The time-based model needs the source times.")
            ok &= np.isfinite(t)

        beam = np.nan
        if fwhm is not None:
            fw = np.asarray(fwhm, dtype=float)
            good = ok & np.isfinite(fw) & (snr >= self.fwhm_snr)
            if good.sum() >= 3:
                beam = float(np.median(fw[good]))
        if not np.isfinite(beam):
            beam = 120.0
        sig = np.hypot(
            CONDON_FACTOR * beam / np.maximum(snr, 1e-3),
            self.sys_floor.to_value(u.arcsec),
        )

        if ok.sum() >= self.min_num:
            z = np.hypot(
                (dra - np.median(dra[ok])) / sig, (ddec - np.median(ddec[ok])) / sig
            )
            ok &= z <= 3.0 * self.clip_sigma
        if ok.sum() < self.min_num:
            log.warning(
                "ParametricPointingOffset.not_enough_sources", n_valid=int(ok.sum())
            )
            return self._zero(), PointingModelStats(n_sources=int(ok.sum())), ok

        # Clip with a smooth GP in the fit coordinate (leave-one-out), so that
        # real structure is not clipped.
        if self.coordinate == "time":
            # Map time to a 1D "position" on a great circle: 1 deg per 10 min.
            t_rel = np.where(np.isfinite(t), t - np.nanmin(t), 0.0)
            vec = _unit_vectors(t_rel / 600.0, np.zeros_like(t))
        else:
            vec = _unit_vectors(ra, dec)
        while True:
            idx = np.flatnonzero(ok)
            fits = [
                _gp_fit(vec[idx], y[idx], sig[idx], 5.0, 10.0, 2.0 * max_off)
                for y in (dra, ddec)
            ]
            z2 = sum(f[2] ** 2 / f[3] for f in fits)
            worst = int(np.argmax(z2))
            if z2[worst] <= self.clip_sigma**2 or len(idx) <= self.min_num:
                break
            ok[idx[worst]] = False
        idx = np.flatnonzero(ok)

        # Candidate designs.
        params: dict = {"coordinate": self.coordinate}
        if self.coordinate == "time":
            t0, t1 = float(t[idx].min()), float(t[idx].max())
            cands = []
            for k in self.node_choices:
                if k > max(1, len(idx) // 3):
                    continue
                nodes = np.linspace(t0, t1, k) if k > 1 else np.array([t0])
                # The node values bound a piecewise-linear model, so the
                # shape check uses the nodes as the grid.
                cands.append(
                    (
                        {"nodes": tuple(nodes.tolist())},
                        _hat_design(t[idx], nodes),
                        np.eye(len(nodes)),
                    )
                )
        else:
            v = _unit_vectors(ra[idx], dec[idx]).sum(0)
            ra0 = float(np.rad2deg(np.arctan2(v[1], v[0])) % 360)
            dec0 = float(np.rad2deg(np.arcsin(v[2] / np.linalg.norm(v))))
            x, y = _tangent_xy(ra[idx], dec[idx], ra0, dec0)
            scale = float(max(np.abs(x).max(), np.abs(y).max(), 1e-3))
            uu, vv = x / scale, y / scale
            params.update(
                ra0=ra0,
                dec0=dec0,
                scale=scale,
                u_range=(float(uu.min()), float(uu.max())),
                v_range=(float(vv.min()), float(vv.max())),
                hull=[(float(a), float(b)) for a, b in _hull(uu, vv)],
            )
            cands = []
            for order in range(self.max_order + 1):
                if len(_legendre_terms(order)) > max(1, len(idx) // 3):
                    continue
                gu, gv = np.meshgrid(
                    np.linspace(uu.min(), uu.max(), 41),
                    np.linspace(vv.min(), vv.max(), 41),
                )
                gu_h, gv_h = _clamp_to_hull(gu.ravel(), gv.ravel(), params["hull"])
                cands.append(
                    (
                        {"order": order},
                        _legendre_design(uu, vv, order),
                        _legendre_design(gu_h, gv_h, order),
                    )
                )

        best = None
        for extra, a, grid in cands:
            for prior in self.prior_choices if a.shape[1] > 1 else (np.inf,):
                pen = np.full(a.shape[1], 1.0 / prior**2)
                if self.coordinate == "tangent":
                    pen[0] = 0.0  # constant term free
                else:
                    # Hat functions: penalise departures from the weighted mean
                    # by a prior on each node; the mean is set by the data.
                    pen[:] = 1.0 / prior**2 if a.shape[1] > 1 else 0.0
                score, coefs, sane = 0.0, [], True
                for yv in (dra, ddec):
                    c, loo = _ridge(
                        a,
                        yv[idx]
                        - (
                            0
                            if self.coordinate == "tangent"
                            else np.average(yv[idx], weights=1 / sig[idx] ** 2)
                        ),
                        sig[idx],
                        pen,
                    )
                    score += np.sum((loo / sig[idx]) ** 2)
                    coefs.append(c)
                    # Shape check: in the box, the model must stay near the
                    # range of its values at the sources.
                    f_src, f_grid = a @ c, grid @ c
                    lo, hi = f_src.min(), f_src.max()
                    slack = 0.5 * (hi - lo) + 2.0
                    if f_grid.min() < lo - slack or f_grid.max() > hi + slack:
                        sane = False
                if not sane and a.shape[1] > 1:
                    continue
                if best is None or score < best[0]:
                    best = (score, extra, coefs, a, prior)
        score, extra, (c_ra, c_dec), a, prior = best
        ranges = []
        for c in (c_ra, c_dec):
            f_src = a @ c
            margin = max(2.0, 0.1 * (f_src.max() - f_src.min()))
            ranges.append([float(f_src.min() - margin), float(f_src.max() + margin)])
        if self.coordinate == "time":
            # Add back the weighted means removed before the fit.
            w = 1 / sig[idx] ** 2
            m_ra = np.average(dra[idx], weights=w)
            m_dec = np.average(ddec[idx], weights=w)
            c_ra = c_ra + m_ra
            c_dec = c_dec + m_dec
            ranges[0] = [ranges[0][0] + m_ra, ranges[0][1] + m_ra]
            ranges[1] = [ranges[1][0] + m_dec, ranges[1][1] + m_dec]
        model = ParametricPointingModel(
            **params,
            **extra,
            coeffs_ra=list(map(float, c_ra)),
            coeffs_dec=list(map(float, c_dec)),
            range_ra=tuple(ranges[0]),
            range_dec=tuple(ranges[1]),
        )

        res_ra = dra[idx] - a @ c_ra
        res_dec = ddec[idx] - a @ c_dec
        mean_ra, mean_dec = float(np.mean(a @ c_ra)), float(np.mean(a @ c_dec))
        if np.hypot(mean_ra, mean_dec) > self.max_model_offset.to_value(u.arcsec):
            log.warning("ParametricPointingOffset.model_offset_above_limit")
            return self._zero(), PointingModelStats(n_sources=len(idx)), ok
        stats = PointingModelStats(
            mean_ra_offset=(mean_ra * u.arcsec).to(u.deg),
            mean_dec_offset=(mean_dec * u.arcsec).to(u.deg),
            stddev_ra_offset=(np.std(res_ra) * u.arcsec).to(u.deg),
            stddev_dec_offset=(np.std(res_dec) * u.arcsec).to(u.deg),
            n_sources=len(idx),
        )
        log.info(
            "ParametricPointingOffset.fit",
            n_used=len(idx),
            n_parameters=model.n_parameters,
            prior=prior,
            **(
                extra if self.coordinate == "tangent" else {"n_nodes": len(model.nodes)}
            ),
        )
        return model, stats, ok
