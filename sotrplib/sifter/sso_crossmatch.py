"""
Crossmatch blind-search candidates against solar-system-object trajectories.

A depth-1 map's exposure is short enough that an SSO's position at the
map's single mean/mid time is a good stand-in for "where it was" -- see
SOCat.get_sources_in_map's per-object time refinement. A week coadd is
different: a blind-search candidate's own observation_mean_time is the
*average* crossing time of however many depth-1 visits happened to touch
that pixel, most of which have nothing to do with any asteroid passing
through -- so an SSO that visited that sky position earlier or later in
the week would be missed by checking only "where was it at the
candidate's mean time". Instead, sample each SSO's trajectory across the
coadd's full observation window and check whether the candidate's
position lies within the match radius of *any* sampled point.
"""

from typing import TYPE_CHECKING

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import angular_separation
from astropy.time import Time
from structlog.types import FilteringBoundLogger

from sotrplib.sources.sources import CrossMatch, MeasuredSource

if TYPE_CHECKING:
    from socat.core import SourceGenerator


def sample_trajectory(
    sg: "SourceGenerator",
    t_min: Time,
    t_max: Time,
    cadence: u.Quantity = 2 * u.hour,
) -> tuple[Time, np.ndarray, np.ndarray]:
    """
    Evaluate a socat SourceGenerator's position at a regular cadence across
    [t_min, t_max]. `cadence` should comfortably resolve the object's
    apparent motion relative to the match radius used downstream, or the
    discretized path can "skip over" a candidate between samples.

    Returns (sample_times, ra_rad, dec_rad).
    """
    duration_sec = (t_max - t_min).sec
    n = max(2, int(duration_sec / cadence.to_value(u.s)) + 1)
    sample_times = t_min + (t_max - t_min) * np.linspace(0, 1, n)

    ra = np.empty(n)
    dec = np.empty(n)
    for i, t in enumerate(sample_times):
        pos, _ = sg.at_time(t)
        ra[i] = pos.ra.to_value(u.rad)
        dec[i] = pos.dec.to_value(u.rad)
    return sample_times, ra, dec


def crossmatch_sso_trajectories(
    candidates: list[MeasuredSource],
    sso_generators: dict[str, "SourceGenerator"],
    t_min: Time,
    t_max: Time,
    radius: u.Quantity = 5.0 * u.arcmin,
    cadence: u.Quantity = 2 * u.hour,
    log: FilteringBoundLogger | None = None,
) -> list[MeasuredSource]:
    """
    For each candidate, check whether any sampled point of any SSO's
    trajectory across [t_min, t_max] lies within `radius`. Matched
    candidates get a CrossMatch appended (angular_separation and
    observation_time are the closest-approach values, not the candidate's
    own observation_mean_time) and source_type set to "sso".

    Parameters
    ----------
    candidates : list[MeasuredSource]
        Blind-search candidates to check -- typically noise_candidates +
        transient_candidates, i.e. ones not already matched to a static
        catalog source.
    sso_generators : dict[str, SourceGenerator]
        e.g. from SOCat.get_sso_generators_in_map(coadd).
    t_min, t_max : Time
        The coadd's observation window. Should match what
        sso_generators' trajectories were built over.
    """
    log = log or structlog.get_logger()
    if not sso_generators or not candidates:
        return candidates

    cand_ra = np.array([c.ra.to_value(u.rad) for c in candidates])
    cand_dec = np.array([c.dec.to_value(u.rad) for c in candidates])

    n_matched = 0
    for name, sg in sso_generators.items():
        # Clip to the generator's own valid range in case it's narrower
        # than the coadd window (shouldn't happen if the generator was
        # built with the coadd's own t_min/t_max, but be defensive).
        lo = max(t_min, sg.t_min)
        hi = min(t_max, sg.t_max)
        if lo >= hi:
            continue
        sample_times, traj_ra, traj_dec = sample_trajectory(sg, lo, hi, cadence=cadence)

        # (n_candidates, n_samples) angular separation matrix.
        sep = angular_separation(
            cand_ra[:, None] * u.rad,
            cand_dec[:, None] * u.rad,
            traj_ra[None, :] * u.rad,
            traj_dec[None, :] * u.rad,
        )
        best_idx = np.argmin(sep, axis=1)
        best_sep = sep[np.arange(len(candidates)), best_idx]

        for i, candidate in enumerate(candidates):
            if best_sep[i] > radius:
                continue
            n_matched += 1
            match = CrossMatch(
                source_id=name,
                source_type="sso",
                probability=1.0,
                angular_separation=best_sep[i].to(u.deg),
                catalog_name="socat",
                catalog_idx=name,
                ra=(traj_ra[best_idx[i]] * u.rad).to(u.deg),
                dec=(traj_dec[best_idx[i]] * u.rad).to(u.deg),
                observation_time=sample_times[best_idx[i]],
            )
            if candidate.crossmatches is None:
                candidate.crossmatches = []
            candidate.crossmatches.append(match)
            candidate.source_type = "sso"
            log.info(
                "crossmatch_sso_trajectories.matched",
                sso_name=name,
                candidate_ra=candidate.ra,
                candidate_dec=candidate.dec,
                closest_approach_time=sample_times[best_idx[i]].isot,
                angular_separation=best_sep[i].to(u.arcmin),
            )

    log.info(
        "crossmatch_sso_trajectories.complete",
        n_candidates=len(candidates),
        n_sso=len(sso_generators),
        n_matched=n_matched,
    )
    return candidates
