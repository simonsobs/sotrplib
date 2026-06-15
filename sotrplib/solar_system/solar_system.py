import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from skyfield.api import load, wgs84
from skyfield.data import mpc
from skyfield.toposlib import GeographicPosition
from structlog.types import FilteringBoundLogger
from tqdm import tqdm

from sotrplib.maps.core import ProcessableMap


def create_observer(
    lat: u.Quantity[u.deg] = -22.96098 * u.deg,
    lon: u.Quantity[u.deg] = -67.7876 * u.deg,
    elev: u.Quantity[u.m] = 5180 * u.m,
) -> GeographicPosition:
    """Create a Skyfield observer for the Simons Observatory LAT site."""
    return wgs84.latlon(
        lat.to_value(u.deg), lon.to_value(u.deg), elevation_m=elev.to_value(u.m)
    )


## need to normalize the names in Paul's file to the MPC format
## warning! this doesn't work with comets (probably fine, but we wont have comet ephemerides)
def normalize_asteroid_name(name: str) -> str:
    """
    Convert asteroid names such as:
        '3552 Don Quixote (1983 SA)' → '(3552) Don Quixote'
        '7335 (1989 JA)'             → '(7335) 1989 JA'
    """
    # Extract leading number and the rest
    m = re.match(r"^\s*(\d+)\s*(.*)$", name)
    if not m:
        return name  # fallback if pattern is unexpected

    number, rest = m.groups()

    # Extract provisional designation at end e.g. (1991 CS)
    provisional = None
    m2 = re.search(r"\(([^)]+)\)\s*$", rest)
    if m2:
        provisional = m2.group(1)
        rest = rest[: m2.start()].strip()  # remove the parentheses section

    # Determine final name
    if rest:  # there's a real name
        final_name = rest
    else:  # no real name -> use the provisional designation
        final_name = provisional if provisional else ""

    if final_name:
        return f"({number}) {final_name}"
    else:
        return f"({number})"


def generate_mpc_orbital_database(
    mpcorb_dat_file,
    asteroid_flux_estimates_file: str = "solar_system_objects.txt",
    output: str = "mpc_orbital_params_bright_asteroids.csv",
    flux_min: u.Quantity[u.mJy] = 10 * u.mJy,
    log: FilteringBoundLogger | None = None,
):
    """Build a CSV orbital database of bright asteroids from MPC and flux estimates.

    Parameters
    ----------
    mpcorb_dat_file : str or Path
        Path to the MPC MPCORB.DAT file.
    asteroid_flux_estimates_file : str, optional
        Path to the SPT asteroid flux estimate text file (tab-delimited,
        columns: name, ..., max_flux_mJy).
    output : str, optional
        Output CSV filename (default ``"mpc_orbital_params_bright_asteroids.csv"``).
    flux_min : Quantity[mJy], optional
        Minimum estimated flux for inclusion (default 10 mJy).
    log : FilteringBoundLogger, optional
        Structured logger.
    """
    log = log or structlog.get_logger()
    log = log.bind(function="solar_system.generate_mpc_orbital_database")
    sso_name, _, _, maxflux = np.loadtxt(
        asteroid_flux_estimates_file, delimiter="\t", dtype="str", unpack=True
    )
    maxflux = np.asarray(maxflux, dtype=float) * u.mJy
    bright_sso_names = []
    for i in range(len(sso_name)):
        if (
            (maxflux[i] > flux_min)
            & (~sso_name[i].startswith("C"))
            & (~sso_name[i].startswith("("))
        ):
            bright_sso_names.append(sso_name[i])
    log.info(
        "solar_system.generate_mpc_orbital_database.sso_estimates_loaded",
        n_bright_asterids=len(bright_sso_names),
        flux_min=flux_min.to_value(u.mJy),
        flux_units=str(flux_min.unit),
    )

    with load.open(mpcorb_dat_file) as f:
        minor_planets = mpc.load_mpcorb_dataframe(f)

    log.info(
        "solar_system.generate_mpc_orbital_database.mpcorb_loaded",
        n_minor_planets=len(minor_planets),
    )

    # Filtering the orbits dataframe to avoid triggering
    # an `EphemerisRangeError` on ill-defined orbits.
    bad_orbits = minor_planets.semimajor_axis_au.isnull()
    minor_planets = minor_planets[~bad_orbits]

    ## get mpc file designations and compare to our bright sso names
    mpc_file_des = set(minor_planets.designation.values)
    new_db = []
    for name in tqdm(bright_sso_names, desc="Generating MPC orbital database"):
        des = normalize_asteroid_name(name)
        # Skip if designation is not in the MPC table
        if des not in mpc_file_des:
            continue
        # Append the row as a dict or Series
        row = minor_planets.loc[minor_planets.designation == des].iloc[0]
        new_db.append(row)
    new_db_df = pd.DataFrame(new_db)
    new_db_df.to_csv(output, index=False)
    log.info(
        "solar_system.generate_mpc_orbital_database.database_generated",
        output_file=output,
        n_asteroids=len(new_db_df),
    )

    return


def load_mpc_orbital_database(
    mpc_database_path: str = "mpc_orbital_params_bright_asteroids.csv",
):
    """Load the MPC orbital database CSV generated by ``generate_mpc_orbital_database``.

    Parameters
    ----------
    mpc_database_path : str, optional
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Orbital parameters with numeric columns converted to float where possible.
    """
    df = pd.read_csv(mpc_database_path, dtype=str)

    ## convert to floats where possible. needed for skyfield
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except ValueError:
            pass

    return df


def load_jpl_ephem_database(
    ephem_file_path: str = "JPL_batched_ephemerides_2015-01-01_2025-01-01.parquet",
    start_time: datetime | Time | None = Time("2025-01-01T00:00:00Z"),
    stop_time: datetime | Time | None = Time("2030-01-01T00:00:00Z"),
    log: FilteringBoundLogger | None = None,
) -> pd.DataFrame:
    """Load a JPL Horizons ephemeris parquet file and filter to a time range.

    Expected columns: ``designation``, ``julian_day``, ``ra_deg``, ``dec_deg``,
    and optionally ``distance_au``.  Files can be generated with
    ``download_ephem_from_horizons.py``.

    Parameters
    ----------
    ephem_file_path : str, optional
        Path to the parquet file.
    start_time : datetime or Time, optional
        Lower bound on Julian day (inclusive).
    stop_time : datetime or Time, optional
        Upper bound on Julian day (inclusive).
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    pd.DataFrame
        Filtered ephemeris rows.
    """
    if isinstance(start_time, datetime):
        start_time = Time(start_time)
    if isinstance(stop_time, datetime):
        stop_time = Time(stop_time)

    log = log or structlog.get_logger()
    log.info(
        "solar_system.load_jpl_ephem_database.loading_ephemerides",
        ephem_file_path=str(ephem_file_path),
        start_time=start_time.iso if start_time else None,
        stop_time=stop_time.iso if stop_time else None,
    )
    df = pd.read_parquet(ephem_file_path)
    if start_time is not None:
        df = df[df["julian_day"].values >= start_time.jd]
    if stop_time is not None:
        df = df[df["julian_day"].values <= stop_time.jd]
    log.info(
        "solar_system.load_jpl_ephem_database.ephemerides_loaded",
        ephem_file_path=str(ephem_file_path),
        start_time=start_time.iso if start_time else None,
        stop_time=stop_time.iso if stop_time else None,
        n_ephemerides=len(np.unique(df["designation"])),
    )
    return df


def interpolate_ephem(
    obj_df,
    target_jd: list[float] | float,
    window: u.Quantity = 0.5 * u.day,
    min_points_for_interp: int = 4,
    log: FilteringBoundLogger | None = None,
) -> SkyCoord:
    """Interpolate an asteroid ephemeris to arbitrary Julian dates using splines.

    Parameters
    ----------
    obj_df : pd.DataFrame
        Ephemeris table for a single object with columns ``julian_day``,
        ``ra_deg``, ``dec_deg``, and optionally ``distance_au``.
    target_jd : float or list of float
        Julian dates at which to interpolate.
    window : Quantity, optional
        Time window around each target JD included in the spline fit
        (default 0.5 day).
    min_points_for_interp : int, optional
        Minimum number of ephemeris points within ``window`` required for
        interpolation; targets with fewer points return NaN (default 4).
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    SkyCoord
        Interpolated positions; NaN coordinates are returned for targets
        where data are insufficient.
    """
    from scipy.interpolate import UnivariateSpline

    log = log or structlog.get_logger()

    target_jd = np.atleast_1d(target_jd)

    if min(obj_df["julian_day"]) > max(target_jd) or max(obj_df["julian_day"]) < min(
        target_jd
    ):
        log.error(
            "solar_system.interpolate_ephem.no_target_jd_in_range",
            target_jd_range=(min(target_jd), max(target_jd)),
            ephem_jd_range=(obj_df["julian_day"].min(), obj_df["julian_day"].max()),
            sso=obj_df["designation"].iloc[0],
        )
        return SkyCoord(ra=np.nan * u.deg, dec=np.nan * u.deg, frame="icrs")

    if (
        min(target_jd) < obj_df["julian_day"].min()
        or max(target_jd) > obj_df["julian_day"].max()
    ):
        log.warn(
            "interpolate_ephem.some_target_jd_out_of_range",
            target_jd_range=(min(target_jd), max(target_jd)),
            ephem_jd_range=(obj_df["julian_day"].min(), obj_df["julian_day"].max()),
        )

    w = window.to_value(u.day)
    subset = obj_df[
        (obj_df["julian_day"] >= target_jd.min() - w)
        & (obj_df["julian_day"] <= target_jd.max() + w)
    ]

    if len(subset) < min_points_for_interp:
        log.warn(
            "interpolate_ephem.too_few_interpolation_points",
            target_jd_range=(target_jd.min(), target_jd.max()),
            n_points=len(subset),
            required_points=min_points_for_interp,
            window=f"{w} days",
        )
        return SkyCoord(
            ra=np.full(len(target_jd), np.nan) * u.deg,
            dec=np.full(len(target_jd), np.nan) * u.deg,
            frame="icrs",
        )

    t = subset["julian_day"].values
    ra = subset["ra_deg"].values
    dec = subset["dec_deg"].values

    ra_spline = UnivariateSpline(t, ra, k=3, s=0)
    dec_spline = UnivariateSpline(t, dec, k=3, s=0)

    has_distance = "distance_au" in subset.columns
    dist_spline = (
        UnivariateSpline(t, subset["distance_au"].values, k=3, s=0)
        if has_distance
        else None
    )

    ras = ra_spline(target_jd)
    decs = dec_spline(target_jd)
    dists = dist_spline(target_jd) if has_distance else np.full(len(target_jd), np.nan)

    # Per-target check: mask times that don't have enough nearby ephem points.
    n_local = ((t >= target_jd[:, None] - w) & (t <= target_jd[:, None] + w)).sum(
        axis=1
    )
    sparse = n_local < min_points_for_interp
    if sparse.any():
        log.warn(
            "interpolate_ephem.too_few_interpolation_points",
            n_sparse_targets=int(sparse.sum()),
            required_points=min_points_for_interp,
            window=f"{w} days",
        )
        ras[sparse] = np.nan
        decs[sparse] = np.nan
        dists[sparse] = np.nan

    return SkyCoord(
        ra=ras * u.deg,
        dec=decs * u.deg,
        distance=dists * u.au,
        frame="icrs",
    )


def get_sso_ephems_at_time(
    ephem_df: pd.DataFrame,
    sample_times: list[datetime] | datetime,
    planets: list[str] = [
        "Mercury",
        "Venus",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
    ],
    observer: GeographicPosition | None = None,
    log: FilteringBoundLogger | None = None,
) -> dict[str, dict[str, SkyCoord | datetime]]:
    """Return ephemeris positions for all SSOs in ``ephem_df`` at ``sample_times``.

    Parameters
    ----------
    ephem_df : pd.DataFrame
        JPL ephemeris table (from ``load_jpl_ephem_database``).  Pass ``None``
        to skip asteroid interpolation and only compute planet positions.
    sample_times : datetime or list of datetime
        UTC datetimes at which to evaluate positions.
    planets : list of str, optional
        Planet names to include via Skyfield DE440s (default: all 7 major planets).
    observer : GeographicPosition, optional
        Observer location for planet parallax.  If ``None``, uses the default
        SO LAT site.
    log : FilteringBoundLogger, optional
        Structured logger.

    Returns
    -------
    dict
        Mapping from object designation to ``{"pos": SkyCoord, "time": ndarray}``.
    """
    log = log or structlog.get_logger()
    sample_times = np.atleast_1d(sample_times)
    time_jd = Time(sample_times, format="datetime").jd
    sso_ephems = {}

    if ephem_df is not None:
        unique_objs = ephem_df["designation"].unique()
        log.info(
            "solar_system.get_sso_ephems_at_time.starting_interpolation",
            n_objects=len(unique_objs),
            sample_times=sample_times,
        )
        for obj in unique_objs:
            obj_df = ephem_df[ephem_df["designation"] == obj]
            interp_pos = interpolate_ephem(obj_df, time_jd)
            if np.any(np.isnan(interp_pos.ra.value)) or np.any(
                np.isnan(interp_pos.dec.value)
            ):
                log.error(
                    "solar_system.get_sso_ephems_at_time.nan_in_pos",
                    object=obj,
                    pos=interp_pos,
                )
                continue
            sso_ephems[obj] = {"pos": interp_pos, "time": sample_times}

    if planets:
        ts = load.timescale()
        skyfield_time = ts.from_datetimes(sample_times)
        eph = load("de440s.bsp")
        earth = eph["earth"]
        observer = observer if observer else create_observer()
        observer_topo = earth + observer
        for p in planets:
            planet_eph = eph[f"{p} Barycenter"]
            ra, dec, distance = (
                observer_topo.at(skyfield_time).observe(planet_eph).radec()
            )
            sso_ephems[p] = {}
            sso_ephems[p]["pos"] = SkyCoord(
                ra=ra.degrees * u.deg,
                dec=dec.degrees * u.deg,
                distance=distance.km * u.km,
                frame="icrs",
            )
            sso_ephems[p]["time"] = np.array(
                [
                    datetime.fromtimestamp(
                        t.utc_datetime().timestamp(), tz=timezone.utc
                    )
                    for t in skyfield_time
                ]
            )

    log.info(
        "solar_system.get_sso_ephems_at_time.ephemerides_computed",
        n_objects=len(sso_ephems),
        time=sample_times,
    )

    return sso_ephems


def get_sso_ephem_in_map(
    input_map: ProcessableMap,
    ephem_df: pd.DataFrame,
    interp_time_range: u.Quantity = 0.5 * u.day,
    interp_to: u.Quantity | None = 10 * u.min,
    observer: GeographicPosition | None = None,
    planets: list[str] = [
        "Mercury",
        "Venus",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
    ],
    log: FilteringBoundLogger | None = None,
) -> dict[str, dict[str, SkyCoord | datetime]]:
    """
    Get solar system objects that fall within the provided map.

    Parameters
    ----------
    input_map : ProcessableMap
        Map to check for solar system objects.
    ephem_df : pd.DataFrame
        DataFrame containing ephemerides of solar system objects.
    interp_time_range : u.Quantity, optional
        Time range for interpolation, by default +- 0.5*u.day
    interp_to: u.Quantity, optional
        Interpolation sample rate for detecting if an asteroid is in the map.
    observer: GeographicPosition, optional
        Observer location for getting planet ephemerides. If None, ignores planets.
    log : FilteringBoundLogger, optional
        Logger for logging information, by default None.

    Returns
    -------
    sso_ephems : dict
        Dictionary mapping object designations to their ephemerides .
    """
    log = log or structlog.get_logger()
    log.info(
        "solar_system.get_sso_ephem_in_map.initialize",
        map_id=input_map.map_id,
        interp_time_range=interp_time_range,
        interp_to=interp_to,
    )
    time_range = [input_map.observation_start, input_map.observation_end]

    time_jd = Time(time_range).jd
    if interp_to is not None:
        time_jd = np.arange(
            np.min(time_jd), np.nanmax(time_jd), interp_to.to_value(u.day)
        )

    log.info(
        "solar_system.get_sso_ephem_in_map.starting_ephemeris_computation",
        julian_dates=time_jd,
        time_range=time_range,
        time_interpolated="True" if interp_to is not None else "False",
    )

    sso_ephems = {}
    unique_objs = ephem_df["designation"].unique()
    for obj in tqdm(unique_objs, desc="Checking solar system objects in map"):
        obj_df = ephem_df[ephem_df["designation"] == obj]
        interp_pos = interpolate_ephem(
            obj_df, time_jd, window=interp_time_range, log=log
        )

        result, _ = input_map.filter_sources(interp_pos)
        if not np.any(result):
            continue
        mean_pos = SkyCoord(
            ra=np.nanmean(interp_pos.ra[result]),
            dec=np.nanmean(interp_pos.dec[result]),
            frame="icrs",
        )

        map_time_at_mean_pos = input_map.time_mean.at(
            [mean_pos.dec.to_value("rad"), mean_pos.ra.to_value("rad")], mode="nn"
        )
        if (
            map_time_at_mean_pos > input_map.observation_end.timestamp()
            or map_time_at_mean_pos < input_map.observation_start.timestamp()
        ):
            log.error(
                "solar_system.get_sso_ephem_in_map.mean_pos_time_out_of_range",
                object=obj,
                mean_pos_time=map_time_at_mean_pos,
                observation_time_range=(
                    input_map.observation_start,
                    input_map.observation_end,
                ),
            )
            continue
        nearest_pos = interpolate_ephem(
            obj_df,
            Time(map_time_at_mean_pos, format="unix", scale="utc").jd,
            window=interp_time_range,
            log=log,
        )
        if np.isnan(nearest_pos.ra.value) or np.isnan(nearest_pos.dec.value):
            log.error(
                "solar_system.get_sso_ephem_in_map.nearest_pos_is_nan",
                object=obj,
                mean_pos_time=map_time_at_mean_pos,
                nearest_pos=nearest_pos,
            )
            continue
        sso_ephems[obj] = {
            "pos": nearest_pos,
            "time": datetime.fromtimestamp(
                float(map_time_at_mean_pos), tz=timezone.utc
            ),
        }

    if planets and observer is not None:
        if input_map.observation_length > timedelta(days=1):
            log.warn(
                "solar_system.get_sso_ephem_in_map.planet_ephem",
                observation_length=input_map.observation_length.to_value(u.day),
                warning="Observation length is long, so planet ephemerides at the average time may be inaccurate.",
            )
        time_delta = (input_map.observation_end - input_map.observation_start) / 2
        mean_time = input_map.observation_start + time_delta
        planet_ephems = get_sso_ephems_at_time(
            ephem_df=None,
            sample_times=mean_time,
            planets=planets,
            observer=observer,
            log=log,
        )
        for p in planets:
            if p not in planet_ephems:
                continue
            pos = planet_ephems[p]["pos"]
            result, _ = input_map.filter_sources(pos)
            if not np.any(result):
                continue
            sso_ephems[p] = {
                "pos": pos,
                "time": planet_ephems[p]["time"],
            }
    return sso_ephems
