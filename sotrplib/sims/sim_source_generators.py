"""
Generators for simulated sources. sim_sources.py contains the definitions
for the different types of sources, and these are the classes that implement
the random generation of those sources.
"""

import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from astropy import units as u
from astropy.coordinates import SkyCoord
from pydantic import AwareDatetime
from socat.client.settings import SOCatClientSettings
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sims.sim_utils import (
    generate_random_positions,
    generate_random_positions_in_map,
)
from sotrplib.source_catalog.core import (
    CrossMatch,
    RegisteredSource,
    RegisteredSourceCatalog,
    SourceCatalog,
)

from .sim_sources import (
    FixedSimulatedSource,
    GaussianTransientSimulatedSource,
    SimulatedSource,
)


def random_datetime(start, end):
    """Generate a random datetime between `start` and `end`"""
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)


def random_timedelta(min_td, max_td):
    """Generate a random timedelta between `min_td` and `max_td`"""
    delta = max_td - min_td
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return min_td + timedelta(seconds=random_second)


class SimulatedSourceGenerator(ABC):
    """Abstract base for simulated source generators."""

    @abstractmethod
    def generate(
        self,
        input_map: ProcessableMap | None = None,
        sky_box: tuple[SkyCoord, SkyCoord] | None = None,
        time_range: tuple[AwareDatetime | None, AwareDatetime | None] | None = None,
    ) -> tuple[list[SimulatedSource], SourceCatalog]:
        """Generate simulated sources within a map or sky region.

        Parameters
        ----------
        input_map : ProcessableMap, optional
            If provided, sources are generated within the map footprint and
            ``sky_box`` is ignored.
        sky_box : tuple of SkyCoord, optional
            ``(lower_left, upper_right)`` sky-coordinate bounding box.
        time_range : tuple of AwareDatetime, optional
            ``(start, end)`` time range for the simulation.

        Returns
        -------
        sources : list of SimulatedSource
            Generated source objects for use in injection.
        catalog : SourceCatalog
            Reference catalog containing at least a fraction of the sources
            (some may be hidden for blind-search testing).
        """
        return


class FixedSourceGenerator(SimulatedSourceGenerator):
    """Generate a fixed number of static (constant-flux) simulated sources.

    Parameters
    ----------
    min_flux : Quantity
        Minimum source flux.
    max_flux : Quantity
        Maximum source flux.
    number : int
        Number of sources to generate.
    catalog_fraction : float, optional
        Fraction of sources placed in the returned reference catalog
        (default 1.0 — all sources visible).
    log : FilteringBoundLogger, optional
        Structured logger.
    """

    def __init__(
        self,
        min_flux: u.Quantity,
        max_flux: u.Quantity,
        number: int,
        catalog_fraction: float = 1.0,
        log: FilteringBoundLogger | None = None,
    ):
        self.min_flux = min_flux.to(u.Jy)
        self.max_flux = max_flux.to(u.Jy)
        self.number = number
        self.catalog_fraction = catalog_fraction
        self.log = log or get_logger()

    def generate(
        self,
        input_map: ProcessableMap | None = None,
        sky_box: tuple[SkyCoord, SkyCoord] | None = None,
        time_range: tuple[AwareDatetime | None, AwareDatetime | None] | None = None,
    ):
        if sky_box is None and input_map is None:
            sky_box = [
                SkyCoord(ra=0.0 * u.deg, dec=-89.99 * u.deg),
                SkyCoord(ra=359.99 * u.deg, dec=89.99 * u.deg),
            ]

        if input_map is not None:
            positions = generate_random_positions_in_map(self.number, input_map.flux)
        else:
            # sky_box[0] = (ra_min, dec_min), sky_box[1] = (ra_max, dec_max).
            # When the box wraps RA=0, sky_box[0].ra > sky_box[1].ra (e.g. 357° vs 3°);
            # generate_random_positions handles that by converting to negative RA.
            positions = generate_random_positions(
                self.number,
                ra_lims=(sky_box[0].ra, sky_box[1].ra),
                dec_lims=(sky_box[0].dec, sky_box[1].dec),
            )

        log = self.log.bind(sky_box=sky_box)

        # Potential issue with this setup is that you've got overlapping sources..?
        # But that's something we should probably handle anyway...
        # Another issue is that we're not actually generating sources on the
        # sphere, but that is not critical.
        base = random.randint(0, 100000)

        log = log.bind(base_id=base)

        sources = [
            RegisteredSource(
                ra=positions[i][1],
                dec=positions[i][0],
                frequency=90.0 * u.GHz,
                flux=random.uniform(
                    self.min_flux.to_value("Jy"), self.max_flux.to_value("Jy")
                )
                * u.Jy,
                source_id=f"sim-{base + i:07d}",
                source_type="simulated",
                err_ra=0.0 * u.deg,
                err_dec=0.0 * u.deg,
                err_flux=0.0 * u.Jy,
                crossmatches=[
                    CrossMatch(
                        source_id=f"sim-{base + i:07d}",
                        source_type="simulated_fixed",
                        catalog_name="simulated",
                        ra=positions[i][1],
                        dec=positions[i][0],
                        angular_separation=0.0 * u.deg,
                        catalog_idx=base + i,
                    )
                ],
            )
            for i in range(self.number)
        ]

        log = log.bind(n_sources=len(sources))

        catalog = RegisteredSourceCatalog(
            sources=sources[: int(self.number * self.catalog_fraction)]
        )
        simulated_sources = [
            FixedSimulatedSource(position=SkyCoord(ra=x.ra, dec=x.dec), flux=x.flux)
            for x in sources
        ]

        log.info("source_generation.fixed.complete")

        return simulated_sources, catalog


class GaussianTransientSourceGenerator(SimulatedSourceGenerator):
    """Generate Gaussian-flare transient sources at random sky positions.

    Parameters
    ----------
    flare_earliest_time : AwareDatetime or None
        Earliest possible flare peak time.  Overridden by the map observation
        start if ``None`` and an ``input_map`` is supplied.
    flare_latest_time : AwareDatetime or None
        Latest possible flare peak time.  Overridden by the map observation
        end if ``None`` and an ``input_map`` is supplied.
    flare_width_shortest : timedelta
        Minimum flare FWHM.
    flare_width_longest : timedelta
        Maximum flare FWHM.
    peak_amplitude_minimum : Quantity
        Minimum peak flux.
    peak_amplitude_maximum : Quantity
        Maximum peak flux.
    number : int
        Number of sources to generate.
    catalog_fraction : float, optional
        Fraction of sources placed in the returned reference catalog
        (default 1.0).
    log : FilteringBoundLogger, optional
        Structured logger.
    """

    def __init__(
        self,
        flare_earliest_time: AwareDatetime | None,
        flare_latest_time: AwareDatetime | None,
        flare_width_shortest: timedelta,
        flare_width_longest: timedelta,
        peak_amplitude_minimum: u.Quantity,
        peak_amplitude_maximum: u.Quantity,
        number: int,
        catalog_fraction: float = 1.0,
        log: FilteringBoundLogger | None = None,
    ):
        self.flare_earliest_time = flare_earliest_time
        self.flare_latest_time = flare_latest_time
        self.flare_width_shortest = flare_width_shortest
        self.flare_width_longest = flare_width_longest
        self.peak_amplitude_minimum = peak_amplitude_minimum.to(u.Jy)
        self.peak_amplitude_maximum = peak_amplitude_maximum.to(u.Jy)
        self.number = number
        self.catalog_fraction = catalog_fraction
        self.log = log or get_logger()

    def generate(
        self,
        input_map: ProcessableMap | None = None,
        sky_box: tuple[SkyCoord, SkyCoord] | None = None,
        time_range: tuple[AwareDatetime | None, AwareDatetime | None] | None = None,
    ):
        if sky_box is None and input_map is None:
            sky_box = [
                SkyCoord(ra=0.0 * u.deg, dec=-89.99 * u.deg),
                SkyCoord(ra=359.99 * u.deg, dec=89.99 * u.deg),
            ]

        if input_map is not None:
            self.flare_earliest_time = (
                input_map.observation_start
                if self.flare_earliest_time is None
                else self.flare_earliest_time
            )
            self.flare_latest_time = (
                input_map.observation_end
                if self.flare_latest_time is None
                else self.flare_latest_time
            )

        else:
            self.flare_earliest_time = (
                time_range[0] if time_range is not None else self.flare_earliest_time
            )
            self.flare_latest_time = (
                time_range[1] if time_range is not None else self.flare_latest_time
            )

        if self.flare_earliest_time is None or self.flare_latest_time is None:
            log = self.log.error(
                "GaussianTransientSourceGenerator.generate.timerange_invalid",
                flare_earliest_time=self.flare_earliest_time,
                flare_latest_time=self.flare_latest_time,
            )
            raise ValueError(
                f"Invalid time range: flare_earliest_time={self.flare_earliest_time}, flare_latest_time={self.flare_latest_time}"
            )

        log = self.log.bind(
            n=self.number,
            sky_box=sky_box,
            time_range=[self.flare_earliest_time, self.flare_latest_time],
        )

        base = random.randint(0, 100000)

        log = log.bind(base_id=base)

        def random_datetime(start, end):
            """Generate a random datetime between `start` and `end`"""
            delta = end - start
            int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
            random_second = random.randrange(int_delta)
            return start + timedelta(seconds=random_second)

        def random_timedelta(min_td, max_td):
            """Generate a random timedelta between `min_td` and `max_td`"""
            delta = max_td - min_td
            int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
            random_second = random.randrange(int_delta)
            return min_td + timedelta(seconds=random_second)

        if input_map is not None:
            positions = generate_random_positions_in_map(self.number, input_map.flux)
        else:
            # sky_box[0] = (ra_min, dec_min), sky_box[1] = (ra_max, dec_max).
            # When the box wraps RA=0, sky_box[0].ra > sky_box[1].ra (e.g. 357° vs 3°);
            # generate_random_positions handles that by converting to negative RA.
            positions = generate_random_positions(
                self.number,
                ra_lims=(sky_box[0].ra, sky_box[1].ra),
                dec_lims=(sky_box[0].dec, sky_box[1].dec),
            )

        sources = [
            RegisteredSource(
                ra=positions[i][1],
                dec=positions[i][0],
                frequency=90.0 * u.GHz,
                flux=random.uniform(
                    self.peak_amplitude_minimum.to_value("Jy"),
                    self.peak_amplitude_maximum.to_value("Jy"),
                )
                * u.Jy,
                source_id=f"sim-{base + i:07d}",
                source_type="simulated",
                err_ra=0.0 * u.deg,
                err_dec=0.0 * u.deg,
                err_flux=0.0 * u.Jy,
                crossmatches=[
                    CrossMatch(
                        source_id=f"sim-{base + i:07d}",
                        catalog_name="simulated",
                        source_type="simulated_gaussian",
                        ra=positions[i][1],
                        dec=positions[i][0],
                        angular_separation=0.0 * u.deg,
                        catalog_idx=base + i,
                    )
                ],
            )
            for i in range(len(positions))
        ]

        catalog = RegisteredSourceCatalog(
            sources=sources[: int(len(sources) * self.catalog_fraction)]
        )
        simulated_sources = [
            GaussianTransientSimulatedSource(
                position=SkyCoord(ra=x.ra, dec=x.dec),
                peak_time=random_datetime(
                    self.flare_earliest_time, self.flare_latest_time
                ),
                flare_width=random_timedelta(
                    self.flare_width_shortest, self.flare_width_longest
                ),
                peak_amplitude=x.flux,
            )
            for x in sources
        ]

        log = log.bind(
            n_simulated_sources=len(simulated_sources),
            injected_in="map" if input_map is not None else "box",
            earliest_flare=min(simulated_sources, key=lambda x: x.peak_time).peak_time,
            latest_flare=max(simulated_sources, key=lambda x: x.peak_time).peak_time,
        )

        log.info("source_simulation.gaussian.complete")

        return simulated_sources, catalog


class SOCatSourceGenerator(SimulatedSourceGenerator):
    """Generate simulated sources drawn from an existing SOCat catalog.

    Sources are injected as a mixture of constant-flux and Gaussian-transient
    types, matched against the same catalog to allow recovery testing.

    Parameters
    ----------
    fraction_fixed : float
        Fraction of catalog sources injected as constant-flux sources.
    fraction_gaussian : float
        Fraction of catalog sources injected as Gaussian-transient sources.
        ``fraction_fixed + fraction_gaussian`` must be ≤ 1.0.
    flare_earliest_time : datetime
        Earliest possible flare peak time.
    flare_latest_time : datetime
        Latest possible flare peak time.
    flare_width_shortest : timedelta
        Minimum flare FWHM.
    flare_width_longest : timedelta
        Maximum flare FWHM.
    peak_amplitude_minimum_factor : float
        Lower bound on the random flux scaling factor applied to catalog fluxes.
    peak_amplitude_maximum_factor : float
        Upper bound on the random flux scaling factor applied to catalog fluxes.
    log : FilteringBoundLogger, optional
        Structured logger.
    """

    def __init__(
        self,
        fraction_fixed: float,
        fraction_gaussian: float,
        flare_earliest_time: datetime,
        flare_latest_time: datetime,
        flare_width_shortest: timedelta,
        flare_width_longest: timedelta,
        peak_amplitude_minimum_factor: float,
        peak_amplitude_maximum_factor: float,
        log: FilteringBoundLogger | None = None,
    ):
        if (fraction_fixed + fraction_gaussian) > 1.0:
            raise ValueError(
                "Incompatible fractions for fixed and gaussian sources in SOCat generation"
            )

        self.fraction_fixed = fraction_fixed
        self.fraction_gaussian = fraction_gaussian
        self.flare_earliest_time = flare_earliest_time
        self.flare_latest_time = flare_latest_time
        self.flare_width_shortest = flare_width_shortest
        self.flare_width_longest = flare_width_longest
        self.peak_amplitude_minimum_factor = peak_amplitude_minimum_factor
        self.peak_amplitude_maximum_factor = peak_amplitude_maximum_factor
        self.log = log or get_logger()

    def generate(
        self,
        input_map: ProcessableMap | None = None,
        sky_box: tuple[SkyCoord, SkyCoord] | None = None,
        time_range: tuple[AwareDatetime | None, AwareDatetime | None] | None = None,
    ):
        self.log = self.log.bind(
            func="sim_source_generator.SOCatSourceGenerator.generate"
        )
        socat_client_settings = SOCatClientSettings()
        socat_client = socat_client_settings.client

        if sky_box is None:
            sky_box = [
                SkyCoord(ra=0.0 * u.deg, dec=-89.99 * u.deg),
                SkyCoord(ra=359.99 * u.deg, dec=89.99 * u.deg),
            ]

        sources = socat_client.get_box_fixed(
            lower_left=sky_box[0], upper_right=sky_box[1]
        )
        random.shuffle(sources)

        number_of_sources = len(sources)

        number_of_fixed = int(number_of_sources * self.fraction_fixed)
        number_of_flares = int(number_of_sources * self.fraction_gaussian)
        self.log.info(
            "sim_source_generator.SOCatSourceGenerator.generate.loaded_catalog",
            n_sources=number_of_sources,
            n_fixed=number_of_fixed,
            n_flares=number_of_flares,
            sky_box=sky_box,
        )
        all_sources = [
            RegisteredSource(
                ra=source.position.ra,
                dec=source.position.dec,
                frequency=90.0 * u.GHz,
                flux=source.flux if source.flux is not None else u.Quantity(0.0, "Jy"),
                source_id=str(source.source_id),
                source_type="simulated",
                err_ra=0.0 * u.deg,
                err_dec=0.0 * u.deg,
                err_flux=0.0 * u.Jy,
                crossmatches=[
                    CrossMatch(
                        ra=source.position.ra,
                        dec=source.position.dec,
                        source_id=str(source.source_id),
                        catalog_name="socat",
                        alternate_names=[source.name] if source.name else [],
                    )
                ],
            )
            for source in sources[: number_of_fixed + number_of_flares]
        ]

        fixed_sources = [
            FixedSimulatedSource(
                position=SkyCoord(ra=source.ra, dec=source.dec), flux=source.flux
            )
            for source in all_sources[:number_of_fixed]
        ]

        flare_sources = [
            GaussianTransientSimulatedSource(
                position=SkyCoord(ra=source.position.ra, dec=source.position.dec),
                peak_time=random_datetime(
                    self.flare_earliest_time, self.flare_latest_time
                ),
                flare_width=random_timedelta(
                    self.flare_width_shortest, self.flare_width_longest
                ),
                peak_amplitude=source.flux
                * random.uniform(
                    self.peak_amplitude_minimum_factor,
                    self.peak_amplitude_maximum_factor,
                ),
            )
            for source in sources[number_of_fixed : number_of_fixed + number_of_flares]
        ]

        catalog = RegisteredSourceCatalog(sources=all_sources)

        return fixed_sources + flare_sources, catalog
