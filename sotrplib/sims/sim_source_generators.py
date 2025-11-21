"""
Generators for simulated sources. sim_sources.py contains the definitions
for the different types of sources, and these are the classes that implement
the random generation of those sources.
"""

import random
from abc import ABC
from datetime import datetime, timedelta

from astropy import units as u
from astropy.coordinates import SkyCoord
from socat.client.settings import SOCatClientSettings
from structlog import get_logger
from structlog.types import FilteringBoundLogger

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
    def generate(
        self, box: tuple[SkyCoord] | None = None
    ) -> tuple[list[SimulatedSource], SourceCatalog]:
        """
        Generate the simulated sources, optionally within a box on the sky.
        Returns both the list of the core 'sources' for use in the source
        simulator, as well as a source catlaog that contains at least some of
        the sources (some may be hidden for testing blind searches)
        """
        return


class FixedSourceGenerator(SimulatedSourceGenerator):
    """
    In-place source generator using photutils.
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

    def generate(self, box: tuple[SkyCoord] | None = None):
        if box is None:
            box = [
                SkyCoord(ra=0.01 * u.deg, dec=-89.99 * u.deg),
                SkyCoord(ra=359.99 * u.deg, dec=89.99 * u.deg),
            ]

        log = self.log.bind(box=box)

        # Potential issue with this setup is that you've got overlapping sources..?
        # But that's something we should probably handle anyway...
        # Another issue is that we're not actually generating sources on the
        # sphere, but that is not critical.
        base = random.randint(0, 100000)

        log = log.bind(base_id=base)

        sources = [
            RegisteredSource(
                ra=random.uniform(box[0].ra.to_value("deg"), box[1].ra.to_value("deg"))
                * u.deg,
                dec=random.uniform(
                    box[0].dec.to_value("deg"), box[1].dec.to_value("deg")
                )
                * u.deg,
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
    def __init__(
        self,
        flare_earliest_time: datetime,
        flare_latest_time: datetime,
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

    def generate(self, box: tuple[SkyCoord] | None = None):
        if box is None:
            box = [
                SkyCoord(ra=0.0 * u.deg, dec=-89.99 * u.deg),
                SkyCoord(ra=359.99 * u.deg, dec=89.99 * u.deg),
            ]

        log = self.log.bind(box=box)

        base = random.randint(0, 100000)

        log = log.bind(base_id=base)

        sources = [
            RegisteredSource(
                ra=random.uniform(box[0].ra.to_value("deg"), box[1].ra.to_value("deg"))
                * u.deg,
                dec=random.uniform(
                    box[0].dec.to_value("deg"), box[1].dec.to_value("deg")
                )
                * u.deg,
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
                        angular_separation=0.0 * u.deg,
                    )
                ],
            )
            for i in range(self.number)
        ]

        catalog = RegisteredSourceCatalog(
            sources=sources[: int(self.number * self.catalog_fraction)]
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
            earliest_flare=min(simulated_sources, key=lambda x: x.peak_time).peak_time,
            latest_flare=max(simulated_sources, key=lambda x: x.peak_time).peak_time,
        )

        log.info("source_simulation.gaussian.complete")

        return simulated_sources, catalog


class SOCatSourceGenerator(SimulatedSourceGenerator):
    """
    A source generator that uses an existing source catalog in SOCat format.
    We then generate simulated sources based on that catalog, so it can be
    matched against the same catalog.
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
        """
        Parameters
        ----------
        fraction_fixed: float
            Fraction of sources in the catalog to inject as 'fixed' flux sources.
        fraction_gaussian: float
            Fraction of sources to inject as gaussian flares.

        Notes
        -----
        The fraction of fixed and gaussian sources together must be less than 1.0.
        For information on the flare parameters, see the `GaussianTransientSourceGenerator` class.
        """

        if (fraction_fixed + fraction_gaussian) > 1.0:
            raise ValueError(
                "Incompatible fractions for fixed and gaussian sources in SOCat generation"
            )

        self.fraction_fixed = fraction_fixed
        self.fraction_guassian = fraction_gaussian
        self.flare_earliest_time = flare_earliest_time
        self.flare_latest_time = flare_latest_time
        self.flare_width_shortest = flare_width_shortest
        self.flare_width_longest = flare_width_longest
        self.peak_amplitude_minimum_factor = peak_amplitude_minimum_factor
        self.peak_amplitude_maximum_factor = peak_amplitude_maximum_factor
        self.log = log or get_logger()

    def generate(self, box: tuple[SkyCoord] | None = None):
        socat_client_settings = SOCatClientSettings()
        socat_client = socat_client_settings.client

        if box is None:
            box = [
                SkyCoord(ra=0.0 * u.deg, dec=-89.99 * u.deg),
                SkyCoord(ra=359.99 * u.deg, dec=89.99 * u.deg),
            ]

        sources = socat_client.get_box(lower_left=box[0], upper_right=box[1])
        random.shuffle(sources)

        number_of_sources = len(sources)

        number_of_fixed = int(number_of_sources * self.fraction_fixed)
        number_of_flares = int(number_of_sources * self.fraction_guassian)

        all_sources = [
            RegisteredSource(
                ra=source.position.ra,
                dec=source.position.dec,
                frequency=90.0 * u.GHz,
                flux=source.flux if source.flux is not None else u.Quantity(0.0, "Jy"),
                source_id=str(source.id),
                source_type="simulated",
                err_ra=0.0 * u.deg,
                err_dec=0.0 * u.deg,
                err_flux=0.0 * u.Jy,
                crossmatches=[
                    CrossMatch(
                        ra=source.position.ra,
                        dec=source.position.dec,
                        source_id=str(source.id),
                        catalog_name="socat",
                        alternate_names=[source.name] if source.name else [],
                    )
                ],
            )
            for source in sources[: number_of_fixed + number_of_flares]
        ]

        fixed_sources = all_sources[:number_of_fixed]

        flare_sources = [
            GaussianTransientSimulatedSource(
                position=source.position,
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
