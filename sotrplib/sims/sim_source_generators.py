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
        catalog_fraction: 1.0,
    ):
        self.min_flux = min_flux.to(u.Jy)
        self.max_flux = max_flux.to(u.Jy)
        self.number = number
        self.catalog_fraction = catalog_fraction

    def generate(self, box: tuple[SkyCoord] | None = None):
        if box is None:
            box = [
                SkyCoord(ra=0.0 * u.deg, dec=-89.99 * u.deg),
                SkyCoord(ra=359.99 * u.deg, dec=89.99 * u.deg),
            ]

        # Potential issue with this setup is that you've got overlapping sources..?
        # But that's something we should probably handle anyway...
        # Another issue is that we're not actually generating sources on the
        # sphere, but that is not critical.
        base = random.randint(0, 100000)

        sources = [
            RegisteredSource(
                ra=random.random(box[0].ra.to_value("deg"), box[1].ra.to_value("deg"))
                * u.deg,
                dec=random.random(
                    box[0].dec.to_value("deg"), box[1].dec.to_value("deg")
                )
                * u.deg,
                frequency="f090",
                flux=random.random(
                    self.min_flux.to_value("Jy"), self.max_flux.to_value("Jy")
                )
                * u.Jy,
                source_id=f"sim-{base + i:07d}",
                source_type="simulated",
                err_ra=0.0 * u.deg,
                err_dec=0.0 * u.deg,
                err_flux=0.0 * u.deg,
            )
            for i in range(self.number)
        ]

        catalog = RegisteredSourceCatalog(
            sources=sources[: int(self.number * self.catalog_fraction)]
        )
        simulated_sources = [
            FixedSimulatedSource(position=SkyCoord(ra=x.ra, dec=x.dec), flux=x.flux)
            for x in sources
        ]

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
        catalog_fraction: 1.0,
    ):
        self.flare_earliest_time = flare_earliest_time
        self.flare_latest_time = flare_latest_time
        self.flare_width_shortest = flare_width_shortest
        self.flare_width_longest = flare_width_longest
        self.peak_amplitude_minimum = peak_amplitude_minimum.to(u.Jy)
        self.peak_amplitude_maximum = peak_amplitude_maximum.to(u.Jy)
        self.number = number
        self.catalog_fraction = catalog_fraction

    def generate(self, box: tuple[SkyCoord] | None = None):
        if box is None:
            box = [
                SkyCoord(ra=0.0 * u.deg, dec=-89.99 * u.deg),
                SkyCoord(ra=359.99 * u.deg, dec=89.99 * u.deg),
            ]

        base = random.randint(0, 100000)

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

        sources = [
            RegisteredSource(
                ra=random.random(box[0].ra.to_value("deg"), box[1].ra.to_value("deg"))
                * u.deg,
                dec=random.random(
                    box[0].dec.to_value("deg"), box[1].dec.to_value("deg")
                )
                * u.deg,
                frequency="f090",
                flux=random.random(
                    self.peak_amplitude_minimum.to_value("Jy"),
                    self.peak_amplitude_maximum.to_value("Jy"),
                )
                * u.Jy,
                source_id=f"sim-{base + i:07d}",
                source_type="simulated",
                err_ra=0.0 * u.deg,
                err_dec=0.0 * u.deg,
                err_flux=0.0 * u.deg,
                crossmatches=[
                    CrossMatch(
                        source_id=f"sim-{base + i:07d}",
                        catalog="simulated",
                        separation=0.0 * u.deg,
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

        return simulated_sources, catalog
