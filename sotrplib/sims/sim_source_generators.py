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


class SimulatedSourceGenerator(ABC):
    def generate(
        self,
        input_map: ProcessableMap | None = None,
        box: tuple[SkyCoord] | None = None,
        time_range: tuple[datetime | None, datetime | None] | None = None,
    ) -> tuple[list[SimulatedSource], SourceCatalog]:
        """
        Generate the simulated sources, optionally within a map or a box on the sky.
        If input_map is provided, box is to be ignored and inject only within the map area.
        If time_range is provided, it specifies the time range for the simulation.
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

    def generate(
        self,
        input_map: ProcessableMap | None = None,
        box: tuple[SkyCoord] | None = None,
        time_range: tuple[datetime | None, datetime | None] | None = None,
    ):
        if box is None and input_map is None:
            box = [
                SkyCoord(ra=0.0 * u.deg, dec=-89.99 * u.deg),
                SkyCoord(ra=359.99 * u.deg, dec=89.99 * u.deg),
            ]

        if input_map is not None:
            positions = generate_random_positions_in_map(self.number, input_map.flux)
        else:
            positions = generate_random_positions(
                self.number,
                ra_lims=(box[0].ra, box[1].ra),
                dec_lims=(box[0].dec, box[1].dec),
            )

        log = self.log.bind(box=box)

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
        flare_earliest_time: datetime | None,
        flare_latest_time: datetime | None,
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
        box: tuple[SkyCoord] | None = None,
        time_range: tuple[datetime | None, datetime | None] | None = None,
    ):
        if box is None and input_map is None:
            box = [
                SkyCoord(ra=0.0 * u.deg, dec=-89.99 * u.deg),
                SkyCoord(ra=359.99 * u.deg, dec=89.99 * u.deg),
            ]

        if input_map is not None:
            if self.flare_earliest_time is None or self.flare_latest_time is None:
                self.flare_earliest_time = input_map.observation_start
                self.flare_latest_time = input_map.observation_end
            else:
                self.flare_earliest_time = self.flare_earliest_time
                self.flare_latest_time = self.flare_latest_time
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
        log = self.log.bind(
            n=self.number,
            box=box,
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
            positions = generate_random_positions(
                self.number,
                ra_lims=(box[0].ra, box[1].ra),
                dec_lims=(box[0].dec, box[1].dec),
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
                        angular_separation=0.0 * u.deg,
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
