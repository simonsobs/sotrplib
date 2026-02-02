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
    PowerLawTransientSimulatedSource,
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
    @abstractmethod
    def generate(
        self,
        input_map: ProcessableMap | None = None,
        box: tuple[SkyCoord] | None = None,
        time_range: tuple[AwareDatetime | None, AwareDatetime | None] | None = None,
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
        time_range: tuple[AwareDatetime | None, AwareDatetime | None] | None = None,
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
                crossmatches=[
                    CrossMatch(
                        source_id=f"sim-{base + i:07d}",
                        source_type="simulated_fixed",
                        catalog_name="simulated",
                        ra=positions[i][1],
                        dec=positions[i][0],
                        angular_separation=0.0 * u.deg,
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
        box: tuple[SkyCoord] | None = None,
        time_range: tuple[AwareDatetime | None, AwareDatetime | None] | None = None,
    ):
        if box is None and input_map is None:
            box = [
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
                        source_type="simulated_gaussian",
                        ra=positions[i][1],
                        dec=positions[i][0],
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


class PowerLawTransientSourceGenerator(SimulatedSourceGenerator):
    def __init__(
        self,
        flare_earliest_time: datetime,
        flare_latest_time: datetime,
        alpha_rise_min: float,
        alpha_rise_max: float,
        alpha_decay_min: float,
        alpha_decay_max: float,
        peak_amplitude_minimum: u.Quantity,
        peak_amplitude_maximum: u.Quantity,
        number: int,
        smoothness_min: float = 1.0,
        smoothness_max: float = 1.0,
        t_ref_min: timedelta = timedelta(days=1),
        t_ref_max: timedelta = timedelta(days=1),
        catalog_fraction: float = 1.0,
        log: FilteringBoundLogger | None = None,
    ):
        """
        Generate power-law transient sources (e.g., GRB afterglows).
        Parameters:
            - flare_earliest_time: Earliest possible peak time
            - flare_latest_time: Latest possible peak time
            - alpha_rise_min/max: Range for rising power-law index (>0)
            - alpha_decay_min/max: Range for decay power-law index (<0)
            - peak_amplitude_minimum/maximum: Range for peak flux
            - number: Number of sources to generate
            - smoothness_min/max: Range for smoothness parameter (default 1.0)
            - t_ref_min/max: Range for reference timescale (default 1 day)
            - catalog_fraction: Fraction of sources to include in catalog
            - log: Logger instance
        """
        self.flare_earliest_time = flare_earliest_time
        self.flare_latest_time = flare_latest_time
        self.alpha_rise_min = alpha_rise_min
        self.alpha_rise_max = alpha_rise_max
        self.alpha_decay_min = alpha_decay_min
        self.alpha_decay_max = alpha_decay_max
        self.peak_amplitude_minimum = peak_amplitude_minimum.to(u.Jy)
        self.peak_amplitude_maximum = peak_amplitude_maximum.to(u.Jy)
        self.number = number
        self.smoothness_min = smoothness_min
        self.smoothness_max = smoothness_max
        self.t_ref_min = t_ref_min
        self.t_ref_max = t_ref_max
        self.catalog_fraction = catalog_fraction
        self.log = log or get_logger()

    def generate(
        self,
        box: tuple[SkyCoord] | None = None,
        time_range: tuple[AwareDatetime | None, AwareDatetime | None] | None = None,
    ):
        if box is None:
            box = [
                SkyCoord(ra=0.0 * u.deg, dec=-89.99 * u.deg),
                SkyCoord(ra=359.99 * u.deg, dec=89.99 * u.deg),
            ]

        time_range = (
            (self.flare_earliest_time, self.flare_latest_time)
            if time_range is None
            else time_range
        )
        log = self.log.bind(box=box, time_range=time_range, n=self.number)
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

        # Generate source catalog entries
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

        # Create catalog with only the specified fraction
        catalog = RegisteredSourceCatalog(
            sources=sources[: int(self.number * self.catalog_fraction)]
        )

        # Generate simulated power-law transient sources
        simulated_sources = [
            PowerLawTransientSimulatedSource(
                position=SkyCoord(ra=x.ra, dec=x.dec),
                peak_time=random_datetime(
                    self.flare_earliest_time, self.flare_latest_time
                ),
                peak_amplitude=x.flux,
                alpha_rise=random.uniform(self.alpha_rise_min, self.alpha_rise_max),
                alpha_decay=random.uniform(self.alpha_decay_min, self.alpha_decay_max),
                smoothness=random.uniform(self.smoothness_min, self.smoothness_max),
                t_ref=random_timedelta(self.t_ref_min, self.t_ref_max),
            )
            for x in sources
        ]

        log = log.bind(
            n_simulated_sources=len(simulated_sources),
            earliest_flare=min(simulated_sources, key=lambda x: x.peak_time).peak_time,
            latest_flare=max(simulated_sources, key=lambda x: x.peak_time).peak_time,
        )
        log.info("source_simulation.powerlaw.complete")

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
        box: tuple[SkyCoord] | None = None,
        time_range: tuple[AwareDatetime | None, AwareDatetime | None] | None = None,
    ):
        self.log = self.log.bind(
            func="sim_source_generator.SOCatSourceGenerator.generate"
        )
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
        number_of_flares = int(number_of_sources * self.fraction_gaussian)
        self.log.info(
            "sim_source_generator.SOCatSourceGenerator.generate.loaded_catalog",
            n_sources=number_of_sources,
            n_fixed=number_of_fixed,
            n_flares=number_of_flares,
            box=box,
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
