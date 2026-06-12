from pathlib import Path

import astropy.units as u
import numpy as np
import structlog
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from pydantic import AwareDatetime, BaseModel
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.utils import skycoord_box_to_enmap_box
from sotrplib.sims import sim_maps


class SimulationParameters(BaseModel):
    """
    Sky simulation parameters (geometry and noise level).
    Defaults are given to be the center of the Deep56 ACT field.
    """

    center_ra: AstroPydanticQuantity[u.deg] = u.Quantity(16.3, "deg")
    center_dec: AstroPydanticQuantity[u.deg] = u.Quantity(-1.8, "deg")
    width_ra: AstroPydanticQuantity[u.deg] = u.Quantity(2.0, "deg")
    width_dec: AstroPydanticQuantity[u.deg] = u.Quantity(1.0, "deg")
    resolution: AstroPydanticQuantity[u.arcmin] = u.Quantity(0.5, "arcmin")
    map_noise: AstroPydanticQuantity[u.Jy] = u.Quantity(0.01, "Jy")


## TODO: if box is supplied, shoudl we override the SimulationParamters?
class SimulatedMap(ProcessableMap):
    def __init__(
        self,
        observation_start: AwareDatetime,
        observation_end: AwareDatetime,
        frequency: str | None = None,
        array: str | None = None,
        instrument: str | None = None,
        simulation_parameters: SimulationParameters | None = None,
        sky_box: tuple[SkyCoord, SkyCoord] | None = None,
        include_half_pixel_offset: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        """
        Parameters
        ----------
        observation_start: datetime
            Start time of the simulated observing session.
        observation_end: datetime
            End time of the simulated observing session.
        frequency: str | None, optional
            The frequency band of the simulation, e.g. f090. Defaults to f090.
        array: str | None, optional
            The array that this map represents.
        simulation_parameters: SimulationParameters, optional
            Parameters for the simulation. If None, defaults will be used.
        box: tuple[SkyCoord, SkyCoord], optional
            Optional sky box to simulate in. By default set via simulation_parameters.
        include_half_pixel_offset: bool, False
            Include the half-pixel offset in pixell constructors.
        log: FilteringBoundLogger, optional
            Logger to use. If None, a new one will be created.
        """
        self.observation_start = observation_start
        self.observation_end = observation_end
        self.sky_box = sky_box
        self.include_half_pixel_offset = include_half_pixel_offset
        self.frequency = frequency or "f090"
        self.array = array or "pa5"
        self.instrument = instrument

        self.observation_length = observation_end - observation_start
        self.observation_time = observation_start + 0.5 * self.observation_length
        self.simulation_parameters = simulation_parameters or SimulationParameters()
        self.log = log or structlog.get_logger()

    @property
    def bbox(self) -> np.ndarray:
        if (flux := getattr(self, "flux", None)) is not None:
            return enmap.box(flux.shape, flux.wcs)
        if self.sky_box is not None:
            return skycoord_box_to_enmap_box(self.sky_box)
        ra_min = (
            self.simulation_parameters.center_ra - self.simulation_parameters.width_ra
        ).to_value(u.rad)
        ra_max = (
            self.simulation_parameters.center_ra + self.simulation_parameters.width_ra
        ).to_value(u.rad)
        dec_min = (
            self.simulation_parameters.center_dec - self.simulation_parameters.width_dec
        ).to_value(u.rad)
        dec_max = (
            self.simulation_parameters.center_dec + self.simulation_parameters.width_dec
        ).to_value(u.rad)
        return np.array([[dec_min, ra_max], [dec_max, ra_min]])

    def build(self):
        log = self.log.bind(parameters=self.simulation_parameters)

        # Flux
        self.flux = sim_maps.make_enmap(
            center_ra=self.simulation_parameters.center_ra,
            center_dec=self.simulation_parameters.center_dec,
            width_ra=self.simulation_parameters.width_ra,
            width_dec=self.simulation_parameters.width_dec,
            resolution=self.simulation_parameters.resolution,
            map_noise=self.simulation_parameters.map_noise,
            log=log,
        )

        self.flux_units = u.Jy

        self.map_resolution = self.simulation_parameters.resolution

        # SNR
        if (
            self.simulation_parameters.map_noise is not None
            and self.simulation_parameters.map_noise > 0.0
        ):
            log.debug(
                "simulated_map.build.noise",
                map_noise=self.simulation_parameters.map_noise,
            )
            self.snr = self.flux / self.simulation_parameters.map_noise.to_value("Jy")
        else:
            log.debug("simulated_map.build.noise.none")
            self.snr = self.flux.copy()
            self.snr.fill(1.0)  # Set SNR to 1 (i.e. all noise).

        # Time simulation is simple: a unique timestamp for each
        # pixel with time increasing mainly across RA.
        time_map = sim_maps.make_time_map(
            imap=self.flux,
            start_time=self.observation_start,
            end_time=self.observation_end,
        )
        self.time_first = time_map
        self.time_last = time_map
        self.time_mean = time_map

        log.debug("simulated_map.build.time")

        # Hits
        self._hits = self._compute_hits()
        log.debug("simulated_map.build.hits")

        log.debug("simulated_map.build.complete")

        return

    def _compute_hits(self):
        return (abs(self.flux) > 0).astype(np.int32)

    def _compute_valid_pixel_mask(self):
        bool_map = (abs(self.flux) > 0).astype(np.int32) & (np.isfinite(self.flux))
        if self.mask is not None:
            bool_map = bool_map & (self.mask > 0)
        return bool_map

    def filter_sources(self, source_positions: SkyCoord):
        """
        Filter sources based on the mask and hits map. Returns the positions of sources that are in valid pixels.

        """
        bool_map = (abs(self.flux) > 0).astype(np.int32) & (np.isfinite(self.flux))
        if self.mask is not None:
            bool_map *= self.mask

        return self._core_filter_sources(source_positions, bool_map)

    def get_pixel_times(self, pix):
        return super().get_pixel_times(pix)

    def apply_mask(self):
        return super().apply_mask()

    def finalize(self):
        super().finalize()


class SimulatedMapFromGeometry(ProcessableMap):
    def __init__(
        self,
        resolution: Quantity,
        geometry_source_map: Path,
        start_time: AwareDatetime | None,
        end_time: AwareDatetime | None,
        time_map_filename: Path | None = None,
        frequency: str | None = None,
        array: str | None = None,
        map_noise: Quantity = u.Quantity(0.01, "Jy"),
        log: FilteringBoundLogger | None = None,
    ):
        """
        Parameters
        ----------
        resolution: Quantity
            The angular resolution of the map.
        geometry_source_map: Path
            Path to the source map that defines the geometry of the simulation.
        time_map_filename: Path | None, optional
            A time map to use, if not present simulated from start time and end time.
        start_time: datetime | None, optional
            Start time of the simulated observing session.
        end_time: datetime | None, optional
            End time of the simulated observing session.
        frequency: str | None, optional
            The frequency band of the simulation, e.g. f090. Defaults to f090.
        array: str | None, optional
            The array that this represents, defaults to pa5
        map_noise: Quantity, optional
            The noise level of the simulated map. Defaults to 0 Jy.
        log: FilteringBoundLogger, optional
            Logger to use. If None, a new one will be created.
        """
        self.resolution = resolution
        self.observation_start = start_time
        self.observation_end = end_time

        self.map_noise = map_noise
        self.geometry_source_map = geometry_source_map
        self.time_map_filename = time_map_filename
        self.frequency = frequency or "f090"
        self.array = array or "pa5"

        if time_map_filename is None and start_time is None and end_time is None:
            raise RuntimeError(
                "One of time_map_filename or start_time and end_time must be provided"
            )

        self.observation_length = end_time - start_time
        self.log = log or structlog.get_logger()

    @property
    def bbox(self) -> np.ndarray:
        if (flux := getattr(self, "flux", None)) is not None:
            return enmap.box(flux.shape, flux.wcs)
        shape, wcs = enmap.read_map_geometry(str(self.geometry_source_map))
        return enmap.box(shape, wcs)

    def build(self):
        log = self.log.bind(geometry_source_map=self.geometry_source_map)

        # Read the geometry source map to get the center and width.
        # TODO: Potentially optimize by actually just reading the header.
        geom_map = enmap.read_map(str(self.geometry_source_map))

        log.debug("simulated_map.build.geometry.read")

        self.flux = sim_maps.make_noise_map(
            imap=geom_map,
            map_noise_jy=self.map_noise.to_value("Jy"),
        )

        self.flux_units = u.Jy

        self.map_resolution = self.resolution

        if self.map_noise:
            log.debug(
                "simulated_map.build.noise",
                map_noise=self.map_noise.to_value("Jy"),
            )
            self.snr = self.flux / self.map_noise.to_value("Jy")
        else:
            log.debug("simulated_map.build.noise.none")
            self.snr = self.flux.copy()
            self.snr.fill(1.0)

        if self.time_map_filename is not None:
            time_map = enmap.read_map(str(self.time_map_filename))
        else:
            time_map = sim_maps.make_time_map(
                imap=self.flux,
                start_time=self.observation_start,
                end_time=self.observation_end,
            )

        self.time_first = time_map
        self.time_last = time_map
        self.time_mean = time_map

        log.debug("simulated_map.build.time")

        # Hits
        self._hits = self._compute_hits()
        log.debug("simulated_map.build.hits")

        log.debug("simulated_map.build.complete")

        return

    def _compute_valid_pixel_mask(self):
        bool_map = (abs(self.flux) > 0).astype(np.int32) & (np.isfinite(self.flux))
        if self.mask is not None:
            bool_map = bool_map & (self.mask > 0)
        return bool_map

    def filter_sources(self, source_positions: SkyCoord):
        bool_map = (abs(self.flux) > 0).astype(np.int32) & (np.isfinite(self.flux))
        if self.mask is not None:
            bool_map *= self.mask

        return self._core_filter_sources(source_positions, bool_map)

    def get_pixel_times(self, pix):
        return super().get_pixel_times(pix)

    def apply_mask(self):
        return super().apply_mask()

    def finalize(self):
        super().finalize()
