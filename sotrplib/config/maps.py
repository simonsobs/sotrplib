"""
Map configuration
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Literal

from astropy import units as u
from astropydantic import AstroPydanticICRS, AstroPydanticQuantity, AstroPydanticUnit
from pydantic import AwareDatetime, BaseModel, Field, model_validator
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import (
    FluxAndSNRMap,
    IntensityAndInverseVarianceMap,
    ProcessableMap,
    RhoAndKappaMap,
)
from sotrplib.maps.database import FluxMapReader, IntensityMapReader, RhoKappaMapReader
from sotrplib.sims.maps import (
    SimulatedMap,
    SimulatedMapFromGeometry,
    SimulationParameters,
)


class MapConfig(BaseModel, ABC):
    """Abstract base for single-map configuration objects.

    Each subclass defines a ``map_type`` discriminator and implements
    ``to_map`` to construct the corresponding ``ProcessableMap``.
    """

    map_type: str

    @abstractmethod
    def to_map(self, log: FilteringBoundLogger | None = None) -> ProcessableMap:
        """Construct the configured map object.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger to pass to the map.

        Returns
        -------
        ProcessableMap
            The constructed map instance.
        """
        return


class MapGeneratorConfig(BaseModel, ABC):
    """Abstract base for map-generator configuration objects.

    Each subclass defines a ``map_generator_type`` discriminator and
    implements ``to_generator`` to construct an iterable of maps (e.g. a
    database reader).
    """

    map_generator_type: str

    @abstractmethod
    def to_generator(
        self, log: FilteringBoundLogger | None = None
    ) -> Iterable[ProcessableMap]:
        """Construct the configured map generator.

        Parameters
        ----------
        log : FilteringBoundLogger, optional
            Structured logger.

        Returns
        -------
        Iterable[ProcessableMap]
            An iterable that yields ``ProcessableMap`` instances.
        """
        return


class SimulatedMapConfig(MapConfig):
    """Configuration for a purely simulated map.

    Creates a ``SimulatedMap`` with Gaussian noise over the SO wide-field
    geometry.

    Fields
    ------
    observation_start : AwareDatetime
        Simulated observation start time.
    observation_end : AwareDatetime
        Simulated observation end time.
    frequency : str
        Frequency band string (default ``"f090"``).
    array : str
        Detector array name (default ``"pa5"``).
    simulation_parameters : SimulationParameters
        Noise and beam parameters for the simulation.
    """

    map_type: Literal["simulated"] = "simulated"
    observation_start: AwareDatetime
    observation_end: AwareDatetime
    frequency: str = "f090"
    array: str = "pa5"
    simulation_parameters: SimulationParameters = Field(
        default_factory=SimulationParameters
    )

    def to_map(self, log: FilteringBoundLogger | None = None) -> SimulatedMap:
        return SimulatedMap(
            observation_start=self.observation_start,
            observation_end=self.observation_end,
            frequency=self.frequency,
            array=self.array,
            simulation_parameters=self.simulation_parameters,
            log=log,
        )


class SimulatedMapFromGeometryConfig(MapConfig):
    """Configuration for a simulated map that inherits geometry from an existing file.

    Fields
    ------
    resolution : Quantity[arcmin]
        Pixel resolution of the output map.
    geometry_source_map : Path
        Path to a FITS file whose geometry is used as the template.
    observation_start, observation_end : AwareDatetime or None
        Observation time bounds.  Either these or ``time_map_filename`` must
        be provided.
    time_map_filename : Path or None
        Per-pixel time map to use instead of fixed start/end times.
    frequency : str
        Frequency band string (default ``"f090"``).
    array : str
        Detector array name (default ``"pa5"``).
    map_noise : Quantity[Jy]
        RMS noise level for the simulated map (default 0.01 Jy).
    """

    map_type: Literal["simulated_geometry"] = "simulated_geometry"
    resolution: AstroPydanticQuantity[u.arcmin]
    geometry_source_map: Path
    observation_start: AwareDatetime | None
    observation_end: AwareDatetime | None
    time_map_filename: Path | None
    frequency: str = "f090"
    array: str = "pa5"
    map_noise: AstroPydanticQuantity[u.Jy] = u.Quantity(0.01, "Jy")

    @model_validator(mode="after")
    def check_time_or_map(self):
        if (
            self.time_map_filename is None
            and self.observation_start is None
            and self.observation_end is None
        ):
            raise ValueError(
                "One of time_map_filename or start_time and end_time must be provided"
            )

    def to_map(
        self, log: FilteringBoundLogger | None = None
    ) -> SimulatedMapFromGeometry:
        return SimulatedMapFromGeometry(
            resolution=self.resolution,
            geometry_source_map=self.geometry_source_map,
            observation_start=self.observation_start,
            observation_end=self.observation_end,
            time_map_filename=self.time_map_filename,
            frequency=self.frequency,
            array=self.array,
            map_noise=self.map_noise,
            log=log,
        )


class RhoKappaMapConfig(MapConfig):
    """Configuration for a matched-filtered rho/kappa map read from FITS files.

    Fields
    ------
    rho_map_path, kappa_map_path : Path
        Paths to the rho and kappa FITS files.
    time_map_path : Path, optional
        Path to the per-pixel time FITS file.
    info_path : Path, optional
        Path to the observation info HDF file.
    frequency, array, instrument : str or None
        Map metadata.
    observation_start, observation_end : AwareDatetime or None
        Observation time bounds.
    sky_box : list of AstroPydanticICRS or None
        Bounding box for a sky cutout.
    flux_units : Unit
        Physical units of the derived flux (default Jy).
    """

    map_type: Literal["filtered"] = "filtered"
    rho_map_path: Path
    kappa_map_path: Path
    time_map_path: Path | None = None
    info_path: Path | None = None
    frequency: str | None = "f090"
    array: str | None = "pa5"
    instrument: str | None = None
    observation_start: AwareDatetime | None = None
    observation_end: AwareDatetime | None = None
    sky_box: list[AstroPydanticICRS] | None = None
    flux_units: AstroPydanticUnit = u.Unit("Jy")

    def to_map(self, log: FilteringBoundLogger | None = None) -> RhoAndKappaMap:
        return RhoAndKappaMap(
            rho_filename=self.rho_map_path,
            kappa_filename=self.kappa_map_path,
            start_time=self.observation_start,
            end_time=self.observation_end,
            time_filename=self.time_map_path,
            info_filename=self.info_path,
            sky_box=self.sky_box,
            frequency=self.frequency,
            array=self.array,
            instrument=self.instrument,
            flux_units=self.flux_units,
            log=log,
        )


class InverseVarianceMapConfig(MapConfig):
    """Configuration for an intensity + inverse-variance map read from FITS files.

    Fields
    ------
    intensity_map_path : Path
        Path to the intensity FITS file.
    weights_map_path : Path
        Path to the inverse-variance FITS file.
    time_map_path : Path, optional
        Path to the per-pixel time FITS file.
    info_path : Path, optional
        Path to the observation info HDF file.
    frequency, array, instrument : str or None
        Map metadata.
    observation_start, observation_end : AwareDatetime or None
        Observation time bounds.
    sky_box : list of AstroPydanticICRS or None
        Bounding box for a sky cutout.
    intensity_units : Unit
        Physical units of the intensity map (default K).
    """

    map_type: Literal["inverse_variance"] = "inverse_variance"
    intensity_map_path: Path
    weights_map_path: Path
    time_map_path: Path | None = None
    info_path: Path | None = None
    frequency: str | None = "f090"
    array: str | None = "pa5"
    instrument: str | None = None
    observation_start: AwareDatetime | None = None
    observation_end: AwareDatetime | None = None
    sky_box: list[AstroPydanticICRS] | None = None
    intensity_units: AstroPydanticUnit = u.Unit("K")

    def to_map(
        self, log: FilteringBoundLogger | None = None
    ) -> IntensityAndInverseVarianceMap:
        return IntensityAndInverseVarianceMap(
            intensity_filename=self.intensity_map_path,
            inverse_variance_filename=self.weights_map_path,
            time_filename=self.time_map_path,
            info_filename=self.info_path,
            start_time=self.observation_start,
            end_time=self.observation_end,
            sky_box=self.sky_box,
            frequency=self.frequency,
            array=self.array,
            instrument=self.instrument,
            intensity_units=self.intensity_units,
            log=log,
        )


class FluxAndSNRMapConfig(MapConfig):
    """Configuration for a pre-computed flux + SNR map read from FITS files.

    Fields
    ------
    flux_map_path, snr_map_path : Path
        Paths to the flux and SNR FITS files.
    time_map_path : Path, optional
        Path to the per-pixel time FITS file.
    info_path : Path, optional
        Path to the observation info HDF file.
    frequency, array, instrument : str or None
        Map metadata.
    observation_start, observation_end : AwareDatetime or None
        Observation time bounds.
    sky_box : list of AstroPydanticICRS or None
        Bounding box for a sky cutout.
    flux_units : Unit
        Physical units of the flux map (default Jy).
    """

    map_type: Literal["flux_and_snr"] = "flux_and_snr"
    flux_map_path: Path
    snr_map_path: Path
    time_map_path: Path | None = None
    info_path: Path | None = None
    frequency: str | None = "f090"
    array: str | None = "pa5"
    instrument: str | None = None
    observation_start: AwareDatetime | None = None
    observation_end: AwareDatetime | None = None
    sky_box: list[AstroPydanticICRS] | None = None
    flux_units: AstroPydanticUnit = u.Unit("Jy")

    def to_map(self, log: FilteringBoundLogger | None = None) -> FluxAndSNRMap:
        return FluxAndSNRMap(
            flux_filename=self.flux_map_path,
            snr_filename=self.snr_map_path,
            time_filename=self.time_map_path,
            info_filename=self.info_path,
            start_time=self.observation_start,
            end_time=self.observation_end,
            sky_box=self.sky_box,
            frequency=self.frequency,
            array=self.array,
            instrument=self.instrument,
            flux_units=self.flux_units,
            log=log,
        )


class MapCatDatabaseConfig(MapGeneratorConfig):
    """Configuration for reading maps from the mapcat database.

    Fields
    ------
    frequency, array, instrument : str or None
        Filter maps by these metadata values.  ``None`` matches all.
    number_to_read : int or None
        Maximum number of maps to return.  ``None`` reads all matching maps.
    start_time, end_time : AwareDatetime or None
        Time window for the query.
    map_ids : list of int or None
        Explicit list of mapcat IDs to read.
    sky_box : list of AstroPydanticICRS or None
        Bounding box for sky cutouts.
    map_units : Unit
        Physical units of the map intensities (default K).
    map_type : {"intensity", "flux", "rhokappa"}
        Which map representation to load.
    rerun : bool
        If ``True``, re-process maps even if already in the pipeline DB.
    """

    map_generator_type: Literal["mapcat_database"] = "mapcat_database"
    frequency: str | None = None
    array: str | None = None
    instrument: str | None = None
    number_to_read: int | None = None  ## if None, all in database will be read
    start_time: AwareDatetime | None = None
    end_time: AwareDatetime | None = None
    map_ids: list[int] | None = None
    sky_box: list[AstroPydanticICRS] | None = None
    map_units: AstroPydanticUnit = u.Unit("K")
    map_type: Literal["intensity", "flux", "rhokappa"] = "intensity"
    rerun: bool = False

    def to_generator(
        self, log: FilteringBoundLogger | None = None
    ) -> Iterable[ProcessableMap]:
        _reader_cls = {
            "intensity": IntensityMapReader,
            "rhokappa": RhoKappaMapReader,
            "flux": FluxMapReader,
        }[self.map_type]
        return _reader_cls(
            number_to_read=self.number_to_read,
            start_time=self.start_time,
            end_time=self.end_time,
            frequency=self.frequency,
            array=self.array,
            instrument=self.instrument,
            sky_box=self.sky_box,
            map_ids=self.map_ids,
            map_units=self.map_units,
            rerun=self.rerun,
            log=log,
        )


AllMapConfigTypes = (
    SimulatedMapConfig
    | SimulatedMapFromGeometryConfig
    | RhoKappaMapConfig
    | InverseVarianceMapConfig
    | FluxAndSNRMapConfig
)

AllMapGeneratorConfigTypes = MapCatDatabaseConfig | list[AllMapConfigTypes]
