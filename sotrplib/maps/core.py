"""
Core map objects.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import structlog
from astropy.units import Quantity, Unit
from numpy.typing import ArrayLike
from pixell import enmap
from pixell.enmap import ndmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.maps import widefield_geometry


class ProcessableMap(ABC):
    snr: ndmap
    "The signal-to-noise ratio map"
    flux: ndmap
    "The flux map"
    time: ndmap | None
    "The time map. If None, usually when the map is a coadd, see the time range"

    observation_length: timedelta
    start_time: datetime
    end_time: datetime

    flux_units: Unit

    @abstractmethod
    def build(self):
        """
        Build the maps for snr, flux, and time. Could include simulating them,
        or reading them from disk.
        """
        return

    @abstractmethod
    def finalize(self):
        """
        Called after source injection to ensure that the snr, flux, and time maps
        are available.
        """
        return


class SimulatedMap(ProcessableMap):
    def __init__(
        self,
        resolution: Quantity,
        start_time: datetime,
        end_time: datetime,
        box: ArrayLike | None = None,
        include_half_pixel_offset: bool = False,
    ):
        """
        Parameters
        ----------
        resolution: Quantity
            The angular resolution of the map.
        start_time: datetime
            Start time of the simulated observing session.
        end_time: datetime
            End time of the simulated observing session.
        box: np.ndarray, optional
            Optional sky box to simulate in. Otherwise covers the whole sky.
        include_half_pixel_offset: bool, False
            Include the half-pixel offset in pixell constructors.
        """
        self.resolution = resolution
        self.start_time = start_time
        self.end_time = end_time
        self.box = box
        self.include_half_pixel_offset = include_half_pixel_offset

        self.observation_length = end_time - start_time
        self.time = None

    def build(self):
        # This is all wrong. See the map sims section in load_map where the
        # real map simulation actually ocurrs...
        if self.box is None:
            empty_map = enmap.zeros(
                *widefield_geometry(
                    res=self.resolution.to_value("arcmin"),
                    include_half_pixel_offset=self.include_half_pixel_offset,
                )
            )
        else:
            shape, wcs = enmap.geometry(self.box, res=self.res)
            empty_map = enmap.zeros(shape, wcs=wcs)

        self.intensity = empty_map.copy()
        self.inverse_variance = empty_map.copy()
        self.time = None

        # TODO: Add hook for simulating sources?

        return

    def finalize(self):
        # TODO: conversion from intensity and ivar to flux and snr.
        return


class IntensityAndInverseVarianceMap(ProcessableMap):
    """
    A set of FITS maps read from disk. Could be Depth 1, could
    be monthly or weekly co-adds. Or something else!
    """

    def __init__(
        self,
        intensity_filename: Path,
        inverse_variance_filename: Path,
        start_time: datetime,
        end_time: datetime,
        box: ArrayLike | None,
        time_filename: Path | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.intensity_filename = intensity_filename
        self.inverse_variance_filename = inverse_variance_filename
        self.time_filename = time_filename
        self.start_time = start_time
        self.end_time = end_time
        self.box = box
        self.log = log or structlog.get_logger()

    def build(self):
        log = self.log.bind(intensity_filename=self.intensity_filename)
        try:
            self.intensity = enmap.read_map(
                str(self.intensity_filename), sel=0, box=self.box
            )
            log.debug("intensity_ivar.intensity.read.sel")
        except IndexError:
            # Intensity map does not have Q, U
            self.intensity = enmap.read_map(str(self.intensity_filename), box=self.box)
            log.debug("intensity_ivar.intensity.read.nosel")

        log = log.new(inverse_variance_filename=self.inverse_variance_filename)
        self.inverse_variance = enmap.read_map(
            self.inverse_variance_filename, box=self.box
        )
        log.debug("intensity_viar.ivar.read")

        log = log.new(time_filename=self.time_filename)
        if self.time_filename is not None:
            # TODO: Handle nuance that the start time is not included.
            self.time = enmap.read_map(self.time_filename, box=self.box)
            log.debug("intensity_ivar.time.read")
        else:
            self.time = None
            log.debug("intensity_ivar.time.none")

        return

    def finalize(self):
        # Do matched filtering to get snr and flux maps.
        return
