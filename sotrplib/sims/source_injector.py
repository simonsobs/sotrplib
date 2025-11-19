"""
Source injectors -- taking simulated sources, and maps, and bringing them
together (since 1997).
"""

import math
import random
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import numpy as np
from astropy import units as u
from astropy.table import Table
from photutils.psf import GaussianPSF
from photutils.psf.simulation import make_model_image
from structlog import get_logger
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sims.sources.core import ProcessableMapWithSimulatedSources

from .sim_sources import SimulatedSource


class SourceInjector(ABC):
    @abstractmethod
    def inject(
        self,
        input_map: ProcessableMap,
        simulated_sources: list[SimulatedSource],
    ) -> ProcessableMapWithSimulatedSources:
        return


class EmptySourceInjector(SourceInjector):
    def inject(
        self,
        input_map: ProcessableMap,
        simulated_sources: list[SimulatedSource],
    ) -> ProcessableMapWithSimulatedSources:
        return input_map


class PhotutilsSourceInjector(SourceInjector):
    def __init__(
        self,
        gauss_fwhm: u.Quantity = 2.2 * u.arcmin,
        gauss_theta_min: u.Quantity[u.deg] = 0 * u.deg,
        gauss_theta_max: u.Quantity[u.deg] = 90 * u.deg,
        fwhm_uncertainty_fraction: float = 0.01,
        progress_bar: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.gauss_fwhm = gauss_fwhm
        self.gauss_theta_min = gauss_theta_min
        self.gauss_theta_max = gauss_theta_max
        self.fwhm_uncertainty_fraction = fwhm_uncertainty_fraction
        self.progress_bar = progress_bar
        self.log = log or get_logger()

        return

    def inject(
        self,
        input_map: ProcessableMap,
        simulated_sources: list[SimulatedSource],
    ) -> ProcessableMapWithSimulatedSources:
        # Create the photutils source injection table and 'model image' - the
        # template to add to the map.

        log = self.log.bind(num_simulated_sources=len(simulated_sources))

        # First, filter the sources for those that lie on top of a pixel with
        # a valid 'hit'.
        def source_to_array_index(source: SimulatedSource):
            return input_map.flux.wcs.world_to_array_index(
                source.position(
                    time=input_map.time_mean
                )  ## TODO: if fast evolving, mean time isnt good enough
            )

        def source_in_map(source: SimulatedSource):
            pixel = source_to_array_index(source=source)

            try:
                return np.any(np.nan_to_num(input_map.flux[pixel]).astype(bool))
            except IndexError:
                return False

        valid_sources = list(filter(source_in_map, simulated_sources))
        observed_times = [
            datetime.fromtimestamp(
                float(input_map.time_mean[source_to_array_index(x)]), tz=timezone.utc
            )
            for x in valid_sources
        ]
        log = log.bind(num_valid_sources=len(valid_sources))

        shape = input_map.flux.shape
        model = GaussianPSF()

        ## TODO: need to get flux at times for each pixel the source is in.
        ## if it's moving, would add to multiple pixels at different times.
        ## if flaring need to calculate flux at integrated time.

        gauss_fwhm_pixels = self.gauss_fwhm.to_value(
            u.arcmin
        ) / input_map.map_resolution.to_value(u.arcmin)
        min_fwhm = (1.0 - self.fwhm_uncertainty_fraction) * gauss_fwhm_pixels
        max_fwhm = (1.0 + self.fwhm_uncertainty_fraction) * gauss_fwhm_pixels
        theta_min = self.gauss_theta_min.to_value(u.deg)
        theta_max = self.gauss_theta_max.to_value(u.deg)

        log = log.bind(
            gauss_fwhm_pixels=gauss_fwhm_pixels, min_fwhm=min_fwhm, max_fwhm=max_fwhm
        )

        table_data = {
            "x_0": [source_to_array_index(x)[1] for x in valid_sources],
            "y_0": [source_to_array_index(x)[0] for x in valid_sources],
            "flux": [
                x.flux(time=observed_times[i]).to_value(input_map.flux_units)
                for i, x in enumerate(valid_sources)
            ],
            "x_fwhm": [random.uniform(min_fwhm, max_fwhm) for _ in valid_sources],
            "y_fwhm": [random.uniform(min_fwhm, max_fwhm) for _ in valid_sources],
            "theta": [random.uniform(theta_min, theta_max) for _ in valid_sources],
        }

        params_table = Table(data=table_data)

        model_image = make_model_image(
            shape=shape,
            model=model,
            params_table=params_table,
            progress_bar=self.progress_bar,
        )

        log.info("source_injection.photutils.model_image_complete")

        # Unit conversion: photituls distributes the flux over the solid angle
        # so this map should be the intensity map in Jy/sr.

        # Beam solid angle
        omega_b = (math.pi / (4.0 * math.log(2.0))) * self.gauss_fwhm * self.gauss_fwhm
        omega_b /= input_map.map_resolution**2

        log = log.bind(omega_b=omega_b)

        simulated_source_flux_map = model_image * omega_b

        # Must assign array indicies so that we can add the maps together and
        # have them _remain ENMAPS_.

        new_flux = input_map.flux.copy()
        new_flux[:] += simulated_source_flux_map[:]

        map_noise = input_map.flux / input_map.snr
        new_snr = input_map.snr.copy()
        new_snr[:] = (new_flux / map_noise)[:]

        log.info("source_injection.photutils.complete")

        return ProcessableMapWithSimulatedSources(
            flux=new_flux, snr=new_snr, time=input_map.time_mean, original_map=input_map
        )
