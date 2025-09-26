"""
Source injectors -- taking simulated sources, and maps, and bringing them
together (since 1997).
"""

import math
import random
from abc import ABC, abstractmethod

from astropy import units as u
from astropy.table import Table
from photutils.psf import GaussianPSF
from photutils.psf.simulation import make_model_image

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


class SimpleSourceInjector(SourceInjector):
    def __init__(
        self,
        gauss_fwhm: u.Quantity = 2.2 * u.arcmin,
        gauss_theta_min: u.Quantity[u.deg] = 0 * u.deg,
        gauss_theta_max: u.Quantity[u.deg] = 90 * u.deg,
        fwhm_uncertainty_fraction: float = 0.01,
        progress_bar: bool = False,
    ):
        self.gauss_fwhm = gauss_fwhm
        self.gauss_theta_min = gauss_theta_min
        self.gauss_theta_max = gauss_theta_max
        self.fwhm_uncertainty_fraction = fwhm_uncertainty_fraction
        self.progress_bar = progress_bar

        return

    def inject(
        self,
        input_map: ProcessableMap,
        simulated_sources: list[SimulatedSource],
    ) -> ProcessableMapWithSimulatedSources:
        # Create the photutils source injection table and 'model image' - the
        # template to add to the map.

        # First, filter the sources for those that lie on top of a pixel with
        # a valid 'hit'.
        def source_to_array_index(source: SimulatedSource):
            return input_map.hits.wcs.world_to_array_index(
                source.position(time=input_map.observation_time)
            )

        def source_in_map(source: SimulatedSource):
            pixel = source_to_array_index(source=source)

            try:
                return bool(input_map.hits[pixel])
            except IndexError:
                return False

        valid_sources = filter(source_in_map, simulated_sources)

        shape = input_map.flux.shape
        model = GaussianPSF()

        gauss_fwhm_pixels = self.gauss_fwhm / input_map.map_resolution
        min_fwhm = (1.0 - self.fwhm_uncertainty_fraction) * gauss_fwhm_pixels
        max_fwhm = (1.0 + self.fwhm_uncertainty_fraction) * gauss_fwhm_pixels
        theta_min = self.gauss_theta_min.to_value(u.deg)
        theta_max = self.gauss_theta_max.to_value(u.deg)

        params_table = Table(
            data={
                "x_0": [source_to_array_index(x)[0] for x in valid_sources],
                "y_0": [source_to_array_index(x)[1] for x in valid_sources],
                "flux": [
                    x.flux(time=input_map.observation_time) for x in valid_sources
                ],
                "x_fwhm": [random.random(min_fwhm, max_fwhm) for _ in valid_sources],
                "y_fwhm": [random.random(min_fwhm, max_fwhm) for _ in valid_sources],
                "theta": [random.random(theta_min, theta_max) for _ in valid_sources],
            }
        )

        model_image = make_model_image(
            shape=shape,
            model=model,
            params_table=params_table,
            progress_bar=self.progress_bar,
        )

        # Unit conversion: photituls distributes the flux over the solid angle
        # so this map should be the intensity map in Jy/sr.

        # Beam solid angle
        omega_b = (math.pi / (4.0 * math.log(2.0))) * self.gauss_fwhm * self.gauss_fwhm
        omega_b /= input_map.map_resolution**2

        simulated_source_flux_map = model_image * omega_b

        new_flux = input_map.flux + simulated_source_flux_map

        map_noise = input_map.snr / input_map.flux - 1.0
        new_snr = new_flux / map_noise

        return ProcessableMapWithSimulatedSources(
            flux=new_flux, snr=new_snr, time=input_map.time_mean, original_map=input_map
        )
