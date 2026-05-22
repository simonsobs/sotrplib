import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from sotrplib.sims.sim_source_generators import FixedSourceGenerator
from sotrplib.sims.source_injector import PhotutilsSourceInjector


@pytest.fixture
def fixed_source_map_builder():
    """Build injected test maps while keeping sources away from map edges."""

    def _build(
        input_map,
        *,
        n_sources,
        min_flux,
        max_flux,
        catalog_fraction=1.0,
        fwhm_uncertainty_fraction=0.01,
        edge_buffer=0.25 * u.deg,
    ):
        generator = FixedSourceGenerator(
            min_flux=min_flux,
            max_flux=max_flux,
            number=n_sources,
            catalog_fraction=catalog_fraction,
        )
        buffered_box = getattr(input_map, "bbox", None)
        if buffered_box is not None:
            buffered_box = [
                SkyCoord(
                    ra=buffered_box[0].ra + edge_buffer,
                    dec=buffered_box[0].dec + edge_buffer,
                ),
                SkyCoord(
                    ra=buffered_box[1].ra - edge_buffer,
                    dec=buffered_box[1].dec - edge_buffer,
                ),
            ]

        simulated_sources, catalog = generator.generate(box=buffered_box)

        injector = PhotutilsSourceInjector(
            fwhm_uncertainty_fraction=fwhm_uncertainty_fraction
        )
        _, new_map = injector.inject(
            input_map=input_map,
            simulated_sources=simulated_sources,
        )

        return new_map, catalog.sources

    return _build
