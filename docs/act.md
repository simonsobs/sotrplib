Example ACT Pipeline
====================

Below, we show an example of the pipeline running on some pre match-filtered
maps from the Atacama Cosmology Telescope.

```python
from astropy import units as u
from pathlib import Path

from sotrplib.handlers.basic import PipelineRunner
from sotrplib.maps.core import RhoAndKappaMap
from sotrplib.maps.preprocessor import KappaRhoCleaner
from sotrplib.outputs.core import PickleSerializer
from sotrplib.sifter.core import DefaultSifter
from sotrplib.source_catalog.database import (
    MockACTDatabase,
    MockForcedPhotometryDatabase,
)
from sotrplib.sources.blind import BlindSearchParameters, SigmaClipBlindSearch
from sotrplib.sources.force import SimpleForcedPhotometry
from sotrplib.sources.subtractor import PhotutilsSourceSubtractor

# The location where the timecode 15386 maps are stored
map_path = Path("./")

# Create the map object; this will provide (eventually) the
# flux, SNR, and time maps.
filtered_map = RhoAndKappaMap(
    rho_filename=map_path / "depth1_1538613353_pa5_f090_rho.fits",
    kappa_filename=map_path / "depth1_1538613353_pa5_f090_kappa.fits",
    time_filename=map_path / "depth1_1538613353_pa5_f090_time.fits",
    flux_units=u.Jy,
    frequency="f090",
    array="pa5"
)

# Set up the sources we wish to search for; in this case we use
# a FITS table.
catalog_path = Path("./PS_S19_f090_2pass_optimalCatalog.fits")

forced_photometry_catalog = MockForcedPhotometryDatabase(
    db_path=catalog_path,
    flux_lower_limit=u.Quantity(1.0, "Jy")
)

# Now the full list of catalog sources to match against during the
# 'sifting' phase. This is the same catalog but without the flux limit.
catalog = MockACTDatabase(
    db_path=catalog_path
)

# Set a very high SNR cut for blind searches
blind_search_params = BlindSearchParameters(sigma_threshold=20.0)

# Create the whole pipeline object:
runner = PipelineRunner(
    maps=[filtered_map], # The pipeline can run a loop over maps if necessary
    preprocessors=[KappaRhoCleaner()], # Clean the rho/kappa maps before snr conversion
    postprocessors=None,
    source_simulators=None,
    forced_photometry=SimpleForcedPhotometry(
        mode="nn", sources=forced_photometry_catalog.source_list
    ), # Perform very simple forced photometry using nearest neighbor calculations
    source_subtractor=PhotutilsSourceSubtractor(), # Remove sources before blind search
    blind_search=SigmaClipBlindSearch(parameters=blind_search_params),
    sifter=DefaultSifter(catalog_sources=catalog.source_list), # Sift out known v.s. truly unknown sources
    outputs=[PickleSerializer(directory=Path("."))] # Write out data as pickle file
)
```