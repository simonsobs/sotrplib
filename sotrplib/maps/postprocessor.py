"""
Dependencies for post-processing (e.g. flat-fielding). Affects
maps in-place. These operations happen after 'finalize', i.e. once
the snr and flux arrays already exist.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import structlog
from astropy import units as u
from astropydantic import AstroPydanticQuantity
from pixell import enmap
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.maps import flat_field_using_photutils, flat_field_using_pixell
from sotrplib.maps.masks import mask_dustgal


class MapPostprocessor(ABC):
    """Abstract base class for map postprocessors.

    Postprocessors run after ``ProcessableMap.finalize`` and operate on the
    finalised ``snr`` and ``flux`` arrays.
    """

    @abstractmethod
    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        """Apply postprocessing to a finalised map.

        Parameters
        ----------
        input_map : ProcessableMap
            Finalised map to postprocess.  Modified in place.

        Returns
        -------
        ProcessableMap
            The (possibly modified) input map.
        """
        return input_map


class GalaxyMask(MapPostprocessor):
    """Postprocessor that multiplies the flux and SNR maps by a galactic mask.

    Parameters
    ----------
    mask_path : Path
        Path to the FITS galactic mask file.
    invert : bool, optional
        If ``True``, use ``1 - mask`` (mask the non-galactic region).
        Default is ``False``.
    log : FilteringBoundLogger, optional
        Structured logger.
    """

    mask_path: Path
    invert: bool = False
    log: FilteringBoundLogger | None = None

    def __init__(
        self,
        mask_path: Path,
        invert: bool = False,
        log: FilteringBoundLogger | None = None,
    ):
        self.mask_path = mask_path
        self.invert = invert
        self.log = log or structlog.get_logger()

    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        galaxy_mask = mask_dustgal(
            imap=input_map.flux,
            galmask=enmap.read_map(
                fname=str(self.mask_path),
                box=input_map.bbox,
            ),
        )
        if self.invert:
            galaxy_mask = 1 - galaxy_mask

        input_map.flux *= galaxy_mask
        input_map.snr *= galaxy_mask

        return input_map


class PixellFlatFielder(MapPostprocessor):
    """Postprocessor that flat-fields the SNR map using the pixell tiling module.

    Parameters
    ----------
    sigma_val : float, optional
        Sigma-clipping threshold (currently unused; reserved for future use).
        Default is 5.0.
    tile_size : Quantity, optional
        Tile size for the flat-fielding grid.  Default is 1 deg.
    mask : enmap.ndmap, optional
        Additional mask to exclude pixels from the flat-field estimate.
    log : FilteringBoundLogger, optional
        Structured logger.
    """

    tile_size: AstroPydanticQuantity = AstroPydanticQuantity(1.0 * u.deg)
    sigma_val: float = 5.0

    def __init__(
        self,
        sigma_val: float = 5.0,
        tile_size: u.Quantity = 1.0 * u.deg,
        mask: enmap.ndmap | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.sigma_val = sigma_val
        self.tile_size = tile_size
        self.mask = mask
        self.log = log or structlog.get_logger()

    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        return flat_field_using_pixell(
            mapdata=input_map,
            tilegrid=self.tile_size.to_value(u.deg),
            log=self.log,
        )


class PhotutilsFlatFielder(MapPostprocessor):
    """Postprocessor that flat-fields the SNR and flux maps using photutils Background2D.

    Parameters
    ----------
    sigma_val : float, optional
        Sigma-clipping threshold for background estimation.  Default is 5.0.
    tile_size : Quantity, optional
        Tile size for the background grid.  Default is 1 deg.
    mask : enmap.ndmap, optional
        Additional mask to exclude pixels from the background estimate.
        Combined with ``input_map.mask`` if both are present.
    log : FilteringBoundLogger, optional
        Structured logger.
    """

    tile_size: AstroPydanticQuantity = AstroPydanticQuantity(1.0 * u.deg)
    sigma_val: float = 5.0

    def __init__(
        self,
        sigma_val: float = 5.0,
        tile_size: u.Quantity = 1.0 * u.deg,
        mask: enmap.ndmap | None = None,
        log: FilteringBoundLogger | None = None,
    ):
        self.sigma_val = sigma_val
        self.tile_size = tile_size
        self.mask = mask
        self.log = log or structlog.get_logger()

    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        if self.mask is not None and input_map.mask is not None:
            combined_mask = self.mask * input_map.mask
        elif self.mask is not None:
            combined_mask = self.mask
        elif input_map.mask is not None:
            combined_mask = input_map.mask
        else:
            combined_mask = None
        return flat_field_using_photutils(
            mapdata=input_map,
            tilegrid=self.tile_size,
            mask=combined_mask,
            sigmaclip=self.sigma_val,
            log=self.log,
        )
