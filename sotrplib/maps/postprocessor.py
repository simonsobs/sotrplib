"""
Dependencies for post-processing (e.g. flat-fielding). Affects
maps in-place. These operations happen after 'finalize', i.e. once
the snr and flux arrays already exist.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from astropy import units as u
from pixell import enmap

from sotrplib.maps.core import ProcessableMap
from sotrplib.maps.masks import mask_dustgal, mask_edge


class MapPostprocessor(ABC):
    @abstractmethod
    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        """
        Provides postprocessing for the input map
        """
        return input_map


class GalaxyMask(MapPostprocessor):
    mask_filename: Path

    def __init__(self, mask_filename: Path):
        self.mask_filename = mask_filename

    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        galaxy_mask = mask_dustgal(
            imap=input_map.flux,
            galmask=enmap.read_map(
                fname=self.mask_filename,
                box=input_map.box,
            ),
        )

        input_map.flux *= galaxy_mask
        input_map.snr *= galaxy_mask

        return input_map


class EdgeMask(MapPostprocessor):
    mask_by: u.Quantity

    def __init__(self, mask_by: u.Quantity = u.Quantity(10.0, "arcmin")):
        self.mask_by = mask_by

    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        # TODO: extract resolution from input map!
        number_of_pixels = self.mask_by / input_map.resolution

        edge_mask = mask_edge(imap=input_map.flux, pix_num=number_of_pixels)

        input_map.flux *= edge_mask
        input_map.snr *= edge_mask

        return input_map


class PixellFlatfield(MapPostprocessor):
    tile_grid: float

    def __init__(self, tile_grid: float = 0.5):
        self.tile_grid = tile_grid

    def postprocess(self, input_map: ProcessableMap) -> ProcessableMap:
        return
