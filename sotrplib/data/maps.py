"""
Load and use maps
"""

from pydantic import BaseModel
from abc import ABC, abstractmethod
from pathlib import Path
from pixell import enmap

class MapBase(ABC):
    intensity: enmap.ndmap
    inverse_variance: enmap.ndmap
    rho: enmap.ndmap
    kappa: enmap.ndmap
    time: enmap.ndmap


class Depth1Map(BaseModel, MapBase):
    intensity_filename: Path
    inverse_variance_name: Path
    rho_name: Path
    kappa_name: Path
    time_name: Path

    def model_post_init(__context) -> None:
        __context.intensity = enmap.read_map(__context.intensity_filename, sel=0)
        __context.inverse_variance = enmap.read_map(__context.inverse_variance_name)
        __context.rho = enmap.read_map(__context.rho_name, sel=0)
        __context.kappa = enmap.read_map(__context.kappa_name, sel=0)
        __context.time = enmap.read_map(__context.time_name, sel=0)



