from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Telescope choice
    observatory: Literal["SO", "ACT", "SPT"] = "SO"
    instrument: Literal["LAT", "ACTPol", "SPT3G", "SAT1", "SAT2", "SAT3"] = "LAT"

    # File locations
    maps: list[Path] = []
    "input maps... the map.fits ones."
    solar_system: Path | None = None
    "path asteroid ephem directories"
    output_dir: Path | None = None
    "path to output databases and / or plots"
    source_cat: Path | None = None
    "path to source catalog"

    # Map settings
    grid_size: float = Field(
        default=1.0, validation_alias=AliasChoices("g", "grid-size", "gridsize")
    )
    "Flatfield tile gridsize (deg)"
    edge_cut: int = 20
    "Npixels to cut from map edges"

    # Read environment and command line settings to override default
    model_config = SettingsConfigDict(
        env_prefix="sotrp_", cli_parse_args=True, extra="ignore"
    )

    # Read json config settings to override default + environment
    @classmethod
    def from_file(cls, config_path: Path | str) -> "Settings":
        with open(config_path, "r") as handle:
            return cls.model_validate_json(handle.read())
