from typing import Any
from pydantic import BaseModel, Field, ValidationError, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from pathlib import Path
import json

class Settings(BaseSettings):
	# Telescope choice
	observatory: str = "SO"
	instrument: str = "LAT"

	# File locations
	maps: str = "" # input maps... the map.fits ones.
	map_dir: str = "" # Location of the depth-1 maps
	solar_system: str = "" # path asteroid ephem directories	
	output_dir: str = "" # path to output databases and / or plots
	source_cat: str = "" # path to source catalog

	# Map settings
	grid_size: float = Field(default=1.0, validation_alias=AliasChoices('g','grid-size', 'gridsize')) # Flatfield tile gridsize (deg)
	edge_cut: int = 20 # Npixels to cut from map edges

	# Read environment and command line settings to override default
	model_config = SettingsConfigDict(env_prefix='sotrp_', cli_parse_args=True, extra='ignore')

	# Read json config settings to override default + environment
	@classmethod
	def from_file(cls, config_path: Path | str) -> "Settings":
		with open(config_path, "r") as handle:
			return cls.model_validate_json(handle.read())

