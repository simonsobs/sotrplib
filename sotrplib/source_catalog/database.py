import os
import threading
from pathlib import Path

import pandas as pd
import structlog
from astropy import units as u
from astropy.io import fits
from filelock import FileLock  # Import FileLock for file-based locking
from socat.client import mock
from structlog.types import FilteringBoundLogger

from sotrplib.maps.core import ProcessableMap
from sotrplib.sources.sources import CrossMatch, MeasuredSource, RegisteredSource
from sotrplib.utils.utils import angular_separation


class MockDatabase:
    def __init__(self, log: FilteringBoundLogger | None = None):
        self.cat = mock.Client()
        self.log = log or structlog.get_logger()
        self.log.info("mock_database.initialized")

    def add_source(self, ra: u.Quantity, dec: u.Quantity, name: str):
        self.cat.create(ra=ra.to(u.deg).value, dec=dec.to(u.deg).value, name=name)

    def get_nearby_source(
        self, ra: u.Quantity, dec: u.Quantity, radius: u.Quantity = 0.1 * u.deg
    ):
        ra = ra.to(u.deg).value
        dec = dec.to(u.deg).value
        radius = radius.to(u.deg).value
        ## ra,dec,radius in same units. default is decimal degrees
        nearby = self.cat.get_box(
            ra_min=ra - radius,
            ra_max=ra + radius,
            dec_min=dec - radius,
            dec_max=dec + radius,
        )
        self.log.info(
            "mock_database.get_nearby_source",
            ra=ra,
            dec=dec,
            radius=radius,
            nearby=nearby,
        )
        sources = []
        if nearby:
            sources = [
                RegisteredSource(
                    ra=s.ra * u.deg,
                    dec=s.dec * u.deg,
                    source_id=str(s.id),
                    crossmatches=[
                        CrossMatch(
                            source_id=str(s.name),
                            probability=1.0,
                            distance=angular_separation(ra, dec, s.ra, s.dec),
                            frequency=90 * u.GHz,
                            catalog_name="mock",
                            catalog_idx=None,
                        )
                    ],
                )
                for s in nearby
            ]
        self.log.info(
            "mock_database.get_nearby_source.nearby_sources",
            sources=sources,
        )
        return sources

    def get_sources_in_box(self, box: list[list[u.Quantity]]) -> list[RegisteredSource]:
        """
        box is [[dec_min, ra_min], [dec_max, ra_max]] in radians
        """
        sources_in_map = self.cat.get_box(
            ra_min=box[0][1].to(u.deg).value,
            ra_max=box[1][1].to(u.deg).value,
            dec_min=box[0][0].to(u.deg).value,
            dec_max=box[1][0].to(u.deg).value,
        )
        sources = []
        for s in sources_in_map:
            sources.append(
                RegisteredSource(
                    ra=s.ra * u.deg,
                    dec=s.dec * u.deg,
                    source_id=str(s.id),
                    crossmatches=[
                        CrossMatch(
                            source_id=str(s.name),
                            probability=1.0,
                            distance=0.0 * u.deg,
                            frequency=90 * u.GHz,
                            catalog_name="mock",
                            catalog_idx=s.id,
                        )
                    ],
                )
            )
        self.log.info(
            "mock_database.get_sources_in_box",
            box=box,
            n_sources=len(sources),
        )
        return sources

    def get_sources_in_map(self, input_map: ProcessableMap) -> list[RegisteredSource]:
        map_bounds = input_map.flux.box() * u.radian
        self.log.info(
            "mock_database.get_sources_in_map",
            map_bounds_deg=map_bounds.to(u.deg),
        )
        # if ra_min is greater than ra_max, we are crossing the 0 point
        # so we need to split the box into two boxes.
        # since pixell uses -180 to 180 convention, we need to go 0 to ra_min then ra_max to 0
        if map_bounds[0][1] > map_bounds[1][1]:
            box1 = [[map_bounds[0][0], 0 * u.rad], [map_bounds[1][0], map_bounds[0][1]]]
            box2 = [[map_bounds[0][0], map_bounds[1][1]], [map_bounds[1][0], 0 * u.rad]]
            sources1 = self.get_sources_in_box(box=box1)
            sources2 = self.get_sources_in_box(box=box2)
            return sources1 + sources2
        else:
            return self.get_sources_in_box(box=map_bounds)

    def get_all_sources(self) -> list[RegisteredSource]:
        sources = self.get_sources_in_box(
            box=[[-90 * u.deg, -180 * u.deg], [90 * u.deg, 180 * u.deg]]
        )
        return sources


class MockSourceCatalog:
    def __init__(
        self,
        db_path: Path,
        log: FilteringBoundLogger | None = None,
        catalog_list: list = [],
        flux_lower_limit: u.Quantity = 0.01 * u.Jy,
    ):
        self.log = log or structlog.get_logger()
        hdu = fits.open(str(db_path))
        mock_cat = hdu[1]
        cat = MockDatabase()
        self.log.info("mock_act_database.loading_act_catalog")
        catalog_list = []
        for i in range(len(mock_cat.data["raDeg"])):  # This could be a zip I guess
            if mock_cat.data["fluxJy"][i] * u.Jy < flux_lower_limit:
                continue
            ra, dec = mock_cat.data["raDeg"][i], mock_cat.data["decDeg"][i]
            if ra > 180:
                ra -= 360  # Convention difference
            name = mock_cat.data["name"][i]
            registered_source = RegisteredSource(
                ra=ra * u.deg,
                dec=dec * u.deg,
                source_id="%s" % str(i).zfill(len(str(len(mock_cat.data["raDeg"])))),
                crossmatches=[
                    CrossMatch(
                        source_id=name,
                        probability=1.0,
                        distance=0.0 * u.deg,
                        frequency=90 * u.GHz,
                        catalog_name="ACT",
                        catalog_idx=i,
                    )
                ],
            )
            catalog_list.append(registered_source)
            cat.add_source(ra=ra * u.deg, dec=dec * u.deg, name=name)
        self.cat = cat
        self.log.info(
            "mock_act_database.loaded_act_catalog",
            n_sources=len(catalog_list),
        )
        self.catalog_list = catalog_list

    def update_catalog(self, new_sources: list, match_radius=0.1):
        """Update the catalog with new sources."""
        for source in new_sources:
            matches = self.cat.get_nearby_source(
                source.ra, source.dec, radius=match_radius
            )
            if not matches:
                s = self.cat.add_source(
                    ra=source.ra, dec=source.dec, name=source.source_id
                )
                self.catalog_list.append(s)
                self.log.info(
                    "mock_act_database.update_catalog.new_source_added", source=s
                )
            else:
                self.log.info(
                    "mock_act_database.update_catalog.existing_source_found",
                    source=source,
                    matching_sources=matches,
                )
                pass


class SourceCatalogDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.file_lock = FileLock(
            f"{db_path}.lock", timeout=1
        )  # Create a file-based lock
        self.initialize_database()

    def initialize_database(self):
        self.df = pd.DataFrame(
            columns=[
                "source_type",
                "ra",
                "dec",
                "err_ra",
                "err_dec",
                "flux",
                "err_flux",
                "frequency",
                "ctime",
                "arr",
                "source_id",
                "catalog_crossmatch",
                "crossmatch_names",
                "crossmatch_probabilities",
                "fit_type",
                "fwhm_ra",
                "fwhm_dec",
                "err_fwhm_ra",
                "err_fwhm_dec",
                "orientation",
            ]
        )

    @property
    def source_list(self):
        return self.df.iterrows()

    def read_database(self):
        with self.file_lock:
            # Check if the lock is stale before acquiring it
            if self._is_lock_stale():
                self.file_lock.release()
            try:
                self.df = pd.read_csv(self.db_path)
            except FileNotFoundError:
                self.initialize_database()

    def add_sources(self, sources: list):
        """Add a list of MeasuredSource objects to the database."""
        for source in sources:
            new_entry = pd.DataFrame(
                [
                    {
                        "source_type": source.source_type,
                        "ra": source.ra,
                        "dec": source.dec,
                        "err_ra": source.err_ra,
                        "err_dec": source.err_dec,
                        "flux": source.flux,
                        "err_flux": source.err_flux,
                        "frequency": source.frequency,
                        "ctime": source.ctime,
                        "arr": source.arr,
                        "source_id": source.source_id,
                        "catalog_crossmatch": source.catalog_crossmatch,
                        "crossmatch_names": source.crossmatch_names,
                        "crossmatch_probabilities": source.crossmatch_probabilities,
                        "fit_type": source.fit_type,
                        "fwhm_ra": source.fwhm_ra,
                        "fwhm_dec": source.fwhm_dec,
                        "err_fwhm_ra": source.err_fwhm_ra,
                        "err_fwhm_dec": source.err_fwhm_dec,
                        "orientation": source.orientation,
                    }
                ]
            )
            self.df = pd.concat([self.df, new_entry], ignore_index=True)

    def write_database(self):
        """Write the DataFrame to a pickle file."""
        # Check if the lock is stale before acquiring it
        if self._is_lock_stale():
            self.file_lock.release()  # Release the stale lock
        with self.file_lock:
            file_exists = os.path.exists(self.db_path)
            self.df.to_csv(self.db_path, mode="a", header=not file_exists, index=False)

    def _is_lock_stale(self):
        """Check if the lock file is stale."""
        lock_file = self.file_lock.lock_file
        if os.path.exists(lock_file):
            # Check if the process that created the lock is still running
            try:
                with open(lock_file, "r") as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)  # Check if the process is alive
            except (ValueError, ProcessLookupError):
                return True  # Lock is stale
        return False

    def observation_exists(self, map_id):
        return not self.df[self.df["map_id"] == map_id].empty

    def ingest_json(self, json_data, overwrite=False):
        import json

        data = json.loads(json_data)
        for entry in data:
            source = MeasuredSource(
                ra=entry["ra"],
                dec=entry["dec"],
                err_ra=entry.get("err_ra"),
                err_dec=entry.get("err_dec"),
                flux=entry["flux"],
                err_flux=entry.get("err_flux"),
                frequency=entry.get("frequency"),
                ctime=entry.get("ctime"),
                arr=entry.get("arr"),
                source_id=entry.get("source_id"),
                catalog_crossmatch=entry.get("catalog_crossmatch", False),
                crossmatch_names=entry.get("crossmatch_names", []),
                crossmatch_probabilities=entry.get("crossmatch_probabilities", []),
                fit_type=entry.get("fit_type", "forced"),
                fwhm_ra=entry.get("fwhm_ra"),
                fwhm_dec=entry.get("fwhm_dec"),
                err_fwhm_ra=entry.get("err_fwhm_ra"),
                err_fwhm_dec=entry.get("err_fwhm_dec"),
                orientation=entry.get("orientation"),
            )
            self.add_source(source, overwrite)
