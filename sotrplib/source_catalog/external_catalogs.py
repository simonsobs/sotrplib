"""
External astronomical catalogs used to find counterparts to transient
candidates: objects detected in the blind search which do not appear in
the SO source catalog.

Each catalog performs cone searches around a candidate position and
returns `CrossMatch` objects with a chance-coincidence probability

    P_chance = 1 - exp(-pi * rho * theta^2)

where theta is the angular separation and rho is the local surface
density of catalog objects at least as bright as the matched object.
A bright, rare object (e.g. a nearby galaxy) close to the candidate
gives a low P_chance; a faint star in a dense field gives a high one.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord
from structlog.types import FilteringBoundLogger

from sotrplib.sources.sources import CrossMatch


def chance_coincidence_probability(
    separation: u.Quantity,
    density: u.Quantity,
) -> float:
    """
    Probability of at least one unrelated catalog object within `separation`,
    given a local surface density `density` of objects at least as bright as
    the match (Poisson chance-coincidence).
    """
    expectation = (np.pi * density * separation**2).to_value(u.dimensionless_unscaled)
    return float(1.0 - np.exp(-expectation))


class ExternalCatalog(ABC):
    """
    An external (non-SO) catalog which can be cone-searched for counterparts.
    """

    catalog_name: str

    @abstractmethod
    def cone_search(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        radius: u.Quantity,
    ) -> list[CrossMatch]:
        """
        Return catalog objects within `radius` of (ra, dec) as CrossMatch
        objects with `chance_probability` filled in.
        """
        return


class TableExternalCatalog(ExternalCatalog):
    """
    An in-memory external catalog backed by coordinate/magnitude arrays,
    typically loaded from a local file so it works on compute nodes with
    no internet access.

    The local density used for the chance-coincidence probability is
    estimated by counting catalog objects at least as bright as the match
    within `density_radius` of the candidate position.
    """

    def __init__(
        self,
        catalog_name: str,
        coords: SkyCoord,
        names: list[str],
        source_type: str | list[str] = "unknown",
        mags: np.ndarray | None = None,
        redshifts: np.ndarray | None = None,
        distances: u.Quantity | None = None,
        density_radius: u.Quantity = 1.0 * u.deg,
        log: FilteringBoundLogger | None = None,
    ):
        self.catalog_name = catalog_name
        self.coords = coords
        self.names = list(names)
        if isinstance(source_type, str):
            self.source_types = [source_type] * len(self.names)
        else:
            self.source_types = list(source_type)
        self.mags = np.asarray(mags, dtype=float) if mags is not None else None
        self.redshifts = (
            np.asarray(redshifts, dtype=float) if redshifts is not None else None
        )
        self.distances = distances
        self.density_radius = density_radius
        self.log = log or structlog.get_logger()
        self.log.info(
            "external_catalog.loaded",
            catalog_name=catalog_name,
            n_objects=len(self.names),
        )

    def __len__(self):
        return len(self.names)

    def cone_search(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        radius: u.Quantity,
    ) -> list[CrossMatch]:
        if len(self) == 0:
            return []
        target = SkyCoord(ra=ra, dec=dec)
        separations = self.coords.separation(target)
        in_density_field = separations < self.density_radius
        n_field = int(np.sum(in_density_field))
        field_area = np.pi * self.density_radius**2

        match_indices = np.where(separations < radius)[0]
        matches = []
        for i in match_indices:
            mag = (
                float(self.mags[i])
                if self.mags is not None and np.isfinite(self.mags[i])
                else None
            )
            if mag is not None:
                brighter = in_density_field & (
                    np.nan_to_num(self.mags, nan=np.inf) <= mag
                )
                n_brighter = int(np.sum(brighter))
            else:
                n_brighter = n_field
            density = max(n_brighter, 1) / field_area
            matches.append(
                CrossMatch(
                    ra=self.coords[i].ra,
                    dec=self.coords[i].dec,
                    source_id=self.names[i],
                    source_type=self.source_types[i],
                    angular_separation=separations[i].to(u.deg),
                    mag=mag,
                    redshift=(
                        float(self.redshifts[i])
                        if self.redshifts is not None and np.isfinite(self.redshifts[i])
                        else None
                    ),
                    distance=(
                        self.distances[i] if self.distances is not None else None
                    ),
                    chance_probability=chance_coincidence_probability(
                        separations[i], density
                    ),
                    catalog_name=self.catalog_name,
                    catalog_idx=int(i),
                )
            )
        return matches

    @classmethod
    def from_table_file(
        cls,
        path: str | Path,
        ra_column: str,
        dec_column: str,
        name_column: str,
        coord_unit: tuple[str, str] = ("deg", "deg"),
        mag_column: str | None = None,
        redshift_column: str | None = None,
        distance_column: str | None = None,
        distance_unit: str = "Mpc",
        source_type: str = "unknown",
        catalog_name: str | None = None,
        density_radius: u.Quantity = 1.0 * u.deg,
        log: FilteringBoundLogger | None = None,
    ) -> "TableExternalCatalog":
        """
        Load a generic external catalog from any table format astropy can read
        (FITS, CSV, ...). Intended for e.g. nearby galaxy catalogs
        (GLADE+, HECATE, 2MRS) where the user supplies the column mapping.
        """
        from astropy.table import Table

        table = Table.read(path)
        coords = SkyCoord(
            ra=np.asarray(table[ra_column], dtype=float),
            dec=np.asarray(table[dec_column], dtype=float),
            unit=coord_unit,
        )

        def _float_column(column):
            if column is None:
                return None
            values = np.ma.filled(
                np.ma.masked_invalid(np.asarray(table[column], dtype=float)), np.nan
            )
            return values

        distances = _float_column(distance_column)
        return cls(
            catalog_name=catalog_name or Path(path).stem,
            coords=coords,
            names=[str(n) for n in table[name_column]],
            source_type=source_type,
            mags=_float_column(mag_column),
            redshifts=_float_column(redshift_column),
            distances=distances * u.Unit(distance_unit)
            if distances is not None
            else None,
            density_radius=density_radius,
            log=log,
        )

    @classmethod
    def from_millionquasar_file(
        cls,
        path: str | Path,
        density_radius: u.Quantity = 1.0 * u.deg,
        log: FilteringBoundLogger | None = None,
    ) -> "TableExternalCatalog":
        """
        Load the Million Quasars (Milliquas) catalog, either as a FITS table
        (with RA/DEC in degrees) or as the pipe-delimited HEASARC text export
        (with sexagesimal coordinates).
        """
        path = Path(path)
        if path.suffix.lower() in (".fits", ".fit", ".fz"):
            names, coords, types, mags, redshifts = _read_milliquas_fits(path)
        else:
            names, coords, types, mags, redshifts = _read_milliquas_ascii(path)
        return cls(
            catalog_name="million_quasar",
            coords=coords,
            names=names,
            source_type=[_milliquas_broad_type(t) for t in types],
            mags=mags,
            redshifts=redshifts,
            density_radius=density_radius,
            log=log,
        )


def _milliquas_broad_type(broad_type: str) -> str:
    """
    Map a Milliquas broad_type code to a coarse object class.
    Leading letter: Q/B (QSO, BL Lac) -> quasar; A/K/N (AGN, narrow-line) -> agn.
    """
    broad_type = (broad_type or "").strip().upper()
    if broad_type[:1] in ("A", "K", "N"):
        return "agn"
    return "quasar"


def _read_milliquas_fits(path: Path):
    from astropy.table import Table

    table = Table.read(path)
    columns = {name.lower(): name for name in table.colnames}

    def _get(*options):
        for option in options:
            if option in columns:
                return table[columns[option]]
        return None

    ra = np.asarray(_get("ra", "radeg"), dtype=float)
    dec = np.asarray(_get("dec", "decdeg"), dtype=float)
    coords = SkyCoord(ra=ra, dec=dec, unit="deg")
    names = [str(n) for n in _get("name", "id")]
    type_column = _get("broad_type", "type")
    types = (
        [str(t) for t in type_column] if type_column is not None else [""] * len(names)
    )
    mag_column = _get("rmag", "bmag", "mag")
    mags = (
        np.ma.filled(np.ma.masked_invalid(np.asarray(mag_column, dtype=float)), np.nan)
        if mag_column is not None
        else None
    )
    z_column = _get("z", "redshift")
    redshifts = (
        np.ma.filled(np.ma.masked_invalid(np.asarray(z_column, dtype=float)), np.nan)
        if z_column is not None
        else None
    )
    return names, coords, types, mags, redshifts


def _read_milliquas_ascii(path: Path):
    """
    Parse the pipe-delimited HEASARC milliquas export. The first '|' line is
    the header; coordinates are sexagesimal (RA in hours, Dec in degrees).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("|"):
                rows.append([part.strip() for part in line.strip("|").split("|")])
    if not rows:
        raise ValueError(f"No pipe-delimited rows found in {path}")
    header = [h.lower() for h in rows[0]]
    column = {name: i for i, name in enumerate(header)}

    def _field(row, name):
        idx = column.get(name)
        return row[idx] if idx is not None and idx < len(row) else ""

    def _mag(row):
        for key in ("rmag", "bmag"):
            try:
                return float(_field(row, key))
            except ValueError:
                continue
        return np.nan

    def _redshift(row):
        try:
            return float(_field(row, "redshift"))
        except ValueError:
            return np.nan

    data = rows[1:]
    names = [_field(row, "name") for row in data]
    coords = SkyCoord(
        ra=[_field(row, "ra") for row in data],
        dec=[_field(row, "dec") for row in data],
        unit=(u.hourangle, u.deg),
    )
    types = [_field(row, "broad_type") for row in data]
    mags = np.asarray([_mag(row) for row in data])
    redshifts = np.asarray([_redshift(row) for row in data])
    return names, coords, types, mags, redshifts


class GaiaCatalog(ExternalCatalog):
    """
    Online Gaia cone-search adapter using astroquery. Requires internet
    access; on failure (e.g. running on a compute node) it logs a warning
    and returns no matches rather than raising.

    The local density is estimated from the query itself: objects at least
    as bright as the match within `density_radius`. If the row limit
    truncates the query the density (and so P_chance) is underestimated.
    """

    catalog_name = "gaia_dr3"

    def __init__(
        self,
        density_radius: u.Quantity = 5.0 * u.arcmin,
        max_mag: float | None = None,
        row_limit: int = 2000,
        log: FilteringBoundLogger | None = None,
    ):
        self.density_radius = density_radius
        self.max_mag = max_mag
        self.row_limit = row_limit
        self.log = log or structlog.get_logger()

    def cone_search(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        radius: u.Quantity,
    ) -> list[CrossMatch]:
        query_radius = max(radius, self.density_radius)
        try:
            from astroquery.gaia import Gaia

            Gaia.ROW_LIMIT = self.row_limit
            results = Gaia.cone_search(
                SkyCoord(ra=ra, dec=dec, frame="icrs"), radius=query_radius
            ).get_results()
        except Exception as e:
            self.log.warning("gaia_catalog.query_failed", error=str(e))
            return []
        if results is None or len(results) == 0:
            return []

        mags = np.ma.filled(
            np.ma.masked_invalid(np.asarray(results["phot_g_mean_mag"], dtype=float)),
            np.nan,
        )
        if self.max_mag is not None:
            keep = np.nan_to_num(mags, nan=np.inf) <= self.max_mag
            results = results[keep]
            mags = mags[keep]
            if len(results) == 0:
                return []

        coords = SkyCoord(
            ra=np.asarray(results["ra"], dtype=float),
            dec=np.asarray(results["dec"], dtype=float),
            unit="deg",
        )
        separations = coords.separation(SkyCoord(ra=ra, dec=dec))
        parallaxes = np.ma.filled(
            np.ma.masked_invalid(np.asarray(results["parallax"], dtype=float)),
            np.nan,
        )
        field_area = np.pi * query_radius**2

        matches = []
        for i in np.where(separations < radius)[0]:
            if np.isfinite(mags[i]):
                n_brighter = int(np.sum(np.nan_to_num(mags, nan=np.inf) <= mags[i]))
            else:
                n_brighter = len(mags)
            density = max(n_brighter, 1) / field_area
            matches.append(
                CrossMatch(
                    ra=coords[i].ra,
                    dec=coords[i].dec,
                    source_id=str(results["designation"][i]),
                    # objects with a measured parallax are galactic stars;
                    # the rest could be anything (quasars, galaxies, ...)
                    source_type="star" if np.isfinite(parallaxes[i]) else "unknown",
                    angular_separation=separations[i].to(u.deg),
                    mag=float(mags[i]) if np.isfinite(mags[i]) else None,
                    chance_probability=chance_coincidence_probability(
                        separations[i], density
                    ),
                    catalog_name=self.catalog_name,
                    catalog_idx=int(i),
                )
            )
        return matches
