"""
Read map files from SPT grid storage with a local scratch cache.

OSG worker nodes have no POSIX access to /sptgrid: they reach it only
through the GridFTP/WebDAV door, with gfal-copy. This module gives SPT3G
map readers one fetch path that works the same way for a remote OSG job
(no other choice) and for a job running where /sptgrid is NFS-mounted
(so that reads go through the grid door instead of the shared NFS link,
and large map files never sit on scratch disk longer than one map's
build() needs them).
"""

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import structlog
from structlog.types import FilteringBoundLogger

GRID_DOOR = "davs://osg-gridftp.grid.uchicago.edu:2880"

_GRID_SCHEMES = ("davs://", "gsiftp://", "root://", "srm://")


def is_grid_url(path: str) -> bool:
    """True if `path` is a grid URL (davs://, gsiftp://, ...), not a local path."""
    return str(path).startswith(_GRID_SCHEMES)


def grid_url(root: str, relative_path: str) -> str:
    """Join a grid door root and a map_path into one URL (plain string join:
    Path() must not touch a URL, since it collapses the "//" after the
    scheme)."""
    return f"{root.rstrip('/')}/{str(relative_path).lstrip('/')}"


def gfal_copy(
    source: str,
    destination: Path,
    timeout: float = 600.0,
    log: FilteringBoundLogger | None = None,
) -> None:
    """Copy `source` (a grid URL) to the local `destination` with gfal-copy."""
    log = log or structlog.get_logger()
    destination.parent.mkdir(parents=True, exist_ok=True)
    log.info("gfal.copy.start", source=source, destination=str(destination))
    try:
        result = subprocess.run(
            ["gfal-copy", "-f", source, f"file://{destination}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as err:
        raise RuntimeError(f"gfal-copy timed out after {timeout}s for {source}") from err
    if result.returncode != 0:
        raise RuntimeError(
            f"gfal-copy failed ({result.returncode}) for {source}: "
            f"{result.stderr.strip()}"
        )
    log.info("gfal.copy.done", source=source, destination=str(destination))


@contextmanager
def fetch_to_scratch(
    source: str,
    scratch_dir: Path,
    keep: bool = False,
    log: FilteringBoundLogger | None = None,
) -> Iterator[Path]:
    """
    Make `source` available as a local file for the body of a `with` block.

    If `source` is a grid URL, gfal-copy it into `scratch_dir` (named by its
    basename) and delete the local copy on exit, unless `keep` is set. If
    `source` is already a local path, yield it unchanged: nothing is copied
    or deleted. This keeps map-reading code the same whether the map comes
    from a grid door or is already staged locally (as in tests).
    """
    if not is_grid_url(source):
        yield Path(source)
        return

    local_path = Path(scratch_dir) / Path(source).name
    if not local_path.exists():
        gfal_copy(source, local_path, log=log)
    try:
        yield local_path
    finally:
        if not keep and local_path.exists():
            local_path.unlink()
