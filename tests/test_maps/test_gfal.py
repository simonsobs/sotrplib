"""
Tests sotrplib.maps.gfal: the grid-door fetch-to-scratch helper used by
SPT map readers.
"""

from unittest.mock import patch

import pytest

from sotrplib.maps import gfal

# ─── is_grid_url / grid_url ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "davs://osg-gridftp.grid.uchicago.edu:2880/sptgrid/data/x.g3.gz",
        "gsiftp://example.org/x.g3.gz",
        "root://example.org/x.g3.gz",
        "srm://example.org/x.g3.gz",
    ],
)
def test_is_grid_url_true(url):
    assert gfal.is_grid_url(url)


@pytest.mark.parametrize("path", ["/sptgrid/data/x.g3.gz", "x.g3.gz", "./x.g3.gz"])
def test_is_grid_url_false(path):
    assert not gfal.is_grid_url(path)


def test_grid_url_joins_root_and_relative_path():
    url = gfal.grid_url(gfal.GRID_DOOR + "/sptgrid/data/onlinemaps/ra0hdec-52.25/", "x.g3.gz")
    assert url == gfal.GRID_DOOR + "/sptgrid/data/onlinemaps/ra0hdec-52.25/x.g3.gz"


def test_grid_url_does_not_mangle_the_double_slash():
    # Path() would collapse "davs://host" to "davs:/host"; grid_url must not.
    url = gfal.grid_url("davs://host:2880/a", "b/c.g3.gz")
    assert url == "davs://host:2880/a/b/c.g3.gz"


# ─── gfal_copy ──────────────────────────────────────────────────────────────


def test_gfal_copy_raises_on_nonzero_exit(tmp_path):
    with patch("sotrplib.maps.gfal.subprocess.run") as run:
        run.return_value.returncode = 1
        run.return_value.stderr = "boom"
        with pytest.raises(RuntimeError, match="gfal-copy failed"):
            gfal.gfal_copy("davs://host/x.g3.gz", tmp_path / "x.g3.gz")


def test_gfal_copy_raises_on_timeout(tmp_path):
    import subprocess

    with patch("sotrplib.maps.gfal.subprocess.run") as run:
        run.side_effect = subprocess.TimeoutExpired(cmd="gfal-copy", timeout=1)
        with pytest.raises(RuntimeError, match="timed out"):
            gfal.gfal_copy("davs://host/x.g3.gz", tmp_path / "x.g3.gz", timeout=1)


def test_gfal_copy_invokes_gfal_copy_with_file_url(tmp_path):
    destination = tmp_path / "sub" / "x.g3.gz"
    with patch("sotrplib.maps.gfal.subprocess.run") as run:
        run.return_value.returncode = 0
        gfal.gfal_copy("davs://host/x.g3.gz", destination)
    args = run.call_args.args[0]
    assert args[0] == "gfal-copy"
    assert "davs://host/x.g3.gz" in args
    assert f"file://{destination}" in args
    # gfal_copy must make the destination's parent directory itself.
    assert destination.parent.exists()


# ─── fetch_to_scratch ───────────────────────────────────────────────────────


def test_fetch_to_scratch_yields_local_path_unchanged(tmp_path):
    local = tmp_path / "already_here.g3.gz"
    local.write_bytes(b"data")
    with gfal.fetch_to_scratch(str(local), tmp_path / "scratch") as path:
        assert path == local
        assert path.read_bytes() == b"data"
    # A local (non-grid) source is never copied or deleted.
    assert local.exists()


def test_fetch_to_scratch_copies_grid_url_and_deletes_on_exit(tmp_path):
    scratch_dir = tmp_path / "scratch"
    url = "davs://host/x.g3.gz"

    def fake_copy(source, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fetched")

    with patch("sotrplib.maps.gfal.gfal_copy", side_effect=fake_copy) as copy:
        with gfal.fetch_to_scratch(url, scratch_dir) as path:
            assert path == scratch_dir / "x.g3.gz"
            assert path.read_bytes() == b"fetched"
            copy.assert_called_once()
        # Deleted once the `with` block exits.
        assert not path.exists()


def test_fetch_to_scratch_keep_does_not_delete(tmp_path):
    scratch_dir = tmp_path / "scratch"
    url = "davs://host/x.g3.gz"

    def fake_copy(source, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fetched")

    with patch("sotrplib.maps.gfal.gfal_copy", side_effect=fake_copy):
        with gfal.fetch_to_scratch(url, scratch_dir, keep=True) as path:
            pass
        assert path.exists()


def test_fetch_to_scratch_reuses_existing_scratch_copy(tmp_path):
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()
    cached = scratch_dir / "x.g3.gz"
    cached.write_bytes(b"already cached")

    with patch("sotrplib.maps.gfal.gfal_copy") as copy:
        with gfal.fetch_to_scratch("davs://host/x.g3.gz", scratch_dir, keep=True) as path:
            assert path.read_bytes() == b"already cached"
        copy.assert_not_called()


def test_fetch_to_scratch_deletes_even_if_body_raises(tmp_path):
    scratch_dir = tmp_path / "scratch"

    def fake_copy(source, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"fetched")

    with patch("sotrplib.maps.gfal.gfal_copy", side_effect=fake_copy):
        with pytest.raises(ValueError):
            with gfal.fetch_to_scratch("davs://host/x.g3.gz", scratch_dir) as path:
                raise ValueError("boom")
    assert not path.exists()
