"""
A blind-search candidate near the true edge of the map array can make
reproject.thumbnails() extract an asymmetrically clipped stamp, which its
own apodization step then fails to broadcast against (a real crash seen
running the SPT3G pipeline at a low blind-search threshold, where far more
candidates sit close to the array boundary than at the usual 5-sigma
threshold). extract_thumbnail() already has this failure mode handled the
same way on the forced-photometry path; SigmaClipBlindSearch.search() must
not let one such candidate take down the whole blind search.
"""

from types import SimpleNamespace
from unittest.mock import patch

from astropy import units as u

from sotrplib.sources.blind import SigmaClipBlindSearch
from sotrplib.sources.sources import MeasuredSource


def _fake_source(source_id: str) -> MeasuredSource:
    return MeasuredSource(ra=10.0 * u.deg, dec=-5.0 * u.deg, source_id=source_id)


def _fake_map() -> SimpleNamespace:
    return SimpleNamespace(finalized=True, map_resolution=0.5 * u.arcmin)


def test_search_skips_candidate_whose_thumbnail_extraction_fails():
    good = _fake_source("good")
    bad = _fake_source("bad")

    # MeasuredSource is a pydantic model: bypass its __setattr__ (which
    # rejects non-field attributes) to stub this one instance's method.
    object.__setattr__(good, "extract_thumbnail", lambda *a, **k: None)
    object.__setattr__(bad, "extract_thumbnail", _raise_broadcast_error)

    searcher = SigmaClipBlindSearch(thumbnail_half_width=4 * u.arcmin)

    with patch("sotrplib.sources.blind.extract_sources", return_value=[good, bad]):
        sources, _ = searcher.search(input_map=_fake_map())

    # Both candidates survive the search -- the failed one stays, flagged,
    # instead of the whole blind search crashing on it.
    assert sources == [good, bad]
    assert good.fit_failed is False
    assert bad.fit_failed is True
    assert bad.fit_failure_reason == "extract_thumbnail_failed"


def test_search_succeeds_when_no_candidate_fails():
    good = _fake_source("good")
    object.__setattr__(good, "extract_thumbnail", lambda *a, **k: None)

    searcher = SigmaClipBlindSearch(thumbnail_half_width=4 * u.arcmin)

    with patch("sotrplib.sources.blind.extract_sources", return_value=[good]):
        sources, _ = searcher.search(input_map=_fake_map())

    assert sources == [good]
    assert good.fit_failed is False


def _raise_broadcast_error(*args, **kwargs):
    raise ValueError(
        "operands could not be broadcast together with shapes (100,3) (1,8) (100,3)"
    )
