"""
Tests for external-catalog crossmatching and prioritization of
transient candidates.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from sotrplib.sifter.transient_crossmatch import (
    DEFAULT_TYPE_PRIORS,
    TransientCrossmatcher,
    rank_crossmatches,
)
from sotrplib.source_catalog.external_catalogs import (
    TableExternalCatalog,
    chance_coincidence_probability,
)
from sotrplib.sources.sources import CrossMatch, MeasuredSource

CANDIDATE_RA = 150.0 * u.deg
CANDIDATE_DEC = 2.0 * u.deg


@pytest.fixture
def candidate():
    return MeasuredSource(
        ra=CANDIDATE_RA,
        dec=CANDIDATE_DEC,
        flux=100.0 * u.mJy,
        measurement_type="blind",
    )


def make_catalog(
    name,
    offsets_arcmin,
    mags,
    source_type,
    n_background=0,
    background_mag=20.0,
):
    """
    Build a catalog with objects offset from the candidate position by
    `offsets_arcmin` (in dec), plus `n_background` objects of magnitude
    `background_mag` scattered in an annulus 0.1-0.5 deg from the
    candidate (inside the 1 deg density radius, outside any match radius).
    """
    decs = [CANDIDATE_DEC.value + off / 60.0 for off in offsets_arcmin]
    ras = [CANDIDATE_RA.value] * len(offsets_arcmin)
    all_mags = list(mags)
    rng = np.random.default_rng(42)
    for _ in range(n_background):
        r = rng.uniform(0.1, 0.5)
        theta = rng.uniform(0, 2 * np.pi)
        ras.append(CANDIDATE_RA.value + r * np.cos(theta))
        decs.append(CANDIDATE_DEC.value + r * np.sin(theta))
        all_mags.append(background_mag)
    return TableExternalCatalog(
        catalog_name=name,
        coords=SkyCoord(ra=ras, dec=decs, unit="deg"),
        names=[f"{name}_{i}" for i in range(len(ras))],
        source_type=source_type,
        mags=np.asarray(all_mags),
        density_radius=1.0 * u.deg,
    )


def test_chance_coincidence_probability_behavior():
    density = 100.0 / (np.pi * u.deg**2)
    p_close = chance_coincidence_probability(0.1 * u.arcmin, density)
    p_far = chance_coincidence_probability(2.0 * u.arcmin, density)
    assert 0.0 <= p_close < p_far <= 1.0
    assert chance_coincidence_probability(0.0 * u.deg, density) == 0.0
    # denser fields give higher chance probabilities
    p_dense = chance_coincidence_probability(1.0 * u.arcmin, 100 * density)
    p_sparse = chance_coincidence_probability(1.0 * u.arcmin, density)
    assert p_dense > p_sparse


def test_cone_search_density_uses_brightness():
    # one bright object and many faint ones at the same separation:
    # the bright match must have a lower chance probability.
    catalog = make_catalog(
        "test",
        offsets_arcmin=[0.5, 0.6],
        mags=[10.0, 20.0],
        source_type="star",
        n_background=200,
        background_mag=20.0,
    )
    matches = catalog.cone_search(CANDIDATE_RA, CANDIDATE_DEC, 2.0 * u.arcmin)
    assert len(matches) == 2
    by_id = {m.source_id: m for m in matches}
    assert by_id["test_0"].chance_probability < by_id["test_1"].chance_probability


def test_bright_galaxy_outranks_quasar_and_dim_star(candidate):
    galaxy_catalog = make_catalog(
        "galaxies", offsets_arcmin=[0.8], mags=[11.0], source_type="galaxy"
    )
    quasar_catalog = make_catalog(
        "quasars",
        offsets_arcmin=[0.5],
        mags=[19.0],
        source_type="quasar",
        n_background=50,
    )
    star_catalog = make_catalog(
        "stars",
        offsets_arcmin=[0.3],
        mags=[19.5],
        source_type="star",
        n_background=500,
    )
    crossmatcher = TransientCrossmatcher(
        catalogs=[star_catalog, quasar_catalog, galaxy_catalog],
        match_radius=2.0 * u.arcmin,
    )
    (matched,) = crossmatcher.crossmatch([candidate])
    assert len(matched.crossmatches) == 3
    best = matched.crossmatches[0]
    assert best.source_type == "galaxy"
    assert best.catalog_name == "galaxies"
    # ranked descending by score, with probabilities filled in
    scores = [m.match_score for m in matched.crossmatches]
    assert scores == sorted(scores, reverse=True)
    assert all(m.probability is not None for m in matched.crossmatches)
    # candidate classified from the best match
    assert matched.source_type == "extragalactic"


def test_no_match_leaves_candidate_untouched(candidate):
    catalog = make_catalog(
        "galaxies", offsets_arcmin=[30.0], mags=[11.0], source_type="galaxy"
    )
    crossmatcher = TransientCrossmatcher(
        catalogs=[catalog], match_radius=2.0 * u.arcmin
    )
    (matched,) = crossmatcher.crossmatch([candidate])
    assert matched.crossmatches == []
    assert matched.source_type is None


def test_min_score_filters_weak_matches(candidate):
    star_catalog = make_catalog(
        "stars",
        offsets_arcmin=[1.9],
        mags=[20.0],
        source_type="star",
        n_background=5000,
    )
    crossmatcher = TransientCrossmatcher(
        catalogs=[star_catalog],
        match_radius=2.0 * u.arcmin,
        min_score=0.5,
    )
    (matched,) = crossmatcher.crossmatch([candidate])
    assert matched.crossmatches == []


def test_match_radius_scales_with_position_error(candidate):
    catalog = make_catalog(
        "galaxies", offsets_arcmin=[4.0], mags=[11.0], source_type="galaxy"
    )
    crossmatcher = TransientCrossmatcher(
        catalogs=[catalog], match_radius=2.0 * u.arcmin, position_error_scale=3.0
    )
    # default radius misses the 4 arcmin offset object
    assert crossmatcher.crossmatch_candidate(candidate) == []
    # a large positional uncertainty widens the search
    candidate.err_ra = 2.0 * u.arcmin
    candidate.err_dec = 2.0 * u.arcmin
    assert len(crossmatcher.crossmatch_candidate(candidate)) == 1


def test_rank_crossmatches_priors():
    matches = [
        CrossMatch(
            source_id="quasar",
            source_type="quasar",
            chance_probability=0.1,
        ),
        CrossMatch(
            source_id="galaxy",
            source_type="galaxy",
            chance_probability=0.1,
        ),
    ]
    ranked = rank_crossmatches(matches)
    # equal chance probability: the type prior breaks the tie
    assert ranked[0].source_id == "galaxy"
    assert ranked[0].match_score == pytest.approx(DEFAULT_TYPE_PRIORS["galaxy"] * 0.9)
    assert ranked[1].match_score == pytest.approx(DEFAULT_TYPE_PRIORS["quasar"] * 0.9)


MILLIQUAS_SAMPLE = """Results from heasarc_milliquas: Million Quasars Catalog (MILLIQUAS), Version 8 (2 August 2023)
Coordinate system: Equatorial
|name                     |ra         |dec        |broad_type|rmag |bmag |redshift|radio_name            |xray_name             |
|PGC 275952               |03 15 34.27|-69 59 56.3|NR        |12.89|13.56|   0.059|RACD J031534.4-695958 |                      |
|MQS J054221.26-695953.9  |05 42 21.28|-69 59 53.8|QX        |18.07|17.46|   0.831|                      |CXOG J054221.1-695953 |
"""


def test_milliquas_ascii_parsing(tmp_path):
    path = tmp_path / "milliquas_sample.txt"
    path.write_text(MILLIQUAS_SAMPLE)
    catalog = TableExternalCatalog.from_millionquasar_file(path)
    assert len(catalog) == 2
    assert catalog.names[0] == "PGC 275952"
    # narrow-line AGN code maps to agn, QSO code to quasar
    assert catalog.source_types == ["agn", "quasar"]
    assert catalog.mags[0] == pytest.approx(12.89)
    assert catalog.redshifts[1] == pytest.approx(0.831)
    # sexagesimal coordinates parsed as (hourangle, deg)
    expected = SkyCoord(ra="05 42 21.28", dec="-69 59 53.8", unit=(u.hourangle, u.deg))
    match = catalog.cone_search(expected.ra, expected.dec, 0.1 * u.arcmin)
    assert len(match) == 1
    assert match[0].source_id == "MQS J054221.26-695953.9"
    assert match[0].source_type == "quasar"


def test_sifter_integration(candidate):
    from sotrplib.sifter.core import SimpleCatalogSifter
    from sotrplib.source_catalog.core import RegisteredSourceCatalog

    galaxy_catalog = make_catalog(
        "galaxies", offsets_arcmin=[0.8], mags=[11.0], source_type="galaxy"
    )
    sifter = SimpleCatalogSifter(
        radius=1.0 * u.arcmin,
        transient_crossmatcher=TransientCrossmatcher(
            catalogs=[galaxy_catalog], match_radius=2.0 * u.arcmin
        ),
    )
    result = sifter.sift(
        sources=[candidate],
        catalogs=[RegisteredSourceCatalog(sources=[])],
        input_map=None,
    )
    # not in the (empty) socat -> transient candidate with a ranked match
    assert len(result.transient_candidates) == 1
    assert result.source_candidates == []
    transient = result.transient_candidates[0]
    assert transient.crossmatches[0].source_type == "galaxy"
    assert transient.source_type == "extragalactic"
