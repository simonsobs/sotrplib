import glob
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyinstrument
import pytest
import structlog
from astropy import units as u
from astropy.coordinates import SkyCoord

from sotrplib.config.config import Settings
from sotrplib.sifter.map_matching import EmptyMapMatcher
from sotrplib.sources.sources import MeasuredSource


# a pytest fixture providing "result" that is parameterized over all
# pickle or json files in the current directory
@pytest.fixture(params=glob.glob("./*.pickle"))
def result(request):
    filename = request.param
    with open(filename, "rb") as handle:
        result = pickle.load(handle)
    return result


def test_structure(result):
    assert "forced_photometry" in result
    assert "sifted_blind_search" in result


def test_number_with_photometry(result):
    photometery = result["forced_photometry"]
    assert len(photometery) > 2000


def test_number_found(result):
    """
    Test that our efficiency of finding candidates is high and that
    we don't spuriously find noise or transient candidates.
    """
    blind_results = result["sifted_blind_search"]
    assert len(blind_results.source_candidates) > 2000
    assert len(blind_results.noise_candidates) == 0
    assert len(blind_results.transient_candidates) == 0


def calculate_separation(candidate: MeasuredSource) -> float:
    ra = candidate.ra
    dec = candidate.dec
    ra_offset = candidate.fit_params["ra_offset"]
    dec_offset = candidate.fit_params["dec_offset"]
    this = SkyCoord(ra=ra, dec=dec)
    other = SkyCoord(ra=ra + ra_offset, dec=dec + dec_offset)
    return this.separation(other).deg


def main():
    config = Settings.from_file("regression.json")
    pipeline = config.to_runner()
    # This script reads the sources from the return value of the runner.
    pipeline.map_matcher = EmptyMapMatcher(hold_all_maps=True)

    # Pointing fits measure the position of all bright sources. Forced fits do
    # this only for significant sources. The runner does not return pointing fits.
    pointing_fits = []
    pointing_force = pipeline.pointing_provider.force

    def record_pointing_fits(*args, **kwargs):
        fits = pointing_force(*args, **kwargs)
        pointing_fits.extend(fits)
        return fits

    pipeline.pointing_provider.force = record_pointing_fits

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(config.log_level),
    )
    log = structlog.get_logger()

    maps = config.maps
    if hasattr(maps, "to_generator"):
        maps = maps.to_generator(log=log)
    else:
        maps = [mm.to_map(log=log) for mm in maps]

    with pyinstrument.Profiler() as profiler:
        results = pipeline.run(maps)
    profiler.write_html("profile.html")
    print(profiler.output_text(unicode=True, color=True))
    result = results[0]
    photometry, sifter_result = result

    found = [candidate for candidate in photometry if not candidate.fit_failed]
    centroided = [candidate for candidate in pointing_fits if not candidate.fit_failed]
    assert len(centroided) > 0

    # simulated catalogs have a random number added to the id
    # assume this corresponds to the first element
    cat = pipeline.source_catalogs[0]
    flux_ratios = [
        candidate.flux / cat.source_by_id(candidate.source_id).flux
        for candidate in found
    ]
    bright_flux_ratios = [
        candidate.flux / cat.source_by_id(candidate.source_id).flux
        for candidate in found
        if cat.source_by_id(candidate.source_id).flux > 0.1 * u.Jy
    ]

    dec_offsets = np.array(
        [candidate.fit_params["dec_offset"].value for candidate in centroided]
    )
    ra_offsets = np.array(
        [candidate.fit_params["ra_offset"].value for candidate in centroided]
    )
    offsets = np.array([calculate_separation(candidate) for candidate in centroided])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    plt.sca(axes[0])
    plt.ecdf(np.abs(dec_offsets), label="Dec")
    plt.ecdf(np.abs(ra_offsets), label="RA")
    plt.ecdf(np.abs(offsets), label="Total")
    plt.xscale("log")
    plt.xlabel("Offset [deg]")
    plt.ylabel("Cumulative Fraction")
    plt.legend(loc="lower right")

    plt.sca(axes[1])
    plt.ecdf(flux_ratios, label="Flux Ratio")
    plt.xscale("log")
    plt.xlabel("Measured / True Flux")

    plt.tight_layout()
    plt.savefig("offset_ecdf.png")
    plt.close(fig)

    summary = dict(
        duration=profiler.last_session.duration,
        number_with_forced_photometry=len(photometry),
        number_with_pointing_fit=len(centroided),
        number_forced_with_free_fit=sum(
            not candidate.fit_params["forced_center"] for candidate in found
        ),
        median_flux_ratio_above_100mJy=float(np.median(bright_flux_ratios)),
        number_blind_sources=len(sifter_result.source_candidates),
        number_noise_sources=len(sifter_result.noise_candidates),
        number_transient_sources=len(sifter_result.transient_candidates),
        mean_offset_deg=float(np.mean(offsets)),
        median_offset_deg=float(np.median(offsets)),
        std_offset_deg=float(np.std(offsets)),
        mean_ra_offset_deg=float(np.mean(ra_offsets)),
        median_ra_offset_deg=float(np.median(ra_offsets)),
        std_ra_offset_deg=float(np.std(ra_offsets)),
        mean_dec_offset_deg=float(np.mean(dec_offsets)),
        median_dec_offset_deg=float(np.median(dec_offsets)),
        std_dec_offset_deg=float(np.std(dec_offsets)),
    )
    with open("summary.json", "w") as handle:
        json.dump(summary, handle, indent=4)


if __name__ == "__main__":
    main()
