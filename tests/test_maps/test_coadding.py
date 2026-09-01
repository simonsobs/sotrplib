import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from pixell import enmap

from sotrplib.maps.core import IntensityAndInverseVarianceMap, RhoAndKappaMap
from sotrplib.maps.map_coadding import IntensityMapCoadder, RhoKappaMapCoadder
from sotrplib.sims import (
    source_injector,
)
from sotrplib.sims.sim_sources import FixedSimulatedSource
from sotrplib.source_catalog.core import RegisteredSource


def test_rhokappa_coadder_separate(separate_map_set_1, separate_map_set_2):
    """Test the RhoKappaMapCoadderConfig coadder."""
    map_path_1 = separate_map_set_1
    map_path_2 = separate_map_set_2

    start_time = Time("2025-10-10", format="iso")
    input_maps = [
        RhoAndKappaMap(
            rho_filename=map_path_1["rho"],
            kappa_filename=map_path_1["kappa"],
            time_filename=map_path_1["time"],
            frequency="f090",
            start_time=start_time,
            end_time=start_time + TimeDelta(3600, format="sec"),
        ),
        RhoAndKappaMap(
            rho_filename=map_path_2["rho"],
            kappa_filename=map_path_2["kappa"],
            time_filename=map_path_2["time"],
            frequency="f090",
            start_time=start_time + TimeDelta(3600, format="sec"),
            end_time=start_time + TimeDelta(7200, format="sec"),
        ),
    ]

    for i in range(len(input_maps)):
        input_maps[i].build()

    assert np.all(input_maps[0].rho != input_maps[1].rho)

    coadder = RhoKappaMapCoadder(frequencies=["f090"])
    coadded_maps = coadder.coadd(input_maps=input_maps)
    assert isinstance(coadded_maps, list)

    coadd = coadded_maps[0]
    coadd.finalize()
    # Check that the coadded map is a CoaddedRhoKappaMap

    assert coadd.flux is not None
    assert coadd.snr is not None
    assert coadd.time_mean is not None
    corners = (
        coadd.bbox
    )  # [[dec_min, ra_max], [dec_max, ra_min]] in radians (pixell east-left)
    b0, b1 = input_maps[0].bbox, input_maps[1].bbox
    assert corners[0][0] == pytest.approx(min(b0[0][0], b1[0][0]))  # dec_min
    assert corners[0][1] == pytest.approx(max(b0[0][1], b1[0][1]))  # ra_max (east edge)
    assert corners[1][0] == pytest.approx(max(b0[1][0], b1[1][0]))  # dec_max
    assert corners[1][1] == pytest.approx(min(b0[1][1], b1[1][1]))  # ra_min (west edge)
    assert np.all(coadd.hits <= 1)


def test_rhokappa_coadder_overlapping(overlapping_map_set_1, overlapping_map_set_2):
    """Test the RhoKappaMapCoadderConfig coadder."""
    map_path_1 = overlapping_map_set_1
    map_path_2 = overlapping_map_set_2

    start_time = Time("2025-10-10", format="iso")
    input_maps = [
        RhoAndKappaMap(
            rho_filename=map_path_1["rho"],
            kappa_filename=map_path_1["kappa"],
            time_filename=map_path_1["time"],
            frequency="f090",
            start_time=start_time,
            end_time=start_time + TimeDelta(3600, format="sec"),
        ),
        RhoAndKappaMap(
            rho_filename=map_path_2["rho"],
            kappa_filename=map_path_2["kappa"],
            time_filename=map_path_2["time"],
            frequency="f090",
            start_time=start_time + TimeDelta(3600, format="sec"),
            end_time=start_time + TimeDelta(7200, format="sec"),
        ),
    ]
    [input_maps[i].build() for i in range(len(input_maps))]
    assert np.all(input_maps[0].rho != input_maps[1].rho)
    coadder = RhoKappaMapCoadder(frequencies=["f090"])
    coadded_maps = coadder.coadd(input_maps=input_maps)
    coadd = coadded_maps[0]
    coadd.finalize()
    # Check that the coadded map is a CoaddedRhoKappaMap
    assert isinstance(coadded_maps, list)
    assert coadd.flux is not None
    assert coadd.snr is not None
    assert coadd.time_mean is not None
    corners = (
        coadd.bbox
    )  # [[dec_min, ra_max], [dec_max, ra_min]] in radians (pixell east-left)
    b0, b1 = input_maps[0].bbox, input_maps[1].bbox
    assert corners[0][0] == pytest.approx(min(b0[0][0], b1[0][0]))  # dec_min
    assert corners[0][1] == pytest.approx(max(b0[0][1], b1[0][1]))  # ra_max (east edge)
    assert corners[1][0] == pytest.approx(max(b0[1][0], b1[1][0]))  # dec_max
    assert corners[1][1] == pytest.approx(min(b0[1][1], b1[1][1]))  # ra_min (west edge)
    assert ~np.all(coadd.hits <= 1)


def test_coadding_with_sources_in_separate_maps(
    separate_map_set_1, separate_map_set_2, separate_source_params
):
    map_path_1 = separate_map_set_1
    map_path_2 = separate_map_set_2

    start_time = Time("2025-10-10", format="iso")
    input_maps = [
        RhoAndKappaMap(
            rho_filename=map_path_1["rho"],
            kappa_filename=map_path_1["kappa"],
            time_filename=map_path_1["time"],
            frequency="f090",
            start_time=start_time,
            end_time=start_time + TimeDelta(3600, format="sec"),
        ),
        RhoAndKappaMap(
            rho_filename=map_path_2["rho"],
            kappa_filename=map_path_2["kappa"],
            time_filename=map_path_2["time"],
            frequency="f090",
            start_time=start_time + TimeDelta(3600, format="sec"),
            end_time=start_time + TimeDelta(7200, format="sec"),
        ),
    ]

    sources = [
        RegisteredSource(
            ra=separate_source_params[s]["ra"],
            dec=separate_source_params[s]["dec"],
            frequency=90.0 * u.GHz,
            flux=separate_source_params[s]["flux"],
            source_id=f"sim-{s}",
            source_type="simulated",
            err_ra=0.0 * u.deg,
            err_dec=0.0 * u.deg,
            err_flux=0.0 * u.Jy,
        )
        for s in separate_source_params
    ]

    [input_maps[i].build() for i in range(len(input_maps))]
    assert np.all(input_maps[0].rho != input_maps[1].rho)

    simulated_sources = [
        FixedSimulatedSource(position=SkyCoord(ra=x.ra, dec=x.dec), flux=x.flux)
        for x in sources
    ]

    coadder = RhoKappaMapCoadder(frequencies=["f090"])
    coadded_maps = coadder.coadd(input_maps=input_maps)
    coadd = coadded_maps[0]
    coadd.finalize()
    injector = source_injector.PhotutilsSourceInjector()
    _, new_map = injector.inject(input_map=coadd, simulated_sources=simulated_sources)
    new_map.finalize()

    assert new_map != coadd
    assert new_map.original_map == coadd

    source_pos_1 = coadd.flux.sky2pix(
        [
            separate_source_params["source1"]["dec"].to_value(u.rad),
            separate_source_params["source1"]["ra"].to_value(u.rad),
        ]
    )
    source_pos_2 = coadd.flux.sky2pix(
        [
            separate_source_params["source2"]["dec"].to_value(u.rad),
            separate_source_params["source2"]["ra"].to_value(u.rad),
        ]
    )
    pix1 = int(source_pos_1[0]), int(source_pos_1[1])
    pix2 = int(source_pos_2[0]), int(source_pos_2[1])
    assert new_map.flux[pix1] - float(coadd.flux[pix1]) == pytest.approx(
        separate_source_params["source1"]["flux"].to_value(u.Jy), abs=0.5
    )
    assert new_map.flux[pix2] - float(coadd.flux[pix2]) == pytest.approx(
        separate_source_params["source2"]["flux"].to_value(u.Jy), abs=0.5
    )
    assert coadd.hits[pix1] == 1
    assert coadd.hits[pix2] == 1


def test_coadding_with_sources_in_overlapping_maps(
    overlapping_map_set_1, overlapping_map_set_2, overlapping_source_params
):
    map_path_1 = overlapping_map_set_1
    map_path_2 = overlapping_map_set_2

    start_time = Time("2025-10-10", format="iso")
    input_maps = [
        RhoAndKappaMap(
            rho_filename=map_path_1["rho"],
            kappa_filename=map_path_1["kappa"],
            time_filename=map_path_1["time"],
            frequency="f090",
            start_time=start_time,
            end_time=start_time + TimeDelta(3600, format="sec"),
        ),
        RhoAndKappaMap(
            rho_filename=map_path_2["rho"],
            kappa_filename=map_path_2["kappa"],
            time_filename=map_path_2["time"],
            frequency="f090",
            start_time=start_time + TimeDelta(3600, format="sec"),
            end_time=start_time + TimeDelta(7200, format="sec"),
        ),
    ]

    sources = [
        RegisteredSource(
            ra=overlapping_source_params[s]["ra"],
            dec=overlapping_source_params[s]["dec"],
            frequency=90.0 * u.GHz,
            flux=overlapping_source_params[s]["flux"],
            source_id=f"sim-{s}",
            source_type="simulated",
            err_ra=0.0 * u.deg,
            err_dec=0.0 * u.deg,
            err_flux=0.0 * u.Jy,
        )
        for s in overlapping_source_params
    ]

    [input_maps[i].build() for i in range(len(input_maps))]
    assert np.all(input_maps[0].rho != input_maps[1].rho)

    simulated_sources = [
        FixedSimulatedSource(position=SkyCoord(ra=x.ra, dec=x.dec), flux=x.flux)
        for x in sources
    ]

    coadder = RhoKappaMapCoadder(frequencies=["f090"])
    coadded_maps = coadder.coadd(input_maps=input_maps)
    coadd = coadded_maps[0]

    coadd.finalize()
    injector = source_injector.PhotutilsSourceInjector()
    _, new_map = injector.inject(input_map=coadd, simulated_sources=simulated_sources)
    new_map.finalize()

    assert new_map != coadd
    assert new_map.original_map == coadd

    source_pos_1 = coadd.flux.sky2pix(
        [
            overlapping_source_params["source1"]["dec"].to_value(u.rad),
            overlapping_source_params["source1"]["ra"].to_value(u.rad),
        ]
    )
    source_pos_2 = coadd.flux.sky2pix(
        [
            overlapping_source_params["source2"]["dec"].to_value(u.rad),
            overlapping_source_params["source2"]["ra"].to_value(u.rad),
        ]
    )
    pix1 = int(source_pos_1[0]), int(source_pos_1[1])
    pix2 = int(source_pos_2[0]), int(source_pos_2[1])
    assert new_map.flux[pix1] - float(coadd.flux[pix1]) == pytest.approx(
        overlapping_source_params["source1"]["flux"].to_value(u.Jy), abs=0.5
    )
    assert new_map.flux[pix2] - float(coadd.flux[pix2]) == pytest.approx(
        overlapping_source_params["source2"]["flux"].to_value(u.Jy), abs=0.5
    )
    assert coadd.hits[pix1] == 2
    assert coadd.hits[pix2] == 1


def test_intensity_coadder_separate(separate_map_set_1, separate_map_set_2):
    """Test the IntensityMapCoadder on non-overlapping raw intensity/ivar maps."""
    map_path_1 = separate_map_set_1
    map_path_2 = separate_map_set_2

    start_time = Time("2025-10-10", format="iso")
    input_maps = [
        IntensityAndInverseVarianceMap(
            intensity_filename=map_path_1["map"],
            inverse_variance_filename=map_path_1["ivar"],
            time_filename=map_path_1["time"],
            frequency="f090",
            start_time=start_time,
            end_time=start_time + TimeDelta(3600, format="sec"),
        ),
        IntensityAndInverseVarianceMap(
            intensity_filename=map_path_2["map"],
            inverse_variance_filename=map_path_2["ivar"],
            time_filename=map_path_2["time"],
            frequency="f090",
            start_time=start_time + TimeDelta(3600, format="sec"),
            end_time=start_time + TimeDelta(7200, format="sec"),
        ),
    ]
    for m in input_maps:
        m.build()

    coadder = IntensityMapCoadder(frequencies=["f090"])
    coadded_maps = coadder.coadd(input_maps=input_maps)
    assert isinstance(coadded_maps, list)

    coadd = coadded_maps[0]
    assert coadd.intensity is not None
    assert coadd.inverse_variance is not None
    assert np.all(np.isfinite(coadd.intensity))
    assert np.all(coadd.hits <= 1)
    corners = coadd.bbox
    b0, b1 = input_maps[0].bbox, input_maps[1].bbox
    assert corners[0][0] == pytest.approx(min(b0[0][0], b1[0][0]))
    assert corners[0][1] == pytest.approx(max(b0[0][1], b1[0][1]))
    assert corners[1][0] == pytest.approx(max(b0[1][0], b1[1][0]))
    assert corners[1][1] == pytest.approx(min(b0[1][1], b1[1][1]))


def test_intensity_coadder_overlapping(overlapping_map_set_1, overlapping_map_set_2):
    """Test the IntensityMapCoadder does an ivar-weighted mean on overlapping maps."""
    map_path_1 = overlapping_map_set_1
    map_path_2 = overlapping_map_set_2

    start_time = Time("2025-10-10", format="iso")
    input_maps = [
        IntensityAndInverseVarianceMap(
            intensity_filename=map_path_1["map"],
            inverse_variance_filename=map_path_1["ivar"],
            time_filename=map_path_1["time"],
            frequency="f090",
            start_time=start_time,
            end_time=start_time + TimeDelta(3600, format="sec"),
        ),
        IntensityAndInverseVarianceMap(
            intensity_filename=map_path_2["map"],
            inverse_variance_filename=map_path_2["ivar"],
            time_filename=map_path_2["time"],
            frequency="f090",
            start_time=start_time + TimeDelta(3600, format="sec"),
            end_time=start_time + TimeDelta(7200, format="sec"),
        ),
    ]
    for m in input_maps:
        m.build()

    coadder = IntensityMapCoadder(frequencies=["f090"])
    coadded_maps = coadder.coadd(input_maps=input_maps)
    coadd = coadded_maps[0]

    assert np.all(coadd.inverse_variance >= 0)
    assert ~np.all(coadd.hits <= 1)

    # Where both maps overlap, the coadd should equal the exact
    # inverse-variance-weighted mean of the two input pixel values.
    overlap = coadd.hits == 2
    assert np.any(overlap)
    intensity_1 = enmap.project(
        input_maps[0].intensity, coadd.intensity.shape, coadd.intensity.wcs
    )
    ivar_1 = enmap.project(
        input_maps[0].inverse_variance, coadd.intensity.shape, coadd.intensity.wcs
    )
    intensity_2 = enmap.project(
        input_maps[1].intensity, coadd.intensity.shape, coadd.intensity.wcs
    )
    ivar_2 = enmap.project(
        input_maps[1].inverse_variance, coadd.intensity.shape, coadd.intensity.wcs
    )
    expected = (intensity_1 * ivar_1 + intensity_2 * ivar_2) / (ivar_1 + ivar_2)
    np.testing.assert_allclose(coadd.intensity[overlap], expected[overlap], rtol=1e-6)
    np.testing.assert_allclose(
        coadd.inverse_variance[overlap], (ivar_1 + ivar_2)[overlap], rtol=1e-6
    )


def test_intensity_coadder_single_map(separate_map_set_1):
    """Coadding a single map should return the map unchanged (mod the wrapper class)."""
    map_path_1 = separate_map_set_1
    start_time = Time("2025-10-10", format="iso")
    input_map = IntensityAndInverseVarianceMap(
        intensity_filename=map_path_1["map"],
        inverse_variance_filename=map_path_1["ivar"],
        time_filename=map_path_1["time"],
        frequency="f090",
        start_time=start_time,
        end_time=start_time + TimeDelta(3600, format="sec"),
    )
    input_map.build()

    coadder = IntensityMapCoadder(frequencies=["f090"])
    coadd = coadder.coadd(input_maps=[input_map])[0]

    np.testing.assert_allclose(coadd.intensity, input_map.intensity)
    np.testing.assert_allclose(coadd.inverse_variance, input_map.inverse_variance)


def test_intensity_coadder_labels_combined_arrays(
    overlapping_map_set_1, overlapping_map_set_2
):
    """
    Coadding maps from different arrays into one "coadd" group should label
    the result with the arrays actually combined (e.g. "i1i6"), not just
    whichever input map happened to be first.
    """
    map_path_1 = overlapping_map_set_1
    map_path_2 = overlapping_map_set_2

    start_time = Time("2025-10-10", format="iso")
    input_maps = [
        IntensityAndInverseVarianceMap(
            intensity_filename=map_path_1["map"],
            inverse_variance_filename=map_path_1["ivar"],
            time_filename=map_path_1["time"],
            frequency="f090",
            array="i6",
            start_time=start_time,
            end_time=start_time + TimeDelta(3600, format="sec"),
        ),
        IntensityAndInverseVarianceMap(
            intensity_filename=map_path_2["map"],
            inverse_variance_filename=map_path_2["ivar"],
            time_filename=map_path_2["time"],
            frequency="f090",
            array="i1",
            start_time=start_time + TimeDelta(3600, format="sec"),
            end_time=start_time + TimeDelta(7200, format="sec"),
        ),
    ]

    coadder = IntensityMapCoadder(frequencies=["f090"])
    coadd = coadder.coadd(input_maps=input_maps)[0]
    assert coadd.array == "i1i6"


def test_intensity_coadder_labels_single_array(
    overlapping_map_set_1, overlapping_map_set_2
):
    """Coadding maps that all share one array should keep that array's plain name."""
    map_path_1 = overlapping_map_set_1
    map_path_2 = overlapping_map_set_2

    start_time = Time("2025-10-10", format="iso")
    input_maps = [
        IntensityAndInverseVarianceMap(
            intensity_filename=map_path_1["map"],
            inverse_variance_filename=map_path_1["ivar"],
            time_filename=map_path_1["time"],
            frequency="f090",
            array="i6",
            start_time=start_time,
            end_time=start_time + TimeDelta(3600, format="sec"),
        ),
        IntensityAndInverseVarianceMap(
            intensity_filename=map_path_2["map"],
            inverse_variance_filename=map_path_2["ivar"],
            time_filename=map_path_2["time"],
            frequency="f090",
            array="i6",
            start_time=start_time + TimeDelta(3600, format="sec"),
            end_time=start_time + TimeDelta(7200, format="sec"),
        ),
    ]

    coadder = IntensityMapCoadder(frequencies=["f090"])
    coadd = coadder.coadd(input_maps=input_maps)[0]
    assert coadd.array == "i6"


def test_rhokappa_coadder_streaming_merge_does_not_repeat_array_tokens(
    overlapping_map_set_1, overlapping_map_set_2
):
    """
    Regression test: streaming/incremental coadding calls coadd_maps()
    repeatedly as [running_coadd, new_map]. running_coadd.array is itself
    already a combined label from the previous merge (e.g. "i1i3"); treating
    that whole string as one opaque token instead of re-parsing it into its
    atomic arrays made the label grow (and repeat tokens) with every merge
    instead of staying deduplicated to the unique arrays actually involved.
    """
    start_time = Time("2025-10-10", format="iso")
    paths = [overlapping_map_set_1, overlapping_map_set_2, overlapping_map_set_1]
    arrays = ["i1", "i3", "i6"]

    coadder = RhoKappaMapCoadder(frequencies=["f090"])
    running = None
    for path, array in zip(paths, arrays):
        m = RhoAndKappaMap(
            rho_filename=path["rho"],
            kappa_filename=path["kappa"],
            time_filename=path["time"],
            frequency="f090",
            array=array,
            start_time=start_time,
            end_time=start_time + TimeDelta(3600, format="sec"),
        )
        running = coadder.coadd_maps([m] if running is None else [running, m])

    assert running.array == "i1i3i6"


def test_intensity_coadder_time_mean_is_averaged_not_summed(tmp_path):
    """
    Regression test: coadding several fully-overlapping maps must produce a
    hits-weighted *average* time_mean, not an ever-growing running sum (see
    update_times() in core.py -- total_hits used to come from a 0/1
    "is this pixel covered" indicator instead of the real accumulated hit
    count, so each additional map multiplied the running sum instead of
    renormalizing it).
    """
    shape, wcs = enmap.geometry(
        np.deg2rad([[-0.5, 0.5], [0.5, -0.5]]), res=np.deg2rad(0.05), proj="car"
    )
    start_time = Time("2025-10-10", format="iso")

    times = [100.0, 200.0, 300.0, 400.0, 500.0]
    input_maps = []
    for i, t in enumerate(times):
        intensity_path = tmp_path / f"map{i}_map.fits"
        ivar_path = tmp_path / f"map{i}_ivar.fits"
        time_path = tmp_path / f"map{i}_time.fits"
        enmap.ones(shape, wcs=wcs).write(intensity_path, fmt="fits")
        enmap.ones(shape, wcs=wcs).write(ivar_path, fmt="fits")
        (enmap.zeros(shape, wcs=wcs) + t).write(time_path, fmt="fits")

        input_maps.append(
            IntensityAndInverseVarianceMap(
                intensity_filename=intensity_path,
                inverse_variance_filename=ivar_path,
                time_filename=time_path,
                frequency="f090",
                start_time=start_time,
                end_time=start_time + TimeDelta(3600, format="sec"),
            )
        )

    coadder = IntensityMapCoadder(frequencies=["f090"])
    coadd = coadder.coadd(input_maps=input_maps)[0]

    # build() converts the raw (seconds-since-start) time map to absolute
    # unix time by adding observation_start; read back what each input map
    # actually ended up with rather than re-deriving that offset here.
    expected_mean = np.mean([float(m.time_mean[0, 0]) for m in input_maps])

    np.testing.assert_allclose(coadd.hits, len(times))
    np.testing.assert_allclose(coadd.time_mean, expected_mean)
