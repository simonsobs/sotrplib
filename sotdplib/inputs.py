import os.path as op
import numpy as np
from astropy.table import Table


def get_fwhm_arcmin(arr, freq):
    ome = -1000  # in nsr
    if arr == "pa4" and freq == "f220":
        ome = 113.34
    if arr == "pa4" and freq == "f150":
        ome = 227.89
    if arr == "pa5" and freq == "f150":
        ome = 219.69
    if arr == "pa5" and freq == "f090":
        ome = 480.93
    if arr == "pa6" and freq == "f150":
        ome = 229.90
    if arr == "pa6" and freq == "f090":
        ome = 492.73
    if arr == "pa7" and freq == "f040":
        ome = 2804
    if arr == "pa7" and freq == "f030":
        ome = 5060
    fwhm = (ome * 1e-9 * 4 * np.log(2.0) / np.pi) ** 0.5 * 180.0 * 60.0 / np.pi
    return fwhm


def get_sourceflux_threshold(freq):
    """
    Get sourceflux threshold measured from mapnoise * 5 / 5

    avg dflux of f090 = 28.517394966150707
    avg dflux of f150 = 53.679349692751394
    avg dflux of f220 = 88.04350067775873
    avg dflux of f040 = 93.94425336313529
    avg dflux of f030 = 133.4519154035587


    Args:
        freq: frequency

    Returns:
        sourceflux threshold [mJy]
    """
    if freq == "f090":
        sourceflux_thresh = 30.0
    elif freq == "f150":
        sourceflux_thresh = 50.0
    elif freq == "f220":
        sourceflux_thresh = 90.0
    elif freq == "f040":
        sourceflux_thresh = 90.0
    elif freq == "f030":
        sourceflux_thresh = 130.0
    else:
        raise ValueError("Frequency not recognized")
    return sourceflux_thresh


def get_sourcecats():
    sourcecat_f090 = op.join(
        op.dirname(__file__), "../data/inputs/PS_S19_f090_2pass_optimalCatalog.fits"
    )
    sourcecat_f150 = op.join(
        op.dirname(__file__), "../data/inputs/PS_S19_f150_2pass_optimalCatalog.fits"
    )
    sourcecat_f220 = op.join(
        op.dirname(__file__), "../data/inputs/PS_S19_f220_2pass_optimalCatalog.fits"
    )
    sourcecat_f030 = op.join(
        op.dirname(__file__),
        "../data/inputs/cmb_night_pa7_f030_4pass_8way_coadd_map_cat.fits",
    )
    sourcecat_f040 = op.join(
        op.dirname(__file__),
        "../data/inputs/cmb_night_pa7_f040_4pass_8way_coadd_map_cat.fits",
    )

    s040 = Table.read(sourcecat_f040)
    s040["RADeg"] = s040["ra"]
    s040["decDeg"] = s040["dec"]
    # fconv_40 = utils.dplanck(40.e9, utils.T_cmb)
    # s040['fluxJy'] = s040['flux'][:, 0] * fconv_40 * 2804.
    s040["fluxJy"] = s040["flux"][:, 0] / 1e3

    s030 = Table.read(sourcecat_f030)
    s030["RADeg"] = s030["ra"]
    s030["decDeg"] = s030["dec"]
    # fconv_30 = utils.dplanck(30.e9, utils.T_cmb)
    # s030['fluxJy'] = s030['flux'][0, :] * fconv_30 * 5060.
    s030["fluxJy"] = s030["flux"][:, 0] / 1e3

    sourcecats = {
        "f090": Table.read(sourcecat_f090),
        "f150": Table.read(sourcecat_f150),
        "f220": Table.read(sourcecat_f220),
        "f030": s030,
        "f040": s040,
    }
    return sourcecats


def get_badmaps():
    """Looks like the first ctime doesn't exist. Keeping for reference here."""
    return ["150214186", "1502421750"]


@np.vectorize
def planck_fluxcal(arr, freq):
    """
    Get flux calibration factor for a given frequency

    Args:
        freq: frequency

    Returns:
        flux calibration factor
    """

    # with open('../../data/inputs/calibs_dict_ACTxACT-dr4.pkl', 'rb') as f:
    # dr6_calib = pk.load(f)
    #
    # calib_dict = {'pa4': {'f150': dr6_calib['dr6_pa4_f150']['calibs'], 'f220': dr6_calib['dr6_pa4_f220']['calibs']},
    #       'pa5': {'f090': dr6_calib['dr6_pa5_f090']['calibs'], 'f150': dr6_calib['dr6_pa5_f150']['calibs']},
    #       'pa6': {'f090': dr6_calib['dr6_pa6_f090']['calibs'], 'f150': dr6_calib['dr6_pa6_f150']['calibs']}
    #                 }

    calib_dict = {
        "pa4": {
            "f150": [1.0072175262677963, 0.0023154207035651023],
            "f220": [1.0340185610989223, 0.01626333139583416],
        },
        "pa5": {
            "f090": [1.0218836321586204, 0.0011691303344386838],
            "f150": [0.9876564792184743, 0.001704546067935355],
        },
        "pa6": {
            "f090": [1.0181628559678786, 0.0013096342890986763],
            "f150": [0.9699498005405339, 0.002141045797557799],
        },
        "pa7": {"f030": [1.0, 1.0], "f040": [1.0, 1.0]},
    }  # Alternatively hardcoded

    return calib_dict[arr][freq][0], calib_dict[arr][freq][1]


@np.vectorize
def get_freq(arr, freq):
    freq_dict = {
        "pa4": {"f150": 148.7e9, "f220": 227.2e9},
        "pa5": {"f090": 96.7e9, "f150": 149.5e9},
        "pa6": {"f090": 95.5e9, "f150": 148.1e9},
        "pa7": {"f030": 27.2e9, "f040": 37.7e9},
    }  # Effective centers from Hasslefield paper 2022 in prep

    return freq_dict[arr][freq]
