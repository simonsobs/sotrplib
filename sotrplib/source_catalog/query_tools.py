import structlog
logger = structlog.get_logger(__name__)
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from astroquery.simbad import Simbad
from tqdm import tqdm


def cone_query_gaia(ra_deg:float,
                    dec_deg:float,
                    radius_arcmin:float=1.0,
                    columns=['designation','ra','ra_error','dec','dec_error','parallax','parallax_error','phot_g_mean_mag', 'phot_g_mean_flux_error']
                    ):
    '''
    Query gaia using astroquery.
    Performs cone search at ra_deg,dec_deg with a search radius of radius_arcmin
    '''
    from astroquery.gaia import Gaia 
    Gaia.ROW_LIMIT = 100  
    coord = SkyCoord(ra=ra_deg, 
                     dec=dec_deg,
                     unit=(u.degree, u.degree), 
                     frame='icrs'
                     )
    gaia_results = Gaia.cone_search(coord, 
                                    radius=u.Quantity(radius_arcmin, u.arcmin),
                                    columns=columns
                                    ).get_results()
    if not gaia_results:
        return {}
    else:
        return gaia_results
    


def SIMBAD(ra, dec, radius):
    """
    Query SIMBAD for objects within a given radius of a given RA/Dec

    Args:
        ra (float): RA in degrees
        dec (float): Dec in degrees
        radius (float): radius in arcmin

    Returns:
        pd.DataFrame: SIMBAD objects within the search radius
        (id, type, sep, mag, dist)
    """

    # add flux field to search
    Simbad.add_votable_fields("flux(G)")
    Simbad.add_votable_fields("otype")
    Simbad.add_votable_fields("sp(S)")
    Simbad.add_votable_fields("plx", "distance")

    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs")
    radius = radius * u.arcmin

    # init df
    df = pd.DataFrame(
        columns=["id", "type", "spectral type", "sep [arcsec]", "mag", "dist [pc]"]
    )

    r = Simbad.query_region(coord, radius=radius)

    if r is not None:
        for j in range(len(r)):
            row = [
                r["MAIN_ID"][j],
                r["OTYPE"][j],
                r["SP_TYPE_S"][j],
                coord.separation(
                    SkyCoord(
                        ra=r["RA"][j],
                        dec=r["DEC"][j],
                        unit=(u.hourangle, u.deg),
                        frame="icrs",
                    )
                )
                .to(u.arcsec)
                .value,
                r["FLUX_G"][j],
                simdist(r[j]),
            ]
            df = df.append(pd.Series(row, index=df.columns), ignore_index=True)

    return df


def simdist(sim, verbose=False):
    """
    Get distance from SIMBAD object

    Args:
        sim (astropy.table.row.Row): SIMBAD object
        verbose (bool): print verbose output

    Returns:
        float: distance in pc
    """
    # Prefer the parallax distance if available:
    if sim["PLX_VALUE"]:
        # Parallax distance:
        if verbose:
            logger.info("Parallax distance", distance_pc=1000.0 / sim["PLX_VALUE"])
        dist = 1000.0 / sim["PLX_VALUE"]
    # but if no parallax we'll take any other distance:
    elif sim["Distance_distance"]:
        if verbose:
            logger.info(
                "Distance from reference",
                bibcode=sim["Distance_bibcode"],
                distance=sim["Distance_distance"],
                unit=sim["Distance_unit"]
            )
        if sim["Distance_unit"] == "kpc":
            dist = sim["Distance_distance"] * 1000
        elif sim["Distance_unit"] == "pc":
            dist = sim["Distance_distance"]

    else:
        if verbose:
            logger.info("No distance available in Simbad.")
        dist = np.nan

    return dist


def pvalue(df, radius=2.0):
    # Query Gaia for all sources from DR3 within ACT footprint and G < match brightness

    Gaia.ROW_LIMIT = 10000
    p = np.zeros(len(df))
    n = np.zeros(len(df))

    radius = radius * u.deg

    for i in tqdm(range(len(df))):
        # if index is same as previous, skip
        if i > 0 and df["event"].iloc[i] == df["event"].iloc[i - 1]:
            continue

        # separate all sources with same index
        dfi = df[df["event"] == df["event"].iloc[i]]

        # parse each mag in dfi
        mags = []
        for j in range(len(dfi)):
            # check if mag is nan
            if dfi["mag"].iloc[j] == dfi["mag"].iloc[j]:
                # if string can convert to float save mag
                try:
                    mags.append(float(dfi["mag"].iloc[j]))
                # if not, continue
                except ValueError:
                    logger.warning("Simbad mag not float", index=i + j)
                    mags.append(np.nan)
                    continue

            else:
                logger.warning("mag is nan", index=i + j)
                mags.append(np.nan)
                continue

        # if no mags, continue
        if len(mags) == 0:
            continue

        # get max mag to perform query excluding nans
        mag = np.nanmax(mags)

        # if nan, continue
        if mag != mag:
            continue

        # get ra/dec
        ra = df["ra"].iloc[i]
        dec = df["dec"].iloc[i]

        query = f"""
        SELECT phot_g_mean_mag, DISTANCE(
            POINT({ra}, {dec}),
            POINT(ra, dec)) AS ang_sep
        FROM gaiadr3.gaia_source_lite
        WHERE 1 = CONTAINS(
            POINT({ra}, {dec}),
            CIRCLE(ra, dec, {radius.to(u.deg).value}))
        AND phot_g_mean_mag <= {mag}
        AND parallax IS NOT NULL
        ORDER BY ang_sep ASC
        """

        job = Gaia.launch_job_async(query)
        r = job.get_results()

        # calculate p-value for each mag
        for j in range(len(mags)):
            # skip if mag is nan
            if mags[j] != mags[j]:
                continue

            # get length of r with mag less than or equal to transient mag
            n[i + j] = len(r[r["phot_g_mean_mag"] <= mags[j]])

            # Calculate density of Gaia sources within 2 deg of transient
            rho = n[i + j] / (np.pi * radius.to(u.rad).value ** 2)

            # get sep
            sep = df["sep [arcsec]"].iloc[i + j] * u.arcsec

            # calculate p-value
            p[i + j] = 1 - np.exp(-rho * sep.to(u.rad).value ** 2 * np.pi)

            # print pvalue
            logger.info("pvalue of object", index=i + j, pvalue=p[i + j])

    # add p-value  to df
    df["pvalue"] = p
    # df['n_stars'] = n
    return df
