import astropy.units as u
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

## used in generate_asteroid_ephem to provide location of observatory.
SO_LAT = {"lat": -22.96098*u.deg,
          "lon": -67.78769*u.deg,
          "elevation": 5147*u.m,
         }

def flux_stm(
    frequency,
    diam,
    r_sun,
    r_earth,
    alpha,
    A=0.0,
    eps=1.0,
    eta=0.756,
    rayleigh_jeans=False,
    T_factor=1.0,
):
    """
    Function to estimate asteroid flux using the Standard Thermal
    Model (STM) (Lebofsky 1986), which assumes a nonrotating or 0-
    thermal-inertia spherical body.
    STM should be considered an upper limit on possible flux, and it
    has lately fallen out of use for NEATM.
    See (Delbo and Harris 2002) for details.

    Arguments
    ---------
    frequency : float
        Observing frequency, Hz.
    diam : float
        Asteroid diameter, meters.
    r_sun, r_earth, alpha : value or list-like
        Asteroid distance to sun, distance to earth, and solar phase
        angle, meters.
    A, eps, eta : float [0, 1, 0.756]
        Asteroid's bond albedo, emissivity, and "beaming parameter."
        Defaults are standard empirical assumptions.
    rayleigh_jeans : bool [False]
        Whether to use long-wavelength Rayleigh-Jeans limit for
        thermal emission.  Doing so may be faster but more inaccurate
        at the few-percent level.
    T_factor : float [1]
        Multiplicatively scales the model asteroid temperature at the
        sub-solar point. Useful for finding temperature from flux,
        otherwise it should be left as default.

    Returns
    ---------
    F : value or list-like
        STM flux density in mJy for given variables.
    """
    from astropy.constants import L_sun
    from scipy.constants import sigma, h, c, k, pi
    from scipy.integrate import quad

    if not np.isscalar(r_sun):
        r_sun = np.array(r_sun)
        r_earth = np.array(r_earth)
        alpha = np.array(alpha)

    if rayleigh_jeans:
        integral = 4 / 9
        F = (
            T_factor
            * 4 ** 0.75
            * pi ** 0.75
            * k
            * L_sun.value ** 0.25
            / 9
            / sigma ** 0.25
            / c ** 2
            * eps ** 0.75
            * (diam) ** 2
            * (r_earth) ** -2
            * (r_sun) ** -0.5
            * (frequency) ** 2
            * (1 - A) ** 0.25
            * eta ** -0.25
        )

    else:
        # incident solar power [W/m^2]
        S = L_sun.value / 4 / pi / (r_sun) ** 2

        # temperature at asteroid sub-solar point [K]
        T_ss = T_factor * (S * (1 - A) / sigma / eps / eta) ** 0.25

        # integrate emission over surface of asteroid
        def func(theta, T_ss, frequency):
            # theta = ang dist from subsolar point
            T_theta = T_ss * np.cos(theta) ** 0.25
            exp = h * (frequency) / k / T_theta
            integrand = np.cos(theta) * np.sin(theta) / (np.exp(exp) - 1)
            return integrand

        if np.isscalar(T_ss):
            integral = quad(lambda x: func(x, T_ss, frequency), a=0, b=pi / 2)[0]
        else:
            integral = np.array(
                [quad(lambda x: func(x, T, frequency), a=0, b=pi / 2)[0] for T in T_ss]
            )

        # final flux calculation
        F = (
            eps
            * (diam) ** 2
            * pi
            * h
            * (frequency) ** 3
            / (r_earth) ** 2
            / c ** 2
            * integral
        )

    # account for solar phase factor, decrease 0.01mag/deg
    F *= 10.0 ** (-0.01 * np.abs(alpha*pi/180.) / 2.5)

    # convert to mJy, F currently in [W m^-2 Hz^-1 = 10^26 Jy]
    F *= 10.0 ** 29
    return F

def get_bond_albedo(geometric_albedo, G=0.15):
    """
    Function to convert geometric albedo to bond albedo as needed for
    flux models.  Converts using the H-G magnitude system for phase
    integral (Bowell 1989).
    
    Arguments
    ---------
    geometric_albedo : float
        Geometric albedo of the object.
    G : float [0.15]
        G parameter of the H-G magnitude system.  Default is standard
        assumption.
    
    Returns
    ---------
    bond_albedo : float
        Calculated bond albedo.
    """
    
    return geometric_albedo * (0.29 + 0.684 * G)
    

def generate_asteroid_ephem(
    asteroid,
    date_start=None,
    date_stop=None,
    step=2 ,
    frequencies=np.array([150e9]),
    observatory=SO_LAT,
    filename=None,
):
    """
    Query JPL HORIZONS and return resulting G3Frame with ephems.  Provide either
    a date_list or both date_start and date_stop.

    Arguments
    ---------
    asteroid : str or int
        Name or designation number of object. For example, 324, '324', or
        'Bamberga' would all generate ephemerides for (324) Bamberga.
    date_start, date_stop : str 
        Date to start and end ephemerides at as 'YYYY-MM-DD HH:MM' format.
    step : int [2]
        Amount of time in hours between each ephemeris if using date_start
        and date_stop. Must be an integer number of hours.
    frequencies : list-like [np.array([90, 150, 220])]
        If desired, list of frequencies in Hz at which we estimate model
        flux for objects present in the field.  Uses Standard Thermal Model to
        predict fluxes, assuming emissivity=1, beaming parameter=0.756
        (standard), and (if not known by JPL) albedo=0.  If JPL does not know
        asteroid diameter, no flux is predicted. More accurate predictions
        should be obtained using NEATM with NEOWISE beaming parameters.
    observatory : str, dict
        Name of observatory if string, otherwise a dict containing lat,lon, elevation.
    query_xyz_pos : bool [False]
        Whether to query ecliptic XYZ position and velocity vectors. If True, adds
        heliocentric and geocentric (technically not geocentric, but rather with
        the origin at observatory) positions of the object.  To know the direction of the
        sun and SPT from the perspective of the object, negate these vectors.
    filename : str
        Path to file you want to save the G3Frame to.
        If no path is provided, the frame is returned without writing to disk.
        If a relative path is provided, the frame is stored in the default
        directory "sso_ephems", which is inside the transients python library.
    append : bool
        If True and the output filename already exists, the output frame is
        appended to the existing file.

    Returns
    -------
    frame : G3Frame
        A G3Frame that contains: useful object properties and a G3TimesampleMap
        with useful ephemerides information in G3Units.
    """
    from astroquery.jplhorizons import Horizons
    from astroquery.jplsbdb import SBDB

    
    if date_start is None:
        date_start = "2025-01-01 00:00"
    
    if date_stop is None:
        date_stop = "2035-12-31 00:00"
    
    
    epochs = {"start": date_start,
              "stop": date_stop,
              "step": "{}h".format(int(step)),
             }
       
 
    eph = Horizons(id=asteroid, location=observatory, id_type="smallbody", epochs=epochs)
    eph = eph.ephemerides(cache=False)
    keys2keep=['datetime_str','datetime_jd','RA','DEC','r','delta','alpha','RA_rate','DEC_rate','targetname']
    for key in eph.colnames:
        if key not in keys2keep:
            eph.remove_column(key)
    eph.rename_column('r','SunDist')
    eph.rename_column('delta','EarthDist')
    eph.rename_column('alpha','PhaseAngle')

    frame = {}
    # try querying other useful information, store if known
    if "(" in eph["targetname"][0]:  # all asteroids
        SBDB_queryname = eph["targetname"][0].split("(")[-1][:-1]
    elif "/" in eph["targetname"][0]:  # comets
        SBDB_queryname = eph["targetname"][0].split("/")[0]
    else:
        SBDB_queryname = asteroid
    #try:
    props = SBDB.query(SBDB_queryname, phys=True, cache=False)
    if "des" in props["object"].keys():
        frame["Designation"] = props["object"]["des"]
    if "shortname" in props["object"].keys():
        frame["Name"] = props["object"]["shortname"]
    if "diameter" in props["phys_par"].keys():
        frame["Diameter"] = props["phys_par"]["diameter"]
    if "albedo" in props["phys_par"].keys():
        frame["GeometricAlbedo"] = float(props["phys_par"]["albedo"])
        if "G" in props["phys_par"].keys():
            frame["G"] = float(props["phys_par"]["G"])
            bond_albedo = get_bond_albedo(
                geometric_albedo=frame["GeometricAlbedo"],
                G=frame["G"]
            )
        else:
            bond_albedo = get_bond_albedo(geometric_albedo=frame["GeometricAlbedo"])
        frame["BondAlbedo"] = bond_albedo
    
    # estimate flux with STM if requested, then store ephems
    if frequencies is None:
        frequencies = []
    elif np.isscalar(frequencies):
        frequencies = [frequencies]
    for freq in frequencies:
        if "BondAlbedo" in frame.keys():
            A = frame["BondAlbedo"]
        else:
            A = 0
        if "Diameter" in frame.keys():
            keyname = "PredictedFlux{}GHz".format(int(freq / 1e9))
            try:
                pred_flux = flux_stm(
                    frequency=freq,
                    diam=frame["Diameter"].to(u.m),
                    r_sun=eph["SunDist"].to(u.m),
                    r_earth=eph["EarthDist"].to(u.m),
                    alpha=eph["PhaseAngle"],
                    A=A,
                )
            
                eph[keyname] = pred_flux.value * u.mJy
            except Exception:
                logger.warning("Failed to estimate flux for asteroid", asteroid=asteroid, diameter=frame.get('Diameter', None))
                eph[keyname] = np.nan
            
    frame["Ephem"] = eph

    # save the frame
    if filename is not None:
        from astropy.io import ascii
        ascii.write(eph, 
                    filename, 
                    format='csv', 
                    fast_writer=False
                    )  
    return frame
    

def load_ephem_files(files):
    """
    Load SSO ephemeris files from disk.

    Arguments
    ---------
    files : str or list
        A string specifying a glob search path where SSO ephem .csv files are
        stored, or a list of filenames to load.

    Returns
    -------
    ephems : list of ephems
        List of ephemeris objects containing SSO ephemeris data for the given
        input files.
    """
    from astropy.io.ascii import read
    from glob import glob

    if isinstance(files, str):
        if '*' in files:
            files = sorted(glob(files))
        else:
            files = [files]
    elif len(files) == 1:
        if '*' in files[0]:
            files = sorted(glob(files[0]))
    
    ephems = []
    for f in files:
        ephems.append(read(f, format='csv', fast_reader=False))

    return ephems


def interpolate_asteroid_ephem(obs_time,
                               asteroid = None,
                               ephem=None,
                               time_step_min=1,
                               time_range_hr=1,
                              ):
    """
    Function to interpolate the ephemeris around obs_time.

    Arguments
    ---------
    obs_time : float
        Observation time, unix time.
    asteroid : str
        Name of asteroid in ephem file.
    ephem : str or list
        A glob string to use for listing ephem files in a directory,
        or a list of ephem filenames, or a list of ephem frames.  If not
        supplied, ephem files will be searched for in the default
        directory.
    time_step_min : float [1]
        Interpolate to this step size, in minutes.
    time_range_hr : float [3]
        Interpolate from obs_time-time_range_hr/2 to obs_time+time_range_hr/2, in hours.

    Returns
    -------
    time, ra, dec : array_like
        Arrays containing time, ra, and dec.
    """
    from astropy.time import Time

    ephem = load_ephem_files(files=ephem)

    interp_ra = []
    interp_dec = []
    interp_times = []

    for frame in ephem:
        if "targetname" not in frame:
            continue
        asteroid_name = frame["targetname"]
        if asteroid and asteroid_name != asteroid:
            continue
        logger.info("Checking asteroid", asteroid=asteroid)
        ephem_time = Time(frame["Ephem"]['datetime_jd'],format='jd').unix
        nearest_ephem = np.argmin(np.abs(ephem_time - int(obs_time)))
        
        first_ephem = nearest_ephem
        last_ephem = nearest_ephem
        while ephem_time[first_ephem] > obs_time-time_range_hr*3600/2:
            first_ephem-=1
            if first_ephem<0:
                first_ephem=0
                break
        while ephem_time[last_ephem] < (obs_time+time_range_hr*3600/2):
            last_ephem+=1
            if last_ephem>=len(ephem_time):
                last_ephem=len(ephem_time)-1
                break
        interp_times = np.arange(ephem_time[first_ephem],
                                 ephem_time[last_ephem],
                                 time_step_min*60,
                                )
        ephem_times = ephem_time[first_ephem:last_ephem]
        ephem_ra = np.asarray(frame['Ephem']['Ra'])[first_ephem:last_ephem]
        ephem_dec = np.asarray(frame['Ephem']['Dec'])[first_ephem:last_ephem]
        
        interp_ra = np.interp(interp_times,ephem_times,ephem_ra)
        interp_dec = np.interp(interp_times,ephem_times,ephem_dec)
        
    return np.array(interp_times), np.array(interp_ra), np.array(interp_dec)


def get_object_position_at_time(obs_times,
                                ephem,
                                max_ra_rate:float=30,
                                max_dec_rate:float=30,
                                asteroid:str=None,
                               ):
    """
    Function to get the position of a SSO at a given time by 
    interpolating over the ephemeris data.
    The ephemeris data should be in the form of an Astropy Table or
    list of Astropy Tables. The function will return
    the right ascension and declination of the SSO at the given
    observation time.

    max_ra_rate, max_dec_rate  set the limits for using RA_rate and
    DEC_rate to extrapolate the position of the SSO.
    If the rate is higher than these limits, the function will
    interpolate the position using scipy spline interpolation of the RA and DEC values.

    Arguments
    ---------
    obs_times : float, str, list
        Observation time(s) in astropy Time or unix time.
    ephem : Astropy Table, str, or list
        Ephemeris table, or file path, or list of tables containing ephemeris data.
    max_ra_rate : float [30]
        Maximum rate of change of RA in arcseconds per hour.
    max_dec_rate : float [30]
        Maximum rate of change of DEC in arcseconds per hour.
    asteroid : str
        Name of asteroid to use.
        If None, the function will all asteroids.
    Returns
    -------
    interpolated_position : dict
        Dict keyed by object name, containing the RA, DEC, and
        UnixTime of the object at the given observation time.
        The RA and DEC are in degrees.
        The UnixTime is the time of the observation in seconds.
    """
    arcsec_per_hour_to_deg_per_sec = 1/3600.0/3600.0
    from astropy.time import Time
    from astropy.table import Table
    from tqdm import tqdm
    ## load the ephemeris data... 
    ## since there are so many options, there are a lot of checks.
    if isinstance(ephem, str):
        ephem = load_ephem_files(files=ephem)
    elif isinstance(ephem, list):
        if isinstance(ephem[0], str):
            ephem = load_ephem_files(files=ephem)
        elif isinstance(ephem[0], Table):
            pass
    elif isinstance(ephem, Table):
        ephem = [ephem]
    elif not isinstance(ephem, list):
        raise ValueError("ephem must be a string or a list of strings or astropy tables.")
    if len(ephem) == 0:
        raise ValueError("No ephemeris data found.")
    
    for eph in tqdm(ephem,desc='calculating unix timestamps'):
        eph['UnixTime'] = Time(eph['datetime_jd'],format='jd').unix

    interpolated_positions = {}
    if isinstance(obs_times, (str,float)):
        obs_times = [obs_times]
    for obs_time in tqdm(obs_times,desc='interpolating positions'):
        if isinstance(obs_time, str):
            obs_time = Time(obs_time).unix
        elif isinstance(obs_time, Time):
            obs_time = obs_time.unix
        elif not isinstance(obs_time, float):
            raise ValueError("obs_time must be a string or a float.")
        if obs_time is None:
            raise ValueError("obs_time must be a string or a float.")
        if obs_time < 0:
            raise ValueError("obs_time must be a positive float.")
        
        
        for eph in ephem:
            if "targetname" not in eph.colnames:
                continue
            asteroid_name = eph["targetname"][0]
            if asteroid and asteroid_name != asteroid:
                continue
            
            ## find nearest ephemeris time
            ## to the observation time
            ephem_time = eph['UnixTime']
            nearest_ephem = np.argmin(np.abs(ephem_time - int(obs_time)))
            ra=np.nan
            dec=np.nan
            if 'RA_rate' in eph.colnames and 'DEC_rate' in eph.colnames:
                ra_rate = eph['RA_rate'][nearest_ephem]
                dec_rate = eph['DEC_rate'][nearest_ephem]
            else:
                ra_rate=np.inf
                dec_rate=np.inf
            if abs(ra_rate)>max_ra_rate or abs(dec_rate)>max_dec_rate:
                from scipy.interpolate import CubicSpline
                inds = np.array([nearest_ephem-1, nearest_ephem, nearest_ephem+1])
                ## check for out of bounds
                if inds[0] < 0:
                    inds[0] = 0
                if inds[2] >= len(ephem_time):
                    inds[2] = len(ephem_time)-1
                ## get the ephemeris data
                eph_times = ephem_time[inds]
                eph_ra = np.asarray(eph['RA'])[inds]
                eph_dec = np.asarray(eph['DEC'])[inds]
                ## interpolate the ephemeris data
                interp_ra = CubicSpline(eph_times, eph_ra)
                interp_dec = CubicSpline(eph_times, eph_dec)
                ra = interp_ra(obs_time)
                dec = interp_dec(obs_time)
            else:
                ra = eph['RA'][nearest_ephem] + (obs_time - ephem_time[nearest_ephem]) * (ra_rate *arcsec_per_hour_to_deg_per_sec)
                dec = eph['DEC'][nearest_ephem] + (obs_time - ephem_time[nearest_ephem]) * (dec_rate*arcsec_per_hour_to_deg_per_sec)
            ra = ra % 360

            if asteroid_name not in interpolated_positions:
                interpolated_positions[asteroid_name] = {"RA": np.atleast_1d(ra),
                                                         "DEC": np.atleast_1d(dec),
                                                         "UnixTime": np.atleast_1d(obs_time),
                                                        }
            else:
                interpolated_positions[asteroid_name]["RA"]=np.concatenate((interpolated_positions[asteroid_name]["RA"],np.atleast_1d(ra)))
                interpolated_positions[asteroid_name]["DEC"]=np.concatenate((interpolated_positions[asteroid_name]["DEC"],np.atleast_1d(dec)))
                interpolated_positions[asteroid_name]["UnixTime"]=np.concatenate((interpolated_positions[asteroid_name]["UnixTime"],np.atleast_1d(obs_time)))
                 
    return interpolated_positions