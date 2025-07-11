import numpy as np
from pixell import enmap
from pixell.utils import degree


def generate_random_positions_in_map(n:int,
                                     imap:enmap.ndmap
                                     ):
    """
    Generate n random positions uniformly distributed in the map.
    Uses the mapshape, to get random pixels, then calculates ra,dec
    using the imap.pix2sky.

    Arguments:
        n (int): Number of positions to generate.
        imap (enmap.ndmap): Input map for generating random positions.

    Returns:
        tuple: Two numpy arrays containing the RA and Dec positions.
    """
    x = np.random.uniform(0,imap.shape[0],n)
    y = np.random.uniform(0,imap.shape[1],n)
    positions = []
    for i in range(n):
        positions.append(tuple(imap.pix2sky((x[i],y[i]))/degree))
    
    return positions


def generate_random_positions(n:int, 
                              imap:enmap.ndmap=None,
                              ra_lims:tuple=None, 
                              dec_lims:tuple=None,
                              ):
    """
    Generate n random positions in RA and Dec uniformly distributed on the sphere
    within the given limits.

    Arguments:
        n (int): Number of positions to generate.
        imap (enmap.ndmap): Input map for generating random positions. If None, ra_lims and dec_lims must be provided.
        ra_lims (tuple): Limits for RA in degrees (min_ra, max_ra).
        dec_lims (tuple): Limits for Dec in degrees (min_dec, max_dec).

    Returns:
        array: zipped array of tuples containing (dec,ra) pairs.
    """
    if ra_lims is None or dec_lims is None:
        if imap is not None:
            shape, wcs = imap.shape, imap.wcs
            dec_min, dec_max = enmap.box(shape, wcs)[:, 0]/degree
            ra_min, ra_max = enmap.box(shape, wcs)[:, 1]/degree
            ra_lims = (ra_min, ra_max)
            dec_lims = (dec_min, dec_max)
        else:
            raise ValueError("Either ra_lims and dec_lims must be provided, or imap must be supplied.")
    ## assume that if limits are something like (350,10) that the ra limits wrap 0, so -10,10
    if ra_lims[0] > ra_lims[1]:
        if ra_lims[0]>180:
            ra_lims[0]-=360
    # Convert limits to radians
    ra_min, ra_max = np.radians(ra_lims)
    dec_min, dec_max = np.radians(dec_lims)

    # Generate RA uniformly between ra_lims
    ra = np.random.uniform(ra_min, ra_max, n)

    # Generate Dec uniformly on the sphere
    z_min = np.sin(dec_min)
    z_max = np.sin(dec_max)
    z = np.random.uniform(z_min, z_max, n)
    dec = np.arcsin(z)

    # Convert back to degrees
    ra = np.degrees(ra)%360
    dec = np.degrees(dec)
    positions = list(zip(dec, ra))
    return positions


def generate_random_flare_times(n:int,
                                start_time:float|str=1.4e9,
                                end_time:float|str=1.7e9,
                               ):
    """
    Generate n random flare times uniformly distributed between start_time and end_time.
    Arguments:
        n (int): Number of flare times to generate.
        start_time (float | str): Start time in unix timestamp or string formatted as "yyyy-mm-dd HH:MM:SS".
        end_time (float | str): End time in unix timestamp or string formatted as "yyyy-mm-dd HH:MM:SS".
    Returns:
        numpy array: Array of random flare times.
    """
    from datetime import datetime
    # Convert string times to unix timestamps if necessary
    if isinstance(start_time, str):
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp()
    if isinstance(end_time, str):
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp()

    return np.random.uniform(start_time, end_time, n)


def generate_random_flare_widths(n:int,
                                  min_width:float=0.1,
                                  max_width:float=10.0,
                                  ):
    """
    Generate n random flare widths uniformly distributed between min_width and max_width.

    Arguments:
        n (int): Number of flare widths to generate.
        min_width (float): Minimum flare width.
        max_width (float): Maximum flare width.

    Returns:
        numpy array: Array of random flare widths.
    """
    return np.random.uniform(min_width, max_width, n)


def generate_random_flare_amplitudes(n:int,
                                     min_amplitude:float=0.1,
                                     max_amplitude:float=10.0,
                                     ):
    """
    Generate n random flare amplitudes uniformly distributed between min_amplitude and max_amplitude.

    Arguments:
        n (int): Number of flare amplitudes to generate.
        min_amplitude (float): Minimum flare amplitude.
        max_amplitude (float): Maximum flare amplitude.

    Returns:
        numpy array: Array of random flare amplitudes.
    """
    return np.random.uniform(min_amplitude, max_amplitude, n)


def make_gaussian_flare(unix_times:np.ndarray, 
                        flare_peak_time:float=0.0, 
                        flare_fwhm_s:float=1.0, 
                        flare_peak_Jy:float=0.1,
                        ):
        """
        Generate a Gaussian flare sampled at the given unix_times.

        Arguments:
            unix_times (numpy array): Array of unix timestamps to sample the flare.
            flare_peak_time (float): Unix timestamp of the flare peak.
            flare_fwhm_s (float): Full width at half maximum (FWHM) of the flare in seconds.
            flare_peak_Jy (float): Amplitude of the flare in Jy.

        Returns:
            numpy array: Array of flare amplitudes sampled at unix_times.
        """
        # Convert FWHM to standard deviation
        sigma = flare_fwhm_s / (2 * np.sqrt(2 * np.log(2)))

        # Compute the Gaussian flare
        flare = flare_peak_Jy * np.exp(-0.5 * ((unix_times - flare_peak_time) / sigma) ** 2)

        return flare


def convert_photutils_qtable_to_json(params,
                                     imap:enmap.ndmap=None,
                                    ):
    '''
    photutils params is an Astropy QTable with:
    id,x_0, y_0, flux, fwhm_x,fwhm_y, theta
    
    if imap supplied, will also get ra,dec
    '''
    from pixell.utils import degree
    json_cat_out = {}
    for p in params:
        for k in p.keys():
            if k not in json_cat_out:
                json_cat_out[k] = []
            json_cat_out[k].append(p[k])
        
        if isinstance(imap,enmap.ndmap):
            dec,ra=imap.pix2sky([p['y_0'],p['x_0']])/degree
        else:
            ra,dec = np.nan,np.nan
        if 'RADeg' not in json_cat_out:
            json_cat_out['RADeg'] = []
            json_cat_out['decDeg'] = []
            json_cat_out['name'] = []
            
        json_cat_out['RADeg'].append(ra)
        json_cat_out['decDeg'].append(dec)
        json_cat_out['name'].append(str(p['id']))
    for key in json_cat_out:
        json_cat_out[key] = np.array(json_cat_out[key])
    flux = json_cat_out.pop('flux')
    json_cat_out['err_fluxJy'] = imap.std() * np.ones_like(flux)
    json_cat_out['fluxJy'] = flux
    json_cat_out['forced'] = np.ones_like(flux)
    
    return json_cat_out


def ra_lims_valid(ra_lims:tuple=None):
    """
    Check if the RA limits are valid.

    Arguments:
        ra_lims (tuple): Tuple containing the RA limits (min_ra, max_ra).

    Returns:
        bool: True if the RA limits are valid, False otherwise.
    """
    if ra_lims is None:
        return False
    if len(ra_lims) != 2:
        return False
    if ra_lims[0] < 0 or ra_lims[1] > 360:
        return False
    
    return True


def dec_lims_valid(dec_lims:tuple=None):
    """
    Check if the Dec limits are valid.

    Arguments:
        dec_lims (tuple): Tuple containing the Dec limits (min_dec, max_dec).

    Returns:
        bool: True if the Dec limits are valid, False otherwise.
    """
    if dec_lims is None:
        return False
    if len(dec_lims) != 2:
        return False
    if dec_lims[0] < -90 or dec_lims[1] > 90:
        return False
    
    return True


def save_transients_to_db(transients, db_path):
    """
    Save a list of SimTransient objects to a SQLite database.

    Parameters:
    - transients: List of SimTransient objects to save.
    - db_path: Path to the SQLite database file.
    """
    import sqlite3
    import pickle as pk
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position BLOB,
            peak_amplitude REAL,
            peak_time REAL,
            flare_width REAL,
            flare_morph TEXT,
            beam_params BLOB
        )
    ''')
    for transient in transients:
        cursor.execute('''
            INSERT INTO transients (position, peak_amplitude, peak_time, flare_width, flare_morph, beam_params)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            pk.dumps((transient.dec, transient.ra)),
            transient.peak_amplitude,
            transient.peak_time,
            transient.flare_width,
            transient.flare_morph,
            pk.dumps(transient.beam_params)
        ))
    conn.commit()
    conn.close()


def load_transients_from_db(db_path):
    """
    Load a list of SimTransient objects from a SQLite database.

    Parameters:
    - db_path: Path to the SQLite database file.

    Returns:
    - List of SimTransient objects.
    """
    import sqlite3
    import pickle as pk
    from .sim_sources import SimTransient
    if not db_path:
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT position, peak_amplitude, peak_time, flare_width, flare_morph, beam_params FROM transients')
    rows = cursor.fetchall()
    conn.close()

    transients = []
    for row in rows:
        position = pk.loads(row[0])
        beam_params = pk.loads(row[5])
        transient = SimTransient(
            position=position,
            peak_amplitude=row[1],
            peak_time=row[2],
            flare_width=row[3],
            flare_morph=row[4],
            beam_params=beam_params
        )
        transients.append(transient)
    return transients


def load_config_yaml(config_path:str):
    """
    Load a configuration file in YAML format.

    Parameters:
    - config_path: Path to the YAML configuration file.

    Returns:
    - Dictionary containing the configuration parameters.
    """
    import yaml
    import os
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_sim_map_group(sim_params):
    freq_arr_idx = sim_params['array_info']['arr']+'_'+sim_params['array_info']['freq']
    indexed_map_groups = {freq_arr_idx:[[i] for i in np.arange(sim_params['maps']['n_realizations'])]}
    indexed_map_group_time_ranges = {freq_arr_idx:[[t] for t in np.linspace(float(sim_params['maps']['min_time']),float(sim_params['maps']['max_time']),int(sim_params['maps']['n_realizations']))]}
    return freq_arr_idx, indexed_map_groups, indexed_map_group_time_ranges


def make_2d_gaussian_model_param_table(imap:enmap.ndmap,
                                      sources:list,
                                      nominal_fwhm_arcmin:float=2.2,
                                      verbose:bool=False,
                                      cuts={}
                                     ):
    """
    Create a 2D Gaussian model parameter table from the sources detected in the image.
    The table contains the parameters for each source, including its position, flux, and FWHM.
    This function also applies cuts to filter out poor fits.
    The function returns an astropy Qtable compatible with photutils psf_image generation.
    """
    from astropy.table import QTable
    from pixell.utils import degree, arcmin
    
    model_params = {'x_0':[],
                    'y_0':[],
                    'x_fwhm':[],
                    'y_fwhm':[],
                    'flux':[],
                    'theta':[],
                    'id':[]
                    }
    id_num = 0
    res_arcmin = abs(imap.wcs.wcs.cdelt[0] * degree / arcmin)

    for i in range(len(sources)):
        s = sources[i]
        pix = imap.sky2pix(np.array([s.dec,s.ra])*degree)
        
        ## unfortunate naming, a,b are semi-major and semi-minor axes, 
        ## but really they're x,y from photutils gaussian fit.
        cut=False
        for cutkey in cuts.keys():
            if cutkey not in s.__dict__.keys():
                raise ValueError(f"Cut {cutkey} not found in source attributes.")
            if s.__dict__[cutkey] is None:
                cut=True
                break
            if np.isnan(s.__dict__[cutkey]) or (s.__dict__[cutkey] < cuts[cutkey][0]) or (s.__dict__[cutkey] > cuts[cutkey][1]):
                if verbose:
                    print(f"Source {i} failed cut {cutkey} with value {s.__dict__[cutkey]}")
                cut=True
                break
        if not cut:
            fwhm_x = s.fwhm_a
            fwhm_y = s.fwhm_b
            if isinstance(fwhm_x,type(None)):
                fwhm_x = nominal_fwhm_arcmin
            if isinstance(fwhm_y,type(None)):
                fwhm_y = nominal_fwhm_arcmin
            if isinstance(s.orientation,type(None)):
                s.orientation = 0.0
            
            omega_b = 1.02*(np.pi / 4 / np.log(2)) * (fwhm_x*fwhm_y / res_arcmin**2)
            # Define the PSF model parameters
            model_params['x_0'].append(pix[1])
            model_params['y_0'].append(pix[0])
            model_params['x_fwhm'].append(fwhm_x/res_arcmin)
            model_params['y_fwhm'].append(fwhm_y/res_arcmin)
            model_params['flux'].append(s.flux*omega_b)
            model_params['theta'].append(s.orientation)
            model_params['id'].append(id_num)
            id_num += 1
    if verbose:
        print(model_params)
    return QTable(model_params)