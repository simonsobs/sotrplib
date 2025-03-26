from pixell.enmap import enmap

class SourceCatalog():
    def __init__(self,ra,dec):
        self.ra=ra
        self.dec=dec

def load_act_catalog(source_cat_file:str='/scratch/gpfs/SIMONSOBS/users/amfoster/depth1_act_maps/inputs/PS_S19_f090_2pass_optimalCatalog.fits',
                     flux_threshold:float=0
                    ):
    '''
    source_cat_file is path to source catalog

    flux_threshold is a threshold, in Jy, below which we ignore the sources.
        i.e. if we want sources in depth-1 map, we don't care about sub mJy sources.
        by default, returns all positive sources.
    '''
    from astropy.table import Table 
    sourcecat=None
    print("Extracting known sources from ACT catalog")
    sourcecat = Table.read(source_cat_file)
    sources = sourcecat[sourcecat["fluxJy"] >= (flux_threshold)]
    sources['RADeg'][sources["RADeg"]<0]+=360.
    print(len(sources["decDeg"]), 'sources above flux threshold %.1f mJy'%(flux_threshold*1000))
    out_dict = {key:sources[key] for key in sources.colnames}
    return out_dict


def convert_gauss_fit_to_source_cat(gauss_fits:list,
                                    uncert_prefix:str='err_'
                                    ):
    '''
    gauss fits is a list of dictionaries of the output params of the gaussian fitting.
    convert that into a dictionary of lists.

    since there are uncertainties on the fits, make keys err_[blah] for those fits.
    this is inspired by fluxJy and err_fluxJy in act table.
    '''
    sources = {}
    for i in range(len(gauss_fits)):
        for key in gauss_fits[i]:
            if key not in sources:
                sources[key] = []
                sources[uncert_prefix+key] = []
            if isinstance(gauss_fits[i][key],tuple):
                keyval,keyvaluncert = gauss_fits[i][key]
                sources[key].append(keyval)
                sources[uncert_prefix+key].append(keyvaluncert)
            else:
                keyval = gauss_fits[i][key]
                sources[key].append(keyval)
    popkeys = [k for k in sources if not sources[k]]
    for k in popkeys:
        sources.pop(k)
    
    return sources

def convert_json_to_act_format(json_list):
    '''
    json list is the list of dictionaries output to json file format.
    this will hopefully be depricated when using the database.

    there are other differences too, but these are the ones we care about now.
    '''
    import numpy as np
    sources = {}
    for i in range(len(json_list)):
        for key in json_list[i]:
            if key not in sources:
                sources[key]=[json_list[i][key]]
            else:
                sources[key].append(json_list[i][key])
    for key in sources:
        sources[key] = np.asarray(sources[key])

    sources['RADeg']=sources['ra']
    del sources['ra']
    
    sources['decDeg']=sources['dec']
    del sources['dec']

    sources['fluxJy'] = sources['flux']/1000.
    del sources['flux']

    sources['err_fluxJy'] = sources['dflux']/1000.
    del sources['dflux']

    sources['name'] = sources['crossmatch_name']
    del sources['crossmatch_name']
    
    return sources


def load_json_test_catalog(source_cat_file:str,
                           flux_threshold:float=0
                           ):
    '''
    Load the output file with SourceCandidate dictionaries 
    which is stored as a json file.

    flux_threshold is a threshold, in Jy, below which we ignore the sources.
        i.e. if we want sources in depth-1 map, we don't care about sub mJy sources.
        by default, returns all positive sources.
    '''
    import json
    with open(source_cat_file, 'r') as f:
        # Load the JSON data into a Python dictionary
        data = [json.loads(line) for line in f]
    
    sources = convert_json_to_act_format(data)
    sources['RADeg'][sources["RADeg"]<0]+=360.

    flux_cut = sources['fluxJy']>=flux_threshold
    for key in sources:
        sources[key] = sources[key][flux_cut]

    return sources

def load_websky_csv_catalog(source_cat_file:str,
                            flux_threshold:float=0
                            ):
    '''
    load the websky catalog from a csv containing the columns
    flux(Jy), ra(deg), dec(deg)
    '''
    from numpy import loadtxt, asarray
    from ..utils.utils import radec_to_str_name
    print('loading websky catalog')
    websky_flux,websky_ra,websky_dec = loadtxt(source_cat_file,
                                               delimiter=',',
                                               unpack=True,
                                               skiprows=1
                                              )
    websky_ra[websky_ra>180.]-=360
    inds = websky_flux>flux_threshold
    print(sum(inds),' sources above %.0f mJy'%(flux_threshold*1000))
    sources = {}
    sources['RADeg'] = websky_ra[inds]
    sources['decDeg'] = websky_dec[inds]
    sources['fluxJy'] = websky_flux[inds]
    sources['err_fluxJy'] = websky_flux[inds]*0.0
    sources['name'] = asarray([radec_to_str_name(sources['RADeg'][i],sources['decDeg'][i]) for i in range(sum(inds))])
    return sources


def load_pandas_catalog(source_cat_file:str,
                        flux_threshold:float=0
                        ):
    '''
    load the source catalog from a pandas dataframe stored in a pickle file.
    '''
    import pandas as pd
    print('loading pandas catalog')
    sources = pd.read_pickle(source_cat_file)
    sources['RADeg'][sources["RADeg"]<0]+=360.
    flux_cut = sources['fluxJy']>=flux_threshold
    sources = sources[flux_cut]
    sources['name'] = sources['sourceID']
    return sources  


def load_catalog(source_cat_file:str,
                 flux_threshold:float=0,
                 mask_outside_map:bool=False,
                 mask_map:enmap=None
                 ):
    '''
    flux_threshold is a threshold, in Jy, below which we ignore the sources.
        i.e. if we want sources in depth-1 map, we don't care about sub mJy sources.
        by default, returns all positive sources.
    mask_outside_map: bool
        if False, do not mask the mapped region, just include all sources in catalog.
    mask_map: enmap
        map with which to do the masking; can be anything that is zero/nan outside observed region.

    Returns:

        sources: source catalog, in astropy.table.table.Table or dict format... dumb but works for now.

    '''

    if '.pkl' in source_cat_file:
        sources=load_pandas_catalog(source_cat_file=source_cat_file,
                                    flux_threshold=flux_threshold
                                    )

    if 'PS_S19_f090_2pass_optimalCatalog.fits' in source_cat_file or 'catmaker' in source_cat_file:
        sources = load_act_catalog(source_cat_file=source_cat_file,
                                   flux_threshold=flux_threshold
                                   )
    if 'websky_cat_100_1mJy.csv' in source_cat_file:
        sources = load_websky_csv_catalog(source_cat_file=source_cat_file,
                                          flux_threshold=flux_threshold
                                         )
    if '.json' in source_cat_file:
        sources = load_json_test_catalog(source_cat_file=source_cat_file,
                                         flux_threshold=flux_threshold
                                         ) 
    if mask_outside_map and not isinstance(mask_map,type(None)):
        from ..sources.finding import mask_sources_outside_map
        source_mask = mask_sources_outside_map(sources,
                                               mask_map
                                              )
        if isinstance(sources,dict):
            for key in sources:
                sources[key]  = sources[key][source_mask]
        else:
            sources = sources[source_mask]

    return sources


def write_json_catalog(outcat,
                       out_dir:str='./',
                       out_name:str='source_catalog.json'
                       ):
    with open(out_dir+out_name,'w') as f:
        for oc in outcat:
            json_string_cand = oc.json()
            f.write(json_string_cand)
            f.write('\n')
    return
