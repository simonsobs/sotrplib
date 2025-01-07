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
    sources = sourcecat[sourcecat["fluxJy"] > (flux_threshold)]
    sources['RADeg'][sources["RADeg"]<0]+=360.
    print(len(sources["decDeg"]), 'sources above flux threshold %.1f mJy'%(flux_threshold*1000))
    return sources