from pydantic import BaseModel


class SourceCandidate(BaseModel):
    ra: float
    dec: float
    err_ra:float
    err_dec:float
    flux: float
    err_flux: float
    snr:float
    freq:str
    ctime:float
    arr:str
    map_id:str=''
    sourceID:str=''
    source_type:str=''
    matched_filtered:bool=False
    renormalized:bool=False
    catalog_crossmatch:bool=False
    crossmatch_names:list=[]
    crossmatch_probabilities:list=[]
    fit_type:str='forced'
    fwhm_a:float=None
    fwhm_b:float=None
    err_fwhm_a:float=None
    err_fwhm_b:float=None
    orientation:float=None
    kron_flux:float=None
    kron_fluxerr:float=None
    kron_radius:float=None
    ellipticity:float=None
    elongation:float=None
    fwhm:float=None
    

    def update_crossmatches(self, 
                            match_names:list, 
                            match_probabilities:list=None,
                           ):
        """
        Update the crossmatch names and probabilities with new matches.
        """
        self.crossmatch_names.extend(match_names)
        if match_probabilities is not None:
            self.crossmatch_probabilities.extend(match_probabilities)
        else:
            self.crossmatch_probabilities.extend([None]*len(match_names))
        
        return
