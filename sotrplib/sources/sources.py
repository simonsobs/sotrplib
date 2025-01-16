from pydantic import BaseModel

class SourceCandidate(BaseModel):
    ra: float
    dec: float
    flux: float
    dflux: float
    snr: float
    freq: str
    ctime: float
    arr: str
    sourceID: str=''
    match_filtered: bool=False
    renormalized: bool=False
    catalog_crossmatch: bool=False
    crossmatch_name : str=''
    kron_flux:float=None
    kron_fluxerr:float=None
    kron_radius:float=None
    ellipticity:float=None
    elongation:float=None
    fwhm:float=None
def hello():
    return 'hello'