from pydantic import BaseModel

class SourceCandidate(BaseModel):
    ra: float
    dec: float
    flux: float
    dflux: float
    snr: float
    freq: str
    ctime: float
    sourceID: str=''
    match_filtered: bool=False
    renormalized: bool=False
    catalog_crossmatch: bool=False
    crossmatch_name : str=''
    ellipticity:float=None
    elongation:float=None
    fwhm:float=None
def hello():
    return 'hello'