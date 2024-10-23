from pydantic import BaseModel

class SourceCandidate(BaseModel):
    ra: float
    dec: float
    flux: float
    dflux: float
    snr: float
    ctime: float
    sourceID: str
    match_filtered: bool
    renormalized: bool
    catalog_crossmatch: bool
