## this will define a class which contains the information related to the observatory such as 
## observing frequencies, wafer names, beam widths, band pass info, observatory location, etc.


## the plan is to migrate most fo the constants in utils which used to live in inputs.py
## AF 23 Oct 24

from typing import Optional
import numpy as np
import json


## Example usage:
'''
import os
cwd = os.getcwd()
with open(os.path.join(cwd,'observatory_information.json'), 'r') as file:
    content = file.read()
    observatory_info = json.loads(content)

observatory_name_alias = {'SPT':'South Pole Telescope',
                         'ACT':'Atacama Cosmology Telescope',
                         'SO':'Simons Observatory'
                         }
instrument_name_alias = {'SPT':'SPT3G',
                         'ACT':'ACTpol',
                         'SATP3':'SAT3'
                         }




act=Observatory.from_name('ACT')
print(act)
actpol = Instrument.from_name(act,'ACTpol')

## get pa5 f150 beam fwhm
pa5_f150_fwhm=actpol.get_fwhm('pa5','f150')
'''


class Observatory:
    def __init__(self, 
                name:str, 
                location:str, 
                latitude:float, 
                longitude:float, 
                elevation:float, 
                observing_bands:list[str], 
                primary_aperture:float, 
                instruments:list[str]
                ):
        
        self.name = name
        self.location = location
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.observing_bands = observing_bands
        self.primary_aperture = primary_aperture
        self.instruments = instruments
        

    @classmethod
    def from_name(cls, name):
        # Search for the observatory with the given name in the JSON data
        for obs in observatory_info["observatories"]:
            if obs["name"] == name or obs["name"]==observatory_name_alias[name]:
                return cls(
                    name=obs["name"],
                    location=obs["location"],
                    latitude=obs["latitude"],
                    longitude=obs["longitude"],
                    elevation=obs["elevation"],
                    observing_bands=obs["observing_bands"],
                    primary_aperture=obs["primary_aperture"],
                    instruments=obs["instruments"]
                )
        return None  # Return None if no observatory with the given name is found

    def __str__(self):
        return f"Observatory: {self.name}, Location: {self.location}, Elevation: {self.elevation}m"


class Instrument:
    def __init__(self,
                 observatory:Observatory,
                 instrument_name:str,
                 latitude:float, 
                 longitude:float, 
                 elevation:float, 
                 eff_band_centers:list[float],
                 wafers:list[str],
                 observing_bands:list[str],
                 primary_aperture:float, 
                 beam_fwhm: list[float],
                 beam_omega: list[float]
                 ):
        self.instrument_name = instrument_name
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.observing_bands = observing_bands
        self.primary_aperture = primary_aperture
        self.eff_band_centers = eff_band_centers
        self.beam_fwhm = beam_fwhm
        self.beam_omega = beam_omega

    @classmethod
    def from_name(cls, observatory, instrument_name):
        # Search for the observatory with the given name in the JSON data
        for inst in observatory.instruments:
            if inst.name == instrument_name or inst["name"]==instrument_name_alias[instrument_name]:
                return cls(observatory=observatory.name,
                           instrument_name=inst['name'],
                           latitude=inst['latitude'],
                           longitude=inst["longitude"],
                           elevation=inst["elevation"],
                           observing_bands=inst["observing_bands"],
                           primary_aperture=inst["primary_aperture"],
                           eff_band_centers=inst['eff_band_centers'],
                           wafers=inst['wafers'],
                           beam_fwhm=inst['beam_fwhm'],
                           beam_omega=inst['beam_omega']
                         )
        return None  # Return None if no observatory with the given name is found

    def __str__(self):
        return f"Observatory: {self.observatory}, Location: {self.location}, Elevation: {self.elevation}m"
    
    def get_fwhm(self,
                 wafer:str,
                 band:str
                 )->Optional[float]:
        
        if wafer in self.beam_fwhm:
            if band in self.beam_fwhm[wafer]:
                return self.beam_fwhm[wafer][band]
        
        if wafer not in self.beam_omega:
            return None
        if band in self.beam_omega[wafer]:
            ## assume beam_omega in nsr
            return (self.beam_omega[wafer][band] * 1e-9 * 4 * np.log(2.0) / np.pi) ** 0.5 * 180.0 * 60.0 / np.pi
        
        return None
