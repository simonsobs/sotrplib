"""
Models for sources
"""

from pydantic import BaseModel
from datetime import datetime

class PotentialSource(BaseModel):
    ra: float
    dec: float
    time: datetime

class SiftedSource(BaseModel):
    ra: float
    dec: float
    time: datetime
    confidence: float

class RemovedSource(BaseModel):
    ra: float
    dec: float
    time: datetime
    reason: str
    confidence: float