"""
Models from the source catalog.
"""

from pydantic import BaseModel

class CatalogSource(BaseModel):
    id: str
    ra: float
    dec: float