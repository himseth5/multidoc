from pydantic import BaseModel, ValidationError
from typing import List
class loinccode(BaseModel):
    """Data Model for describing the column"""
    loinc_code: str
    # column: str
    # Type: str
class loinc(BaseModel):
    """Data Model for describing the columns"""
    loinc_codes: List[loinccode]