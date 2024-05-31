from pydantic import BaseModel
from typing import List

class Request(BaseModel):
    query: str

class Response(BaseModel):
    text: str
    links: List[str]

class ValidationError(BaseModel):
    loc: List[str]
    msg: str
    type: str

class HTTPValidationError(BaseModel):
    detail: List[ValidationError]
