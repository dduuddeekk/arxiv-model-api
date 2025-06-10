from typing import List, Optional
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    abstract: str
    url: str
    published: str
    authors: Optional[List[str]] = []

class SearchResponse(BaseModel):
    success: bool
    code: str
    message: str
    data: List[SearchResult]
