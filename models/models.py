from pydantic import BaseModel, HttpUrl
from typing import List

class ImageURL(BaseModel):
    url: HttpUrl

class ShelfCounts(BaseModel):
    shelf_number: int
    ramy_count: int
    other_count: int

class AnalysisResponse(BaseModel):
    total_products: int
    total_ramy: int
    shelves: List[ShelfCounts]
class ClassificationCounts(BaseModel):
    class_1_count: int
    class_2_count: int
    class_3_count: int
    total_ramy_products: int    