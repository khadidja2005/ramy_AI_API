from fastapi import APIRouter
from Controlers.Segmentation.controler import load_model
from pydantic import BaseModel, HttpUrl

from fastapi import APIRouter, HTTPException
from models.models import ImageURL, AnalysisResponse , ClassificationCounts


from Controlers.Segmentation.controler import analyze_products , classify_products

router_seg = APIRouter()


@router_seg.post("/analyze_shelves/", response_model=AnalysisResponse)
async def analyze_shelves_endpoint(image_url: ImageURL):
    try:
        return await analyze_products(image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router_seg.post("/classify_products/", response_model=ClassificationCounts)
async def classify_products_endpoint(image_url: ImageURL):
    try:
        return await classify_products(image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))