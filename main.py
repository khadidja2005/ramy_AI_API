from fastapi import FastAPI
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
from Routes.urls import router_seg
from Controlers.Segmentation.controler import load_model , load_classification_model
app = FastAPI()
os.environ["PORT"] = "8000"
os.environ["HOST"] = "0.0.0.0"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None



app.include_router(router_seg, prefix="/api" , tags=["shelf-analysis"]) , 

@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts"""
    try:
        load_model()
        load_classification_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
@app.get("/")
async def root ():
    return {"message" : "Hello World"}

