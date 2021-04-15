from fastapi import APIRouter

from api import caricature, root

api_router = APIRouter()

api_router.include_router(root.router)
api_router.include_router(caricature.router, prefix="/caricature", tags=["caricature"])