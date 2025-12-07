# app/routers/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/ping")
async def ping():
    """
    简单健康检查接口，用于 K8s / 网关探活。
    """
    return {"status": "ok"}