# app/routers/fusion.py
from fastapi import APIRouter, Depends
from drone_sys.app.schemas.fusion import FusionRequest, FusionResponse
from drone_sys.app.services.fusion_service import FusionService
from drone_sys.app.dependencies.common import get_fusion_service

router = APIRouter()


@router.post("/run", response_model=FusionResponse)
async def run_fusion(
    req: FusionRequest,
    fusion_service: FusionService = Depends(get_fusion_service),
):
    """
    多源融合接口示例：
    - 输入多帧轨迹数据（可以是 radar / 5g / tdoa 等）
    - 调用服务层做算法推理
    """
    result = await fusion_service.run_fusion(req)
    return result