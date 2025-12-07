# app/services/fusion_service.py
from typing import List
import numpy as np

from drone_sys.app.schemas.fusion import (
    FusionRequest,
    FusionResponse,
    FusionResultPoint,
)
from drone_sys.app.core.config import settings
from drone_sys.app.services.external_service import call_external_http
from drone_sys.app.utils.logging import logger


class FusionService:
    """
    融合业务封装：
    - 可以在这里加载模型 / 做预处理 / 调外部服务 / 后处理
    """

    def __init__(self):
        # TODO: 可以在这里加载模型，例如 self.model = load_model(settings.MODEL_PATH)
        logger.info("FusionService initialized. (这里可以加载模型)")

    async def run_fusion(self, req: FusionRequest) -> FusionResponse:
        """
        示例流程：
        1. 对输入进行简单聚合（这里用均值示意）
        2. 可选：调用外部 HTTP 服务做推理
        3. 返回融合轨迹
        """

        # 1）简单按 timestamp 聚合：这里用 numpy 做个假融合
        by_ts = {}
        for p in req.points:
            by_ts.setdefault(p.timestamp, []).append([p.x, p.y, p.z])

        fused_traj: List[FusionResultPoint] = []
        for ts, pts in by_ts.items():
            arr = np.array(pts, dtype=float)
            mean_xyz = arr.mean(axis=0)
            fused_traj.append(
                FusionResultPoint(
                    timestamp=ts,
                    x=float(mean_xyz[0]),
                    y=float(mean_xyz[1]),
                    z=float(mean_xyz[2]),
                )
            )

        # 2）可选：调用外部 HTTP 服务做进一步处理（示例）
        # external_resp = await call_external_http(
        #     f"{settings.FUSION_ENGINE_BASE_URL}/post_process",
        #     payload={"uav_id": req.uav_id, "traj": [p.dict() for p in fused_traj]},
        # )

        logger.info(f"Fusion done for uav_id={req.uav_id}, frames={len(fused_traj)}")

        return FusionResponse(
            uav_id=req.uav_id,
            fused_traj=sorted(fused_traj, key=lambda p: p.timestamp),
        )