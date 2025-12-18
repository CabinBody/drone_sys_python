# app/schemas/fusion.py
from typing import List, Literal, Optional
from pydantic import BaseModel


class SingleSourcePoint(BaseModel):
    """
    单源单帧数据示例（你以后可以按实际字段扩展，比如 lat/lon/alt/conf 等）
    """
    source: Literal["radar", "5g", "tdoa", "gps"]
    timestamp: int
    x: float
    y: float
    z: float
    conf: Optional[float] = 1.0


class FusionRequest(BaseModel):
    """
    多源多帧融合请求：
    - 可以是一段时间窗口内某个目标的多源点集
    """
    uav_id: str
    points: List[SingleSourcePoint]


class FusionResultPoint(BaseModel):
    """
    模型输出的一帧融合结果
    """
    timestamp: int
    x: float
    y: float
    z: float


class FusionResponse(BaseModel):
    """
    接口统一响应结构，可以后续加 code / msg 等。
    """
    uav_id: str
    fused_traj: List[FusionResultPoint]