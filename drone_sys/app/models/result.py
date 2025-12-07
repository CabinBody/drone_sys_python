# app/models/inference_result.py
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Result:
    """
    示例：统一表示一次推理结果的数据结构。
    - 不依赖 FastAPI / Pydantic，纯 Python 业务内使用
    """
    uav_id: str
    traj: List[Tuple[float, float, float]]  # [(x,y,z), ...]