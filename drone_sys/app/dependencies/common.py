# app/dependencies/common.py
from functools import lru_cache
from drone_sys.app.services.fusion_service import FusionService


@lru_cache()
def get_fusion_service() -> FusionService:
    """
    用 lru_cache 模拟简单的单例：
    - FastAPI 的 Depends 会调用这个函数，但真正的实例只会初始化一次
    - 适合放模型、连接池等重资源
    """
    return FusionService()