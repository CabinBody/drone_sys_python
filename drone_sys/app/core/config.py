# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    全局配置：统一从环境变量 / .env 读取（你用 conda 也没关系）。
    """
    # 基本服务配置
    APP_NAME: str = "UAV Algorithm Service"
    DEBUG: bool = True

    # 例如：外部 HTTP 服务地址
    FUSION_ENGINE_BASE_URL: str = "http://localhost:8080"

    # 模型路径示例
    # MODEL_PATH: str = "models/graph_fusion.pt"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 全局配置单例，其他地方直接 from app.core.config import settings
settings = Settings()