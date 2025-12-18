# app/services/external_service.py
import httpx
from drone_sys.app.utils.logging import logger


async def call_external_http(url: str, payload: dict, timeout: float = 3.0) -> dict:
    """
    统一封装外部 HTTP 调用，方便后期加重试 / 监控 / 超时控制。
    """
    logger.debug(f"Calling external service: {url}")
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        logger.debug(f"External service response: {data}")
        return data