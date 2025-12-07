# drone_sys/app/routers/helloworld.py
from fastapi import APIRouter

# 必须叫 router（自动加载器要找这个变量）
router = APIRouter(
    prefix="/hello",    # 路由前缀：你的接口是 /hello/world
    tags=["Hello"],     # 文档分组名
)

@router.get("/world")
async def hello_world():
    """
    一个最简单的测试接口。
    """
    return {"message": "Hello, World!"}