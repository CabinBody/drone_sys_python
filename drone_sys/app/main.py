# drone_sys/app/main.py
from fastapi import FastAPI
from drone_sys.app.routers import get_all_routers


def create_app() -> FastAPI:
    app = FastAPI(
        title="UAV Algorithm Service",
        description="多源融合 / 算法推理 HTTP 服务",
        version="0.1.0",
    )

    #自动注册所有 routers
    for r in get_all_routers():
        app.include_router(r)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("drone_sys.app.main:app", host="0.0.0.0", port=8080, reload=True)