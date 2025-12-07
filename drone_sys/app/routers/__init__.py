# drone_sys/app/routers/__init__.py
from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path
from typing import List

from fastapi import APIRouter


def get_all_routers() -> List[APIRouter]:
    """
    自动遍历本包下所有模块，收集其中名为 `router` 的 APIRouter 实例。
    只要你在 routers 目录里新建一个 xxx.py，里面定义 `router` 变量，
    就会自动被 main.py 注册。
    """
    routers: List[APIRouter] = []

    # 当前包名，比如 'drone_sys.app.routers'
    package_name = __name__
    # 当前目录路径
    package_path = Path(__file__).resolve().parent

    for module_info in iter_modules([str(package_path)]):
        module_name = module_info.name

        # 跳过私有模块和非业务文件
        if module_name.startswith("_") or module_name.startswith("."):
            continue

        # 导入模块，比如 drone_sys.app.routers.health
        full_module_name = f"{package_name}.{module_name}"
        module = import_module(full_module_name)

        # 如果模块里有名为 router 的变量且是 APIRouter，就收集起来
        router = getattr(module, "router", None)
        if isinstance(router, APIRouter):
            routers.append(router)

    return routers