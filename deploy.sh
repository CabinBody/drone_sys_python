#!/usr/bin/env bash
set -e

##################### 配置 #####################

PROJECT_DIR="/home/drone_sys_python"
TAR_NAME="drone_sys_python.tar.gz"
VENV_DIR="$PROJECT_DIR/venv"
LOG_DIR="$PROJECT_DIR/logs"

PYTHON_BIN="python3.12"
APP_MODULE="drone_sys.app.main:app"
HOST="127.0.0.1"
PORT=8001

##################### 准备 #####################

echo "📦 工作目录：$PROJECT_DIR"
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

##################### 杀掉旧进程（不再依赖 PID 文件） #####################

echo "🔍 查找旧的 uvicorn 进程..."
OLD_PIDS=$(ps aux | grep "uvicorn $APP_MODULE" | grep -v grep | awk '{print $2}')

if [ -n "$OLD_PIDS" ]; then
  echo "🛑 发现旧进程：$OLD_PIDS"
  echo "🛑 正常 kill..."
  kill $OLD_PIDS || true
  sleep 1

  # 再查一次，有残留就直接 kill -9
  REMAIN_PIDS=$(ps aux | grep "uvicorn $APP_MODULE" | grep -v grep | awk '{print $2}')
  if [ -n "$REMAIN_PIDS" ]; then
    echo "⚠️ 发现残留进程（强制 kill -9）：$REMAIN_PIDS"
    kill -9 $REMAIN_PIDS || true
  fi
else
  echo "ℹ️ 未发现旧的 uvicorn 进程"
fi

echo "👌 uvicorn 已全部停止"

##################### 虚拟环境 #####################

if [ ! -d "$VENV_DIR" ]; then
  echo "🐍 创建虚拟环境：$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "👉 激活虚拟环境"
source "$VENV_DIR/bin/activate"

echo "⬆️ 升级 pip"
pip install --upgrade pip

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "📦 安装依赖"
  pip install -r "$PROJECT_DIR/requirements.txt" --no-cache-dir
else
  echo "⚠️ 未找到 requirements.txt（可能是空项目）"
fi

##################### 启动新的 uvicorn #####################

echo "🚀 启动 uvicorn：$APP_MODULE ($HOST:$PORT)"
nohup uvicorn "$APP_MODULE" \
  --host "$HOST" \
  --port "$PORT" \
  >> "$LOG_DIR/server.log" 2>&1 &

NEW_PID=$!
echo "🎉 新进程! PID：$NEW_PID"
echo "📜 日志文件：$LOG_DIR/server.log"
echo "📘 内部调试:  http://$HOST:$PORT/hello/world"
echo "🌐 外网访问:  http://cabinbody.cn/drone-fusion/hello/world"