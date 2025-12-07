#!/usr/bin/env bash
set -e

##################### 配置 #####################

# 项目根目录
PROJECT_DIR="/opt/drone_sys_python"

# 上传后的压缩包名称（和 upload.sh 保持一致）
TAR_NAME="drone_sys_python.tar.gz"

# 虚拟环境目录
VENV_DIR="$PROJECT_DIR/venv"

# 日志 & PID 文件
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$PROJECT_DIR/uvicorn.pid"

# Python 版本（你说服务器用 3.12）
PYTHON_BIN="python3.12"

# Uvicorn 配置
APP_MODULE="drone_sys.app.main:app"
HOST="127.0.0.1"   # 只在本机监听，给 Nginx 反向代理用
PORT=8001          # 内部端口，Nginx 转发 /droneFusion 到这里

##################### 解压 #####################

echo "📦 解压到 $PROJECT_DIR"
cd "$PROJECT_DIR"

echo "🧹 清理旧代码（保留 deploy.sh / $TAR_NAME / venv）"
find "$PROJECT_DIR" -mindepth 1 -maxdepth 1 \
  ! -name "deploy.sh" \
  ! -name "$TAR_NAME" \
  ! -name "venv" \
  -exec rm -rf {} \;

echo "📦 解压新代码：$TAR_NAME"
tar -xzf "$TAR_NAME"

##################### Python 3.12 环境 #####################

mkdir -p "$LOG_DIR"

if ! command -v "$PYTHON_BIN" &> /dev/null; then
  echo "❌ 未找到 $PYTHON_BIN，请先在服务器安装（例如：apt install python3.12 python3.12-venv）"
  exit 1
fi

# 创建或复用 venv
if [ ! -d "$VENV_DIR" ]; then
  echo "🐍 使用 $PYTHON_BIN 创建虚拟环境：$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "👉 激活虚拟环境：$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "⬆️ 升级 pip"
pip install --upgrade pip

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "📦 安装依赖：requirements.txt"
  pip install -r "$PROJECT_DIR/requirements.txt" --no-cache-dir
else
  echo "⚠️ 未找到 requirements.txt，跳过依赖安装"
fi

##################### 停掉旧进程 #####################

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  OLD_PID=$(cat "$PID_FILE")
  echo "🛑 停止旧的 uvicorn：PID=$OLD_PID"
  kill "$OLD_PID" || true
  rm -f "$PID_FILE"
else
  echo "ℹ️ 无旧 uvicorn 进程"
fi

##################### 启动服务 #####################

echo "🚀 启动 uvicorn：$APP_MODULE ($HOST:$PORT)"
nohup uvicorn "$APP_MODULE" \
  --host "$HOST" \
  --port "$PORT" \
  >> "$LOG_DIR/server.log" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"

echo "🎉 部署成功，服务已启动：PID=$NEW_PID"
echo "📜 日志文件：$LOG_DIR/server.log"
echo "📘 内部访问: http://$HOST:$PORT/hello/world"
echo "🌐 Nginx 映射后访问: http://81.70.81.41:8080/droneFusion/hello/world"