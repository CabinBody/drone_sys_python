#!/usr/bin/env bash
set -e

############### 配置项（按需改动） ###############

# 服务器地址 & 用户
SERVER_USER="ubuntu"
SERVER_HOST="81.70.81.41"
# Hello@123456

# 本地项目路径（当前目录）
PROJECT_DIR="$(pwd)"

# 上传到服务器的路径
REMOTE_DIR="/opt/drone_sys_python"

# 压缩包名称
TAR_NAME="drone_sys_python.tar.gz"

############### 本地打包 ###############

echo "📦 压缩项目：$PROJECT_DIR → $TAR_NAME"
tar -czf "$TAR_NAME" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='logs' \
    --exclude='.idea' \
    .

############### 上传到服务器 ###############

echo "📤 上传到服务器：$SERVER_HOST:$REMOTE_DIR"
ssh "$SERVER_USER@$SERVER_HOST" "mkdir -p $REMOTE_DIR"
scp "$TAR_NAME" "$SERVER_USER@$SERVER_HOST:$REMOTE_DIR/"

############### 远程执行 deploy.sh ###############

echo "🚀 触发服务器部署脚本"
ssh "$SERVER_USER@$SERVER_HOST" "cd $REMOTE_DIR && chmod +x deploy.sh && ./deploy.sh"

echo "🎉 部署完成！"
