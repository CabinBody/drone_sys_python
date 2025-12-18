# app/utils/logging.py
import logging

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

logger = logging.getLogger("uav-service")