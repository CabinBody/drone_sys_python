#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_call import call_qwen_model
import threading

TOTAL_TOKENS = 0
TOKEN_LOCK = threading.Lock()

# =========================================================
# Config
# =========================================================

PROMPTS_JSONL = "./sft_build/prompts.jsonl"     # 你生成的 prompts 文件
OUT_DIR = "./sft_build"
OUTPUTS_JSONL = os.path.join(OUT_DIR, "llm_outputs.jsonl")  # 强模型原始输出落盘（可断点续跑）
SFT_JSONL = os.path.join(OUT_DIR, "sft.jsonl")              # 最终 SFT 数据集

# 并发与重试
MAX_WORKERS = 8
MAX_RETRIES = 4
RETRY_BASE_SLEEP = 1.0  # 秒

# 是否每次都重新调用（False=断点续跑，已存在 sample_id 就跳过）
FORCE_RECALL = False

# 只抽样多少条（None=全部）
LIMIT: Optional[int] = None

# =========================================================
# Output validation
# =========================================================
FIRST_LINE_RE = re.compile(r"^(?P<id>[A-Za-z0-9_]+)，请(悬停|原地降落|返航|正常飞行)$")

def validate_llm_answer(uav_id: str, ans: str) -> Tuple[bool, str]:
    """
    你的最终输出约束：
    - 两行
    - 第一行：{uav_id}，请 + 四选一
    - 第二行：理由: + <=50字
    """
    if ans is None:
        return False, "empty"

    lines = [ln.strip() for ln in ans.strip().splitlines() if ln.strip() != ""]
    if len(lines) != 2:
        return False, "not_two_lines"

    m = FIRST_LINE_RE.match(lines[0])
    if not m:
        return False, "bad_first_line"

    if m.group("id") != uav_id:
        return False, "uav_id_mismatch"

    if not lines[1].startswith("理由:"):
        return False, "bad_reason_prefix"

    return True, "ok"


# =========================================================
# I/O Helpers
# =========================================================

def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_existing_outputs(path: str) -> Dict[str, Dict[str, Any]]:
    """
    读取已有 outputs.jsonl，返回 {sample_id: record}
    """
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("sample_id")
            if sid:
                out[sid] = r
    return out

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================================================
# Strong model API call (YOU IMPLEMENT THIS)
# =========================================================
def call_strong_model(prompt: str) -> str:
    """
    使用你封装好的 call_qwen_model(prompt)
    返回强模型输出文本，并统计 token 消耗
    """
    global TOTAL_TOKENS

    # === 调用你的强模型 ===
    res, tokens = call_qwen_model(prompt)

    # === 累计 token（线程安全）===
    with TOKEN_LOCK:
        TOTAL_TOKENS += tokens
        total_now = TOTAL_TOKENS

    # === 打印本次消耗 ===
    print(f"[LLM] tokens_used={tokens}, total_tokens={total_now}")

    # === 返回模型文本结果 ===
    # ⚠️ 假设 res 就是模型直接输出的字符串
    # 如果 res 是 dict / response 对象，请在这里取 text
    return res.strip()



# =========================================================
# Worker
# =========================================================
def label_one(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    对单条 prompt 调用强模型，做重试，返回要落盘的输出记录
    """
    sid = row["sample_id"]
    uav_id = row["uav_id"]
    prompt = row["prompt"]

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ans = call_strong_model(prompt)
            ok, msg = validate_llm_answer(uav_id, ans)
            record = {
                "sample_id": sid,
                "uav_id": uav_id,
                "ok": 1 if ok else 0,
                "check": msg,
                "answer": ans,
                "ts": int(time.time()),
            }
            return record
        except Exception as e:
            last_err = str(e)
            sleep_s = RETRY_BASE_SLEEP * (2 ** (attempt - 1)) + random.uniform(0, 0.3)
            time.sleep(sleep_s)

    # 全部失败
    return {
        "sample_id": sid,
        "uav_id": uav_id,
        "ok": 0,
        "check": "exception",
        "answer": "",
        "error": last_err,
        "ts": int(time.time()),
    }


# =========================================================
# Main: call LLM and save outputs.jsonl
# =========================================================
def run_labeling():
    rows = read_jsonl(PROMPTS_JSONL, limit=LIMIT)
    print(f"[INFO] loaded prompts: {len(rows)}")

    existing = load_existing_outputs(OUTPUTS_JSONL)
    print(f"[INFO] existing outputs: {len(existing)}")

    # 筛选需要调用的样本
    todo = []
    for r in rows:
        sid = r["sample_id"]
        if (not FORCE_RECALL) and sid in existing:
            continue
        todo.append(r)

    print(f"[INFO] to call LLM: {len(todo)} (FORCE_RECALL={FORCE_RECALL})")

    if not todo:
        return

    # 并发调用
    ok_cnt = 0
    bad_cnt = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(label_one, r) for r in todo]
        for fut in as_completed(futures):
            rec = fut.result()
            append_jsonl(OUTPUTS_JSONL, rec)
            if rec.get("ok") == 1:
                ok_cnt += 1
            else:
                bad_cnt += 1

    print(f"[OK] wrote outputs: {OUTPUTS_JSONL}")
    print(f"[STAT] ok={ok_cnt}, bad={bad_cnt}")


# =========================================================
# Build SFT jsonl (messages format) from outputs
# =========================================================
def build_sft():
    rows = read_jsonl(PROMPTS_JSONL, limit=LIMIT)
    outputs = load_existing_outputs(OUTPUTS_JSONL)

    kept = 0
    dropped = 0

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SFT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            sid = r["sample_id"]
            uav_id = r["uav_id"]

            out = outputs.get(sid)
            if not out:
                continue

            if out.get("ok") != 1:
                dropped += 1
                continue

            ans = (out.get("answer") or "").strip()
            ok, msg = validate_llm_answer(uav_id, ans)
            if not ok:
                dropped += 1
                continue

            item = {
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": ans},
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] wrote sft: {SFT_JSONL}")
    print(f"[STAT] kept={kept}, dropped={dropped} (invalid or missing outputs)")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    # 1) 先跑 run_labeling()：调用强模型产出 OUTPUTS_JSONL
    # 2) 再跑 build_sft()：合并成 SFT_JSONL
    #
    # 你也可以先注释其中一个，只跑单步。
    run_labeling()
    build_sft()
    print(f"\n[SUMMARY] TOTAL TOKENS USED = {TOTAL_TOKENS}")