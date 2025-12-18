#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import uuid
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

# =========================
# Config
# =========================
SEED = 42
random.seed(SEED)

N_SAMPLES = 5000

# 你模板里的阈值
SEP_X = 50  # m

# 你离线产生的电量状态阈值（用于生成 low_battery/critical_battery 与 battery_pct 一致）
BAT_LOW = 20
BAT_CRIT = 10  # 你要求电量 10-100，本脚本会让 critical_battery=1 时 battery_pct=10~14（可调整）

# 7 类样本分布（总和不必严格=1，会自动归一化）
WEIGHTS = {
    "A4": 0.20,          # 无告警
    "A3_ONLY": 0.12,     # 仅续航
    "A2_ONLY": 0.15,     # 仅近距冲突
    "A1_ONLY": 0.12,     # 仅空域限制
    "A1_A2": 0.15,       # A1+A2
    "A2_A3": 0.10,       # A2+A3
    "A1_A2_A3": 0.16,    # 全触发（重点）
}

OUT_DIR = "./sft_build"
PROMPTS_JSONL = os.path.join(OUT_DIR, "prompts.jsonl")
SFT_JSONL = os.path.join(OUT_DIR, "sft.jsonl")

# 是否在生成后直接调用强模型（你可以先 False，只生成 prompts）
CALL_LLM_NOW = False


# =========================
# Your final prompt template (rendered with concrete values)
# =========================
PROMPT_TEMPLATE = """【任务】
你是低空空域安全指挥决策模型。
基于下述“态势输入”和“告警规则（仅定义触发阈值）”，为目标无人机生成最终指挥指令及综合原因说明。

【输出约束】
- 第一行必须严格为：{uav_id}，请悬停 / 原地降落 / 返航 / 正常飞行（四选一）
- 然后输出一段“综合原因说明,50字以内”，开头是"理由:"，然后是原因内容，必须体现“权衡与取舍”，并最终落到“为什么选择该指令”
- 不允许输出其他内容
- 不得引入输入中不存在的新事实

========================
【态势输入】

【目标无人机】
- uav_id: {uav_id}
- lat: {lat}
- lon: {lon}
- alt_m: {alt_m}
- speed_mps: {speed_mps}
- battery_pct: {battery_pct}

【200m内其他无人机】（已离线计算）
- count: {nearby_count}
- items:
{nearby_items}

【500m内限制区域】
- count: {zone_count}
- items:
{zone_items}

【派生状态】
- nearest_uav_id: {nearest_uav_id}
- nearest_sep_m: {nearest_sep_m}
- any_route_intersect: {any_route_intersect}
- low_battery: {low_battery}
- critical_battery: {critical_battery}

========================
【告警规则 ALERT_RULES（仅阈值，不含指令映射）】

【A1 空域限制告警】
- 触发条件：存在 inside_zone = 1 或 over_limit_alt = 1

【A2 近距冲突告警】
- 触发条件：any_route_intersect = 1 且 nearest_sep_m < {sep_x}

【A3 续航告警】
- 触发条件：low_battery = 1 或 critical_battery = 1

【A4 无告警】
- 触发条件：未触发 A1/A2/A3

========================
【阈值】
- SEP_X = {sep_x} m

========================
【综合原因说明写作要求（必须遵守）】

原因说明必须是一段完整中文叙述，但要包含以下四类信息（可按自然语言串起来，不要分条，不要写小标题）：
1) 你判断触发了哪些告警（可多个，如空域限制和近距冲突），不用引用对应事实字段与数值，只需要列举就行。
2) 解释各告警背后的风险含义（例如空域合规风险/近距冲突风险/续航风险），且必须体现“哪个更紧急、哪个是次要”。
3) 说明你在四类指令中如何取舍：为什么不是另外三种（至少点出1–2个关键否决理由，例如“返航会拉长冲突窗口”或“原地降落对地面风险不明”等；不得编造新事实）。
4) 最后用一句话明确收束：因此选择“(你输出的指令)”作为当前最合适的指挥命令。

【禁止项】
- 不得只复述规则或只列出告警编号
- 不得引入输入之外的环境信息（天气、风、地面人群、机场等）
"""


# =========================
# Data model
# =========================
@dataclass
class NearbyUAV:
    id: str
    route_intersect: int  # 0/1
    min_sep_m: int        # already computed


@dataclass
class RestrictZone:
    zone_id: str
    inside_zone: int      # 0/1
    over_limit_alt: int   # 0/1


@dataclass
class Sample:
    sample_id: str
    scenario: str            # one of 7
    uav_id: str
    lat: float
    lon: float
    alt_m: int
    speed_mps: float
    battery_pct: int

    nearby: List[NearbyUAV]
    zones: List[RestrictZone]

    # derived
    nearest_uav_id: str
    nearest_sep_m: int
    any_route_intersect: int
    low_battery: int
    critical_battery: int

    sep_x: int = SEP_X


# =========================
# Helpers
# =========================
def _rand_lat_lon(base_lat=39.98, base_lon=116.30, span=0.01):
    # 简单生成，不要求地理精确（你不让模型算距离，距离由你离线给出即可）
    lat = base_lat + random.uniform(-span, span)
    lon = base_lon + random.uniform(-span, span)
    return round(lat, 6), round(lon, 6)

def _rand_uav_id(prefix="UAV"):
    return f"{prefix}_{random.randint(1, 999):03d}"

def _alloc_counts(weights: Dict[str, float], n: int) -> Dict[str, int]:
    keys = list(weights.keys())
    w = [weights[k] for k in keys]
    s = sum(w)
    w = [x / s for x in w]

    # 先按比例取整，再把余数补到随机 keys 上
    raw = [int(n * wi) for wi in w]
    remain = n - sum(raw)
    for _ in range(remain):
        raw[random.randrange(len(raw))] += 1
    return {k: raw[i] for i, k in enumerate(keys)}

def _battery_for_flags(low: int, crit: int) -> int:
    # 电量范围要求 10-100
    if crit == 1:
        return random.randint(max(BAT_CRIT, 10), 14)  # 10-14
    if low == 1:
        return random.randint(15, max(BAT_LOW - 1, 15))  # 15-19
    return random.randint(max(BAT_LOW, 20), 100)  # 20-100

def _make_nearby_list(want_a2: bool) -> List[NearbyUAV]:
    # nearby 0-10
    cnt = random.randint(0, 10)
    if cnt == 0 and want_a2:
        cnt = random.randint(1, 3)

    items: List[NearbyUAV] = []
    for _ in range(cnt):
        items.append(
            NearbyUAV(
                id=_rand_uav_id(),
                route_intersect=0,
                min_sep_m=random.randint(60, 200),
            )
        )

    if want_a2:
        # 强制至少一个交叉 + 最近距离 < SEP_X
        # 把第一架改成交叉且很近
        if cnt == 0:
            items.append(NearbyUAV(id=_rand_uav_id(), route_intersect=1, min_sep_m=random.randint(10, SEP_X - 1)))
        else:
            items[0].route_intersect = 1
            items[0].min_sep_m = random.randint(10, SEP_X - 1)

        # 其他无人机可随机
        for i in range(1, len(items)):
            items[i].route_intersect = 1 if random.random() < 0.2 else 0
            items[i].min_sep_m = random.randint(30, 200)
    else:
        # 确保 A2 不触发：要么无交叉，要么最近距离>=SEP_X
        # 这里简化：全部 route_intersect=0；且若有最近距离也给 >= SEP_X
        for it in items:
            it.route_intersect = 0
            it.min_sep_m = random.randint(SEP_X, 200)

    return items

def _make_zone_list(want_a1: bool) -> List[RestrictZone]:
    # zones 0-5
    cnt = random.randint(0, 5)
    if cnt == 0 and want_a1:
        cnt = random.randint(1, 2)

    zones: List[RestrictZone] = []
    for i in range(cnt):
        zones.append(
            RestrictZone(
                zone_id=f"RZ_{random.randint(1, 99):02d}",
                inside_zone=0,
                over_limit_alt=0,
            )
        )

    if want_a1:
        # 至少一个 inside_zone 或 over_limit_alt 为 1
        z = zones[0] if zones else RestrictZone(zone_id=f"RZ_{random.randint(1, 99):02d}", inside_zone=0, over_limit_alt=0)
        # 随机选触发方式
        if random.random() < 0.5:
            z.inside_zone = 1
            z.over_limit_alt = 0 if random.random() < 0.7 else 1
        else:
            z.inside_zone = 0 if random.random() < 0.7 else 1
            z.over_limit_alt = 1
        if zones:
            zones[0] = z
        else:
            zones.append(z)
    else:
        # 确保不触发：全部 0
        for z in zones:
            z.inside_zone = 0
            z.over_limit_alt = 0

    return zones

def _derive_from_lists(nearby: List[NearbyUAV]) -> (str, int, int):
    if not nearby:
        return "NONE", 999, 0
    nearest = min(nearby, key=lambda x: x.min_sep_m)
    any_intersect = 1 if any(x.route_intersect == 1 for x in nearby) else 0
    return nearest.id, nearest.min_sep_m, any_intersect

def _render_nearby_items(nearby: List[NearbyUAV]) -> str:
    if not nearby:
        return "  - (none)\n"
    lines = []
    for it in nearby:
        lines.append(f"  - id: {it.id}\n"
                     f"    route_intersect: {it.route_intersect}\n"
                     f"    min_sep_m: {it.min_sep_m}\n")
    return "".join(lines)

def _render_zone_items(zones: List[RestrictZone]) -> str:
    if not zones:
        return "  - (none)\n"
    lines = []
    for z in zones:
        lines.append(f"  - zone_id: {z.zone_id}\n"
                     f"    inside_zone: {z.inside_zone}\n"
                     f"    over_limit_alt: {z.over_limit_alt}\n")
    return "".join(lines)

def render_prompt(s: Sample) -> str:
    return PROMPT_TEMPLATE.format(
        uav_id=s.uav_id,
        lat=s.lat,
        lon=s.lon,
        alt_m=s.alt_m,
        speed_mps=s.speed_mps,
        battery_pct=s.battery_pct,
        nearby_count=len(s.nearby),
        nearby_items=_render_nearby_items(s.nearby),
        zone_count=len(s.zones),
        zone_items=_render_zone_items(s.zones),
        nearest_uav_id=s.nearest_uav_id,
        nearest_sep_m=s.nearest_sep_m,
        any_route_intersect=s.any_route_intersect,
        low_battery=s.low_battery,
        critical_battery=s.critical_battery,
        sep_x=s.sep_x,
    )

def compute_alerts(s: Sample) -> Dict[str, int]:
    a1 = 1 if any(z.inside_zone == 1 or z.over_limit_alt == 1 for z in s.zones) else 0
    a2 = 1 if (s.any_route_intersect == 1 and s.nearest_sep_m < s.sep_x) else 0
    a3 = 1 if (s.low_battery == 1 or s.critical_battery == 1) else 0
    a4 = 1 if (a1 == 0 and a2 == 0 and a3 == 0) else 0
    return {"A1": a1, "A2": a2, "A3": a3, "A4": a4}

def _scenario_targets(scenario: str) -> Dict[str, int]:
    # want_a1/a2/a3
    mapping = {
        "A4": {"A1": 0, "A2": 0, "A3": 0},
        "A3_ONLY": {"A1": 0, "A2": 0, "A3": 1},
        "A2_ONLY": {"A1": 0, "A2": 1, "A3": 0},
        "A1_ONLY": {"A1": 1, "A2": 0, "A3": 0},
        "A1_A2": {"A1": 1, "A2": 1, "A3": 0},
        "A2_A3": {"A1": 0, "A2": 1, "A3": 1},
        "A1_A2_A3": {"A1": 1, "A2": 1, "A3": 1},
    }
    return mapping[scenario]

def generate_one_sample(scenario: str) -> Sample:
    want = _scenario_targets(scenario)

    sample_id = str(uuid.uuid4())
    uav_id = _rand_uav_id(prefix="UAV")

    lat, lon = _rand_lat_lon()
    alt_m = random.randint(30, 200)
    speed_mps = round(random.uniform(0.0, 18.0), 1)

    # battery flags
    if want["A3"] == 1:
        # 决定 low/critical 的组合：critical 更少一些
        critical = 1 if random.random() < 0.35 else 0
        low = 1 if critical == 0 else 1  # critical=1 时 low 也置 1（符合你规则“或”且更严格）
    else:
        low, critical = 0, 0

    battery_pct = _battery_for_flags(low, critical)

    # nearby & zones
    nearby = _make_nearby_list(want_a2=(want["A2"] == 1))
    zones = _make_zone_list(want_a1=(want["A1"] == 1))

    nearest_uav_id, nearest_sep_m, any_intersect = _derive_from_lists(nearby)

    s = Sample(
        sample_id=sample_id,
        scenario=scenario,
        uav_id=uav_id,
        lat=lat,
        lon=lon,
        alt_m=alt_m,
        speed_mps=speed_mps,
        battery_pct=battery_pct,
        nearby=nearby,
        zones=zones,
        nearest_uav_id=nearest_uav_id,
        nearest_sep_m=nearest_sep_m,
        any_route_intersect=1 if (want["A2"] == 1) else 0,  # A2 要求 any_route_intersect=1
        low_battery=low,
        critical_battery=critical,
        sep_x=SEP_X,
    )

    # 二次校验：确保告警组合严格匹配
    alerts = compute_alerts(s)
    tgt = {"A1": want["A1"], "A2": want["A2"], "A3": want["A3"]}
    if any(alerts[k] != tgt[k] for k in tgt.keys()):
        # 失败则重生成（很少发生）
        return generate_one_sample(scenario)
    return s


# =========================
# Strong model API stub
# =========================
def call_strong_model(prompt: str) -> str:
    """
    TODO: 你接入真实 API。
    返回必须是两行：
    1) {uav_id}，请悬停/原地降落/返航/正常飞行
    2) 理由:xxxxx（<=50字）
    """
    raise NotImplementedError("Please implement your strong model API call here.")


# =========================
# Output validation (for SFT)
# =========================
FIRST_LINE_RE = re.compile(r"^(?P<id>[A-Za-z0-9_]+)，请(悬停|原地降落|返航|正常飞行)$")

def validate_llm_answer(uav_id: str, ans: str) -> (bool, str):
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
    reason = lines[1][len("理由:"):].strip()
    if len(reason) > 50:
        return False, "reason_too_long"
    return True, "ok"


# =========================
# Build prompts.jsonl
# =========================
def build_prompts_jsonl(n_samples: int) -> List[Dict[str, Any]]:
    os.makedirs(OUT_DIR, exist_ok=True)

    alloc = _alloc_counts(WEIGHTS, n_samples)
    rows: List[Dict[str, Any]] = []

    for scenario, cnt in alloc.items():
        for _ in range(cnt):
            s = generate_one_sample(scenario)
            prompt = render_prompt(s)
            alerts = compute_alerts(s)

            row = {
                "sample_id": s.sample_id,
                "scenario": s.scenario,
                "uav_id": s.uav_id,
                "alerts": alerts,              # 方便后续统计/抽样
                "structured": asdict(s),        # 保留结构化输入，方便你复现或做审计
                "prompt": prompt,
            }
            rows.append(row)

    random.shuffle(rows)

    with open(PROMPTS_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] wrote prompts: {PROMPTS_JSONL}  (n={len(rows)})")
    return rows


# =========================
# Build sft.jsonl (messages format)
# =========================
def build_sft_jsonl_from_outputs(prompts_rows: List[Dict[str, Any]],
                                outputs: Dict[str, str]):
    """
    outputs: {sample_id: llm_answer}
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    kept = 0
    bad = 0

    with open(SFT_JSONL, "w", encoding="utf-8") as f:
        for r in prompts_rows:
            sid = r["sample_id"]
            uav_id = r["uav_id"]
            if sid not in outputs:
                continue
            ans = outputs[sid]
            ok, msg = validate_llm_answer(uav_id, ans)
            if not ok:
                bad += 1
                continue

            item = {
                "messages": [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": ans.strip()},
                ]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[OK] wrote sft: {SFT_JSONL}  kept={kept}, dropped={bad}")


# =========================
# Optional: end-to-end call now
# =========================
def run_end_to_end():
    rows = build_prompts_jsonl(N_SAMPLES)

    if not CALL_LLM_NOW:
        print("[INFO] CALL_LLM_NOW=False, stop after prompts.jsonl")
        print("       Next: call your strong LLM to produce outputs, then run build_sft_jsonl_from_outputs().")
        return

    outputs = {}
    for r in rows:
        sid = r["sample_id"]
        prompt = r["prompt"]
        ans = call_strong_model(prompt)
        outputs[sid] = ans

    build_sft_jsonl_from_outputs(rows, outputs)


if __name__ == "__main__":
    run_end_to_end()