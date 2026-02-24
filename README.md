# drone_sys_python 融合模块使用说明

本文档只覆盖融合模块，包含三部分：
1. 数据生成
2. 模型训练
3. 融合 HTTP 接口使用

## 1. 数据生成

### 1.1 生成原始多模态数据（truth + 5 模态）
在 `drone_sys/app/core/datasetBuilder` 目录执行：

```powershell
cd drone_sys/app/core/datasetBuilder
python .\generate_bluesky_dataset.py
```

脚本会读取：
- `drone_sys/app/core/datasetBuilder/dataset_config.json`

主要输出（按 profile 生成）：
- `drone_sys/app/core/datasetBuilder/dataset/training-datasets/<profile>/batchXX/truth.csv`
- `drone_sys/app/core/datasetBuilder/dataset/training-datasets/<profile>/batchXX/gps.csv`
- `drone_sys/app/core/datasetBuilder/dataset/training-datasets/<profile>/batchXX/radar.csv`
- `drone_sys/app/core/datasetBuilder/dataset/training-datasets/<profile>/batchXX/5g_a.csv`
- `drone_sys/app/core/datasetBuilder/dataset/training-datasets/<profile>/batchXX/tdoa.csv`
- `drone_sys/app/core/datasetBuilder/dataset/training-datasets/<profile>/batchXX/acoustic.csv`

### 1.2 生成置信度并转换为训练输入
继续在 `drone_sys/app/core/datasetBuilder` 目录执行：

```powershell
python .\transfer_confidence.py `
  --dataset-dir .\dataset\training-datasets `
  --output-dir .\dataset-processed\train-datasets `
  --root-mode `
  --worker-num 16
```

说明：
- `--root-mode`：把 `training-datasets` 下多个 profile 汇总并重排为统一 batch，便于训练。
- 输出目录会包含带 `confidence` 的 5 模态 CSV，结构用于融合模型训练。

---

## 2. 模型训练

在 `drone_sys/app/core/droneFusion` 目录执行：

```powershell
cd ..\droneFusion
python .\train.py
```

训练配置入口（直接改代码中的 dataclass）：
- `drone_sys/app/core/droneFusion/train.py`
- `DataConfig`：数据目录、窗口长度、模态配置等
- `ModelConfig`：模型结构
- `TrainConfig`：batch size、epoch、lr、模型保存路径等

默认关键输出：
- `drone_sys/app/core/droneFusion/graph_fusion_model_processed.pt`
- `drone_sys/app/core/droneFusion/graph_norm_stats_processed_sparse_enu.pth`

说明：
- 融合接口推理默认读取上面两个文件，请保证训练完成后它们在该目录下。

---

## 3. 融合接口使用

### 3.1 启动服务
推荐在项目根目录执行：

```powershell
cd D:\MyCode\drone_sys_python
python -m drone_sys.app.main
```

也支持在 `drone_sys/app` 目录执行：

```powershell
python .\main.py
```

### 3.2 接口路径
- 主接口：`POST /fusion/run`
- 兼容路径：`POST /run`

如果网关启用了 `root_path=/drone-fusion`，则外部访问路径通常为：
- `POST /drone-fusion/fusion/run`

### 3.3 请求格式
请求体支持两种：
1. 顶层直接是 `list`（长度需 `>= window_size`，当前模型常见为 20）
2. 对象包裹：`{"uav_id": "...", "data": [ ...N条... ]}`

每条数据可包含 5 个模态对象（缺失的模态可以不传）：
- `gps`
- `radar`
- `fiveg`（也兼容 `5g` / `5g_a`）
- `tdoa`
- `acoustic`

约束与建议：
- 输入长度最少为模型 `window_size`；只传最小长度（如 20）时通常接近单窗口效果。
- 为了更接近 `evaluate.py` 的效果，建议传入连续 `60~120` 条（多窗口重叠融合）。
- 第 1 条里 `gps/radar/fiveg/tdoa` 至少一个要有有效 `lat/lon/alt`，否则无法建立推理坐标参考。
- 未传某模态时，接口会自动按缺失处理（`missing_flag=1`）。
- 建议尽量提供质量参数（如 `Nsat/DOP/RTK`、`E/Ptrk`、`SNR/RSSI/ploss`、`e/eps_sync`），否则会使用保守默认值，精度可能下降。

最小请求骨架：

```json
{
  "uav_id": "UAV00001",
  "data": [
    {
      "timestamp": 1770811667.0,
      "gps": {"lat": 45.2782, "lon": 5.3185, "alt": 133.6, "vx": 21.8, "vy": 34.4, "vz": 0.0, "speed": 40.8, "Nsat": 15, "DOP": 1.4, "RTK": "FIX"},
      "radar": {"lat": 45.27821, "lon": 5.31849, "alt": 133.1, "E": 0.85, "Ptrk": 0.91},
      "fiveg": {"lat": 45.27819, "lon": 5.31851, "alt": 134.0, "SNR": 16.0, "RSSI": -68.0, "d": 66.0, "ploss": 0.10},
      "tdoa": {"lat": 45.278205, "lon": 5.318495, "alt": 133.3, "e": 9.0, "eps_sync": 32.0},
      "acoustic": {"detected_flag": 1, "SNRa": 10.0, "n": 0.25}
    }
  ]
}
```

下面示例是 **20 条最小可运行示例**（便于联调）；实际部署建议使用连续 `60~120` 条以启用多窗口融合，并尽量接近 `evaluate.py` 的效果。

```json
{
  "uav_id": "UAV00001",
  "data": [
    {
      "timestamp": 0,
      "gps":    { "lat": 39.90000, "lon": 116.30000, "alt": 120.0, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 },
      "radar":  { "lat": 39.90001, "lon": 116.29999, "alt": 121.0, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 },
      "fiveg":  { "lat": 39.89998, "lon": 116.30003, "alt": 118.0, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 },
      "tdoa":   { "lat": 39.90004, "lon": 116.29997, "alt": 123.0, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 },
      "acoustic": { "detected_flag": 1 }
    },
    {
      "timestamp": 1,
      "gps":    { "lat": 39.90005, "lon": 116.30008, "alt": 120.2, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 },
      "radar":  { "lat": 39.90006, "lon": 116.30007, "alt": 121.2, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 },
      "fiveg":  { "lat": 39.90003, "lon": 116.30011, "alt": 118.2, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 },
      "tdoa":   { "lat": 39.90009, "lon": 116.30005, "alt": 123.2, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 }
    },
    {
      "timestamp": 2,
      "gps":    { "lat": 39.90010, "lon": 116.30016, "alt": 120.4, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 },
      "radar":  { "lat": 39.90011, "lon": 116.30015, "alt": 121.4, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 },
      "fiveg":  { "lat": 39.90008, "lon": 116.30019, "alt": 118.4, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 },
      "tdoa":   { "lat": 39.90014, "lon": 116.30013, "alt": 123.4, "vx": 8.0, "vy": 5.0, "vz": 0.1, "speed": 9.43 }
    },
    {
      "timestamp": 3,
      "gps":    { "lat": 39.90015, "lon": 116.30024, "alt": 120.6, "vx": 8.1, "vy": 5.1, "vz": 0.1, "speed": 9.57 },
      "radar":  { "lat": 39.90016, "lon": 116.30023, "alt": 121.6, "vx": 8.1, "vy": 5.1, "vz": 0.1, "speed": 9.57 },
      "fiveg":  { "lat": 39.90013, "lon": 116.30027, "alt": 118.6, "vx": 8.1, "vy": 5.1, "vz": 0.1, "speed": 9.57 }
    },
    {
      "timestamp": 4,
      "gps":    { "lat": 39.90020, "lon": 116.30032, "alt": 120.8, "vx": 8.1, "vy": 5.1, "vz": 0.1, "speed": 9.57 },
      "radar":  { "lat": 39.90021, "lon": 116.30031, "alt": 121.8, "vx": 8.1, "vy": 5.1, "vz": 0.1, "speed": 9.57 },
      "tdoa":   { "lat": 39.90024, "lon": 116.30029, "alt": 123.8, "vx": 8.1, "vy": 5.1, "vz": 0.1, "speed": 9.57 }
    },
    {
      "timestamp": 5,
      "gps":    { "lat": 39.90025, "lon": 116.30040, "alt": 121.0, "vx": 8.2, "vy": 5.0, "vz": 0.1, "speed": 9.61 },
      "fiveg":  { "lat": 39.90023, "lon": 116.30043, "alt": 119.0, "vx": 8.2, "vy": 5.0, "vz": 0.1, "speed": 9.61 },
      "tdoa":   { "lat": 39.90029, "lon": 116.30037, "alt": 124.0, "vx": 8.2, "vy": 5.0, "vz": 0.1, "speed": 9.61 },
      "acoustic": { "detected_flag": 1 }
    },
    {
      "timestamp": 6,
      "gps":    { "lat": 39.90030, "lon": 116.30048, "alt": 121.2, "vx": 8.2, "vy": 5.0, "vz": 0.1, "speed": 9.61 },
      "radar":  { "lat": 39.90031, "lon": 116.30047, "alt": 122.2, "vx": 8.2, "vy": 5.0, "vz": 0.1, "speed": 9.61 },
      "fiveg":  { "lat": 39.90028, "lon": 116.30051, "alt": 119.2, "vx": 8.2, "vy": 5.0, "vz": 0.1, "speed": 9.61 },
      "acoustic": { "detected_flag": 1 }
    },
    {
      "timestamp": 7,
      "gps":    { "lat": 39.90035, "lon": 116.30056, "alt": 121.4, "vx": 8.3, "vy": 5.0, "vz": 0.1, "speed": 9.69 },
      "radar":  { "lat": 39.90036, "lon": 116.30055, "alt": 122.4, "vx": 8.3, "vy": 5.0, "vz": 0.1, "speed": 9.69 },
      "tdoa":   { "lat": 39.90039, "lon": 116.30053, "alt": 124.4, "vx": 8.3, "vy": 5.0, "vz": 0.1, "speed": 9.69 }
    },
    {
      "timestamp": 8,
      "radar":  { "lat": 39.90041, "lon": 116.30063, "alt": 122.6, "vx": 8.3, "vy": 5.0, "vz": 0.1, "speed": 9.69 },
      "fiveg":  { "lat": 39.90038, "lon": 116.30067, "alt": 119.6, "vx": 8.3, "vy": 5.0, "vz": 0.1, "speed": 9.69 },
      "tdoa":   { "lat": 39.90044, "lon": 116.30061, "alt": 124.6, "vx": 8.3, "vy": 5.0, "vz": 0.1, "speed": 9.69 },
      "acoustic": { "detected_flag": 1 }
    },
    {
      "timestamp": 9,
      "radar":  { "lat": 39.90046, "lon": 116.30071, "alt": 122.8, "vx": 8.4, "vy": 5.1, "vz": 0.1, "speed": 9.83 },
      "fiveg":  { "lat": 39.90043, "lon": 116.30075, "alt": 119.8, "vx": 8.4, "vy": 5.1, "vz": 0.1, "speed": 9.83 },
      "tdoa":   { "lat": 39.90049, "lon": 116.30069, "alt": 124.8, "vx": 8.4, "vy": 5.1, "vz": 0.1, "speed": 9.83 }
    },
    {
      "timestamp": 10,
      "radar":  { "lat": 39.90051, "lon": 116.30079, "alt": 123.0, "vx": 8.4, "vy": 5.1, "vz": 0.1, "speed": 9.83 },
      "fiveg":  { "lat": 39.90048, "lon": 116.30083, "alt": 120.0, "vx": 8.4, "vy": 5.1, "vz": 0.1, "speed": 9.83 },
      "acoustic": { "detected_flag": 1 }
    },
    {
      "timestamp": 11,
      "gps":    { "lat": 39.90055, "lon": 116.30088, "alt": 122.2, "vx": 8.5, "vy": 5.2, "vz": 0.1, "speed": 9.96 },
      "fiveg":  { "lat": 39.90053, "lon": 116.30091, "alt": 120.2, "vx": 8.5, "vy": 5.2, "vz": 0.1, "speed": 9.96 },
      "tdoa":   { "lat": 39.90059, "lon": 116.30085, "alt": 125.2, "vx": 8.5, "vy": 5.2, "vz": 0.1, "speed": 9.96 }
    },
    {
      "timestamp": 12,
      "gps":    { "lat": 39.90060, "lon": 116.30096, "alt": 122.4, "vx": 8.5, "vy": 5.2, "vz": 0.1, "speed": 9.96 },
      "radar":  { "lat": 39.90061, "lon": 116.30095, "alt": 123.4, "vx": 8.5, "vy": 5.2, "vz": 0.1, "speed": 9.96 },
      "fiveg":  { "lat": 39.90058, "lon": 116.30099, "alt": 120.4, "vx": 8.5, "vy": 5.2, "vz": 0.1, "speed": 9.96 },
      "tdoa":   { "lat": 39.90064, "lon": 116.30093, "alt": 125.4, "vx": 8.5, "vy": 5.2, "vz": 0.1, "speed": 9.96 }
    },
    {
      "timestamp": 13,
      "gps":    { "lat": 39.90065, "lon": 116.30104, "alt": 122.6, "vx": 8.6, "vy": 5.2, "vz": 0.1, "speed": 10.05 },
      "radar":  { "lat": 39.90066, "lon": 116.30103, "alt": 123.6, "vx": 8.6, "vy": 5.2, "vz": 0.1, "speed": 10.05 },
      "fiveg":  { "lat": 39.90063, "lon": 116.30107, "alt": 120.6, "vx": 8.6, "vy": 5.2, "vz": 0.1, "speed": 10.05 }
    },
    {
      "timestamp": 14,
      "gps":    { "lat": 39.90070, "lon": 116.30112, "alt": 122.8, "vx": 8.6, "vy": 5.2, "vz": 0.1, "speed": 10.05 },
      "radar":  { "lat": 39.90071, "lon": 116.30111, "alt": 123.8, "vx": 8.6, "vy": 5.2, "vz": 0.1, "speed": 10.05 },
      "tdoa":   { "lat": 39.90074, "lon": 116.30109, "alt": 125.8, "vx": 8.6, "vy": 5.2, "vz": 0.1, "speed": 10.05 },
      "acoustic": { "detected_flag": 1 }
    },
    {
      "timestamp": 15,
      "gps":    { "lat": 39.90075, "lon": 116.30120, "alt": 123.0, "vx": 8.7, "vy": 5.3, "vz": 0.1, "speed": 10.19 },
      "fiveg":  { "lat": 39.90073, "lon": 116.30123, "alt": 121.0, "vx": 8.7, "vy": 5.3, "vz": 0.1, "speed": 10.19 },
      "tdoa":   { "lat": 39.90079, "lon": 116.30117, "alt": 126.0, "vx": 8.7, "vy": 5.3, "vz": 0.1, "speed": 10.19 }
    },
    {
      "timestamp": 16,
      "gps":    { "lat": 39.90080, "lon": 116.30128, "alt": 123.2, "vx": 8.7, "vy": 5.3, "vz": 0.1, "speed": 10.19 },
      "radar":  { "lat": 39.90081, "lon": 116.30127, "alt": 124.2, "vx": 8.7, "vy": 5.3, "vz": 0.1, "speed": 10.19 },
      "fiveg":  { "lat": 39.90078, "lon": 116.30131, "alt": 121.2, "vx": 8.7, "vy": 5.3, "vz": 0.1, "speed": 10.19 },
      "tdoa":   { "lat": 39.90084, "lon": 116.30125, "alt": 126.2, "vx": 8.7, "vy": 5.3, "vz": 0.1, "speed": 10.19 },
      "acoustic": { "detected_flag": 1 }
    },
    {
      "timestamp": 17,
      "gps":    { "lat": 39.90085, "lon": 116.30136, "alt": 123.4, "vx": 8.8, "vy": 5.4, "vz": 0.1, "speed": 10.33 },
      "radar":  { "lat": 39.90086, "lon": 116.30135, "alt": 124.4, "vx": 8.8, "vy": 5.4, "vz": 0.1, "speed": 10.33 },
      "fiveg":  { "lat": 39.90083, "lon": 116.30139, "alt": 121.4, "vx": 8.8, "vy": 5.4, "vz": 0.1, "speed": 10.33 }
    },
    {
      "timestamp": 18,
      "gps":    { "lat": 39.90090, "lon": 116.30144, "alt": 123.6, "vx": 8.8, "vy": 5.4, "vz": 0.1, "speed": 10.33 },
      "radar":  { "lat": 39.90091, "lon": 116.30143, "alt": 124.6, "vx": 8.8, "vy": 5.4, "vz": 0.1, "speed": 10.33 },
      "tdoa":   { "lat": 39.90094, "lon": 116.30141, "alt": 126.6, "vx": 8.8, "vy": 5.4, "vz": 0.1, "speed": 10.33 }
    },
    {
      "timestamp": 19,
      "gps":    { "lat": 39.90095, "lon": 116.30152, "alt": 123.8, "vx": 8.9, "vy": 5.5, "vz": 0.1, "speed": 10.46 },
      "radar":  { "lat": 39.90096, "lon": 116.30151, "alt": 124.8, "vx": 8.9, "vy": 5.5, "vz": 0.1, "speed": 10.46 },
      "fiveg":  { "lat": 39.90093, "lon": 116.30155, "alt": 121.8, "vx": 8.9, "vy": 5.5, "vz": 0.1, "speed": 10.46 },
      "tdoa":   { "lat": 39.90099, "lon": 116.30149, "alt": 126.8, "vx": 8.9, "vy": 5.5, "vz": 0.1, "speed": 10.46 },
      "acoustic": { "detected_flag": 1 }
    }
  ]
}
```

### 3.4 返回格式
返回为 JSON list（长度通常与输入条数一致，不再固定为 20），每个元素：

```json
[
  {
    "timestamp": 1770811667.0,
    "lat": 45.27820,
    "lon": 5.31850,
    "alt": 133.60
  }
]
```

### 3.5 调用示例（PowerShell）

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8080/fusion/run `
  -Method Post `
  -ContentType "application/json" `
  -InFile .\request.json
```

`request.json` 可以使用上面的 20 条最小示例；若希望结果更接近离线评测（`evaluate.py`），建议改为连续 `60~120` 条请求体。

### 3.6 注意事项（多窗口融合 / 实时接入）
- 当前接口已支持多窗口融合：输入长度 `>= window_size` 时会自动走重叠窗口 + merge 推理。
- 实时接入建议按 `uav_id` 维护滚动缓存（推荐最近 `60~120` 条），每次请求后取返回列表最后一个点作为当前融合结果。
- 当缓存长度 `< window_size` 时，不建议调用模型融合；可先使用单模态结果或等待缓存区凑够窗口长度。
- 若某时刻判断 `gps` 不可信，可在该时刻直接不传 `gps`（或显式 `missing_flag=1`），接口会按缺失模态处理。
- 时间戳建议单调递增，且同一 packet 内各模态尽量对应同一时刻（或近似同一时刻），否则会影响对齐与融合效果。
