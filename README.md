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
1. 顶层直接是 `list`（长度必须为 20）
2. 对象包裹：`{"uav_id": "...", "data": [ ...20条... ]}`

每条数据应包含 5 个模态对象：
- `gps`
- `radar`
- `fiveg`（也兼容 `5g` / `5g_a`）
- `tdoa`
- `acoustic`

约束：
- 固定 20 条时间步。
- 第 1 条里 `gps/radar/fiveg/tdoa` 至少一个要有有效 `lat/lon/alt`，否则无法建立推理坐标参考。

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

实际调用时请补齐 `data` 共 20 条。

### 3.4 返回格式
返回为 JSON list（长度通常为 20），每个元素：

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

`request.json` 即上述 20 条请求体。
