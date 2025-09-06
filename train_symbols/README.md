# YOLO Symbol Detection Training

这个项目用于训练YOLO模型来检测几何图形中的符号标记，支持从PGDP5K、PGPS9K和Geometry3K数据集训练统一的符号检测模型。

## 项目概述

- **目标**: 训练小目标检测模型识别几何符号（直角、角弧、等长刻度、平行标记等）
- **输入**: 三个数学几何数据集的原始标注
- **输出**: `png2svg/models/symbols_yolo.onnx` (ONNX格式，动态输入)
- **模型**: Ultralytics YOLO11 (针对小目标优化)

## 目录结构

```
train_symbols/
├── configs/                 # 配置文件
│   ├── classes.yaml         # 类别定义和映射
│   ├── dataset.yaml.j2      # 数据集配置模板
│   ├── train.small.yaml     # 快速训练配置
│   └── train.full.yaml      # 完整训练配置
├── converters/              # 数据转换器
│   ├── common.py            # 通用工具函数
│   ├── pgdp5k.py           # PGDP5K转换器
│   ├── pgps9k.py           # PGPS9K转换器
│   └── geometry3k.py       # Geometry3K转换器
├── scripts/                 # 核心脚本
│   ├── 00_build_dataset.py  # 数据集构建
│   ├── 01_train.py          # 模型训练
│   ├── 02_export_onnx.py    # ONNX导出
│   ├── 03_sanity_check.py   # 模型验证
│   └── 04_stats.py          # 统计分析
├── requirements.txt         # Python依赖
├── Makefile                # 快捷命令
└── README.md               # 项目文档
```

## 快速开始

### 1. 环境准备

```bash
cd train_symbols

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保项目根目录下有以下数据集：

```
../eval/
├── PGDP5K/
│   ├── annotations/
│   ├── train/
│   ├── val/
│   └── test/
├── PGPS9K/
│   ├── diagram_annotation.json
│   └── Diagram/
└── Geometry3K/
    └── symbols/
```

### 3. 完整训练流程

使用Makefile快捷命令：

```bash
# 方式一：一键执行全流程
make all

# 方式二：分步执行
make build      # 构建数据集
make train-full # 完整训练
make export     # 导出ONNX
make check      # 验证模型
```

或者手动执行脚本：

```bash
# 1. 构建统一数据集
python scripts/00_build_dataset.py

# 2. 训练模型（选择一种）
python scripts/01_train.py --cfg ../configs/train.small.yaml  # 快速版本
python scripts/01_train.py --cfg ../configs/train.full.yaml   # 完整版本

# 3. 导出ONNX模型
python scripts/02_export_onnx.py

# 4. 验证模型效果
python scripts/03_sanity_check.py

# 5. 分析数据集统计
python scripts/04_stats.py
```

## 详细说明

### 支持的符号类别

模型支持检测以下12类几何符号：

1. `right_angle` - 直角标记
2. `arc_1` - 单弧角标记  
3. `arc_2` - 双弧角标记
4. `arc_3` - 三弧角标记
5. `tick_1` - 单刻度等长标记
6. `tick_2` - 双刻度等长标记
7. `tick_3` - 三刻度等长标记
8. `parallel_mark` - 平行标记
9. `arrow_head` - 箭头
10. `dot_filled` - 实心点
11. `dot_hollow` - 空心点
12. `extend_mark` - 延长线标记

### 数据集转换

各数据集的原始标注会通过映射转换为统一类别：

- **PGDP5K**: JSON格式，bbox为[x,y,w,h]
- **PGPS9K**: JSON格式，类似PGDP5K结构
- **Geometry3K**: Pascal VOC XML格式，bbox为[xmin,ymin,xmax,ymax]

### 训练配置

**小规模训练** (`train.small.yaml`):
- 图像尺寸: 960px
- 训练轮数: 60 epochs
- 适用于快速测试和验证

**完整训练** (`train.full.yaml`):
- 图像尺寸: 1280px  
- 训练轮数: 160 epochs
- 针对小目标优化的完整配置

### 模型优化特性

- **小目标优化**: 使用较大输入分辨率(1280px)
- **数据增强**: Mosaic、翻转、缩放等
- **后处理**: 包含NMS的完整推理流程
- **动态输入**: ONNX模型支持不同尺寸输入

## 脚本说明

### 00_build_dataset.py
构建YOLO格式数据集，将三个数据集的标注统一转换。

```bash
python scripts/00_build_dataset.py \
    --max-per-dataset 1000 \  # 限制每个数据集的样本数（调试用）
    --seed 1234 \             # 随机种子
    --ratios 0.8 0.1 0.1      # train/val/test分割比例
```

### 01_train.py
使用Ultralytics YOLO训练模型。

```bash
python scripts/01_train.py \
    --data ../datasets/pg_symbols/dataset.yaml \
    --model yolo11n.pt \      # 预训练模型
    --cfg ../configs/train.full.yaml \
    --device 0 \              # GPU设备
    --resume                  # 恢复训练
```

### 02_export_onnx.py
将训练好的模型导出为ONNX格式。

```bash
python scripts/02_export_onnx.py \
    --model runs/detect/symbols/weights/best.pt \
    --output-name symbols_yolo.onnx \
    --dynamic \               # 动态batch size
    --opset 12                # ONNX操作集版本
```

### 03_sanity_check.py
对模型进行可视化验证。

```bash
python scripts/03_sanity_check.py \
    --model ../png2svg/models/symbols_yolo.onnx \
    --split val \             # 测试集split
    --num-images 20 \         # 测试图像数量
    --conf 0.25               # 置信度阈值
```

### 04_stats.py
生成数据集统计信息和可视化图表。

```bash
python scripts/04_stats.py \
    --data ../datasets/pg_symbols \
    --output ../outputs/stats \
    --tail-threshold 10       # 长尾类别阈值
```

## 输出文件

训练完成后会生成以下文件：

- `datasets/pg_symbols/` - 统一的YOLO数据集
- `runs/detect/symbols/` - 训练日志和模型权重
- `png2svg/models/symbols_yolo.onnx` - 最终ONNX模型
- `outputs/sanity_check/` - 预测可视化结果
- `outputs/stats/` - 统计分析报告

## 使用建议

### 快速验证流程
```bash
# 使用小数据集快速测试完整流程
make build-small
make train-small  
make export
make check
```

### 生产环境流程
```bash
# 完整数据集训练
make build
make train-full
make export
make check
make stats
```

### 性能调优

1. **数据不平衡**: 查看`04_stats.py`输出的长尾分析
2. **小目标检测**: 增大输入分辨率到1280px或更高
3. **训练收敛**: 监控`runs/detect/symbols/`中的训练日志
4. **推理速度**: 调整ONNX导出的`opset`版本

## 故障排除

### 常见问题

1. **数据集路径错误**
   ```
   Error: Dataset directory not found
   ```
   检查`eval/`目录下是否有对应的数据集。

2. **GPU内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   减小`batch`大小或使用`--device cpu`。

3. **类别映射问题**
   ```
   Warning: Class 'xxx' not in mapping, skipping
   ```
   检查`configs/classes.yaml`中的映射定义。

### 调试技巧

- 使用`--max-per-dataset 100`快速验证流程
- 检查`outputs/stats/`中的数据分布分析
- 查看`outputs/sanity_check/`中的预测结果
- 监控`runs/detect/symbols/`中的训练指标

## 许可和数据使用

本项目仅用于学术研究。数据集使用请遵循各自的许可协议：
- PGDP5K、PGPS9K、Geometry3K数据集的使用需要遵循其原始许可

## 贡献

如需改进或扩展功能，请确保：
1. 遵循现有代码结构
2. 添加适当的日志和错误处理  
3. 更新相关文档
4. 测试完整流程

---

**项目完成后，最终的ONNX模型将自动放置在`png2svg/models/symbols_yolo.onnx`，可直接被png2svg矢量化管线调用。**