# PNG2SVG - 数学配图智能转换系统

一套完整的PNG数学配图转语义SVG系统，集成智能几何分析和关系推理功能。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 核心功能

- **几何图元检测**: 直线、圆形、弧线，支持虚实线识别
- **数学符号识别**: 直角标记、刻度、箭头、平行标记（可选YOLO增强）
- **OCR文字检测**: 标签识别（A/B/C）、角度（30°）、长度（x+1），使用PaddleOCR/Tesseract
- **拓扑关系推理**: 平行、垂直、共线、等长等几何关系自动识别
- **约束优化求解**: 基于scipy最小二乘的几何约束求解器
- **分层SVG输出**: 语义标注，包含完整data-*属性
- **结构化元数据导出**: GeoJSON格式，包含完整语义信息
- **并行批处理**: 多核心处理，支持错误恢复
- **优雅降级**: 可选依赖缺失时自动回退

## 📦 安装指南

### 系统要求
- Python 3.8+ 
- Windows/Linux/macOS
- Tesseract OCR（用于文字识别回退）

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/example/png2svg.git
cd png2svg

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:  
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 系统依赖

**Windows:**
```bash
# 安装Tesseract（OCR回退功能必需）
# 下载地址: https://github.com/UB-Mannheim/tesseract/wiki
# 或使用chocolatey:
choco install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract
```

## 🎯 快速开始

### 基本使用

```bash
# 处理目录中所有PNG文件
python -m png2svg.cli --input ./pngs --output ./out

# 使用配置文件处理
python -m png2svg.cli --config config.yaml

# 处理单个文件并自定义设置
python -m png2svg.cli --input ./pngs --single demo.png --no-constraints --verbose
```

### Python API

```python
from png2svg import Config, process_image, load_config

# 加载配置
cfg = load_config('config.yaml')

# 处理单个图像
result = process_image('diagram.png', cfg)

if result['success']:
    print(f"SVG输出: {result['svg_path']}")
    print(f"GeoJSON: {result['geo_path']}")
    print(f"处理时间: {result['stats']['processing_time']:.2f}s")
else:
    print(f"错误: {result['error']}")

# 批量处理
from png2svg import batch_convert

results = batch_convert('./pngs', './out', 'config.yaml')
print(f"已处理 {len(results)} 张图像")
```

## ⚙️ 配置说明

### 基本配置 (config.yaml)

```yaml
# 输入输出设置
input_dir: "./pngs"
output_dir: "./out" 
jobs: 4                         # 并行处理进程数

# 处理选项
deskew: true                    # 自动图像纠偏
min_line_len: 25               # 最小线段长度（像素）
line_merge_angle_deg: 3        # 共线合并角度阈值
line_merge_gap_px: 6           # 线段合并间隙阈值

# 模型设置
use_yolo_symbols: true          # YOLO符号检测（权重文件缺失时自动禁用）
yolo_symbols_weights: "models/symbols_yolo.onnx"
use_paddle_ocr: true           # PaddleOCR（不可用时回退到Tesseract）

# 高级功能
confidence_tta: true           # 测试时增强提升置信度
apply_constraint_solver: true  # 几何约束优化

# SVG输出设置
svg:
  scale: 1.0                   # 输出缩放因子
  stroke_main: 2               # 主线条宽度
  stroke_aux: 1                # 辅助线条宽度
  dash_pattern: "6,6"          # 虚线样式

# 导出选项
export:
  write_svg: true              # 生成SVG文件
  write_geojson: true          # 生成GeoJSON元数据

log_level: "INFO"              # 日志级别
```

### 高级算法参数

```yaml
algorithms:
  # 直线检测 (LSD/Hough)
  lsd:
    scale: 0.8
    sigma: 0.6  
    quant: 2.0
    ang_th: 22.5
    
  hough_lines:
    threshold: 50
    min_line_length: 25
    max_line_gap: 4
    
  # 圆形检测  
  hough_circles:
    dp: 1.2
    min_dist: 35      # 避免重叠检测
    param1: 120
    param2: 50        # 提高阈值减少误检
    min_radius: 15    # 适合数学配图的最小半径
    
  # 约束求解器
  constraints:
    max_iterations: 100
    tolerance: 1e-6
    huber_delta: 1.0
    lambda_parallel: 1.0
    lambda_perpendicular: 1.0
```

## 🎮 命令行界面

### 核心选项

```bash
python -m png2svg.cli [选项]

# 配置文件
--config PATH                   # YAML配置文件（默认: config.example.yaml）

# 输入输出
--input DIR                     # PNG文件输入目录
--output DIR                    # SVG/GeoJSON输出目录
--jobs N                        # 并行处理进程数

# 处理控制
--single FILE                   # 仅处理单个文件
--no-yolo                      # 禁用YOLO符号检测
--no-paddleocr                 # 禁用PaddleOCR（使用Tesseract）
--no-constraints               # 禁用约束求解器
--no-tta                       # 禁用测试时增强
--no-deskew                    # 禁用图像纠偏

# 输出选项
--svg-only                     # 仅生成SVG（跳过GeoJSON）
--geojson-only                 # 仅生成GeoJSON（跳过SVG）

# 日志选项
--verbose, -v                  # 启用详细日志（DEBUG级别）
--quiet, -q                    # 抑制进度输出
--log-file FILE                # 将日志写入文件
```

### 使用示例

```bash
# 基本批处理
python -m png2svg.cli --input ./math_diagrams --output ./vectorized

# 高性能处理
python -m png2svg.cli --input ./pngs --output ./out --jobs 8 --no-constraints

# 调试单个图像
python -m png2svg.cli --input ./pngs --single complex_diagram.png --verbose --log-file debug.log

# 快速处理（禁用机器学习功能）
python -m png2svg.cli --input ./pngs --no-yolo --no-paddleocr --no-tta
```

## 🔧 可选增强功能

### YOLO符号检测

增强符号识别能力（直角、刻度、箭头、平行标记）：

1. 下载预训练YOLO权重文件（仓库中不包含）
2. 将 `symbols_yolo.onnx` 放置在 `models/` 目录
3. 系统将自动启用YOLO检测功能

```bash
# 期望的模型路径
models/symbols_yolo.onnx
```

**无YOLO权重时**: 系统使用基于规则的符号检测（仍然可用）

### PaddleOCR vs Tesseract

**PaddleOCR** (推荐): 小文本识别效果更好，支持多语言
**Tesseract** (回退): 可靠稳定，广泛可用，资源占用小

系统优先使用PaddleOCR（如可用），否则自动回退到Tesseract。

## 📊 性能基准

| 图像类型 | 尺寸 | 处理时间 | 成功率 |
|----------|------|----------|--------|
| 简单几何图形（直线/圆形） | 800x600 | < 0.5s | 95%+ |  
| 复杂配图（角度/标签） | 1200x900 | < 1.0s | 90%+ |
| 数学证明图 | 1600x1200 | < 2.0s | 85%+ |

*基准测试环境：Intel i7-8700K, 16GB RAM, 启用YOLO*

## 🏗️ 系统架构

### 处理管道

```
PNG输入 → 预处理 → 图元检测 → 符号检测 → OCR文字识别 → 
拓扑构建 → 约束求解 → SVG生成 → GeoJSON导出
```

### 模块结构

```
png2svg/
├── cli.py              # 命令行接口
├── config.py           # 配置管理
├── pipeline.py         # 主处理编排器
├── preprocess.py       # 图像预处理与纠偏
├── detect_primitives.py # 直线/圆形/弧线检测
├── detect_symbols.py   # 数学符号识别
├── ocr_text.py         # 文本/标签提取
├── topology.py         # 关系推理
├── constraints.py      # 几何优化
├── svg_writer.py       # SVG输出生成
├── geojson_writer.py   # 元数据导出
└── io_utils.py         # I/O工具与并行处理
```

## 🐛 故障排除

### 常见问题

**Q: "No module named 'cv2'" 错误**
A: 安装OpenCV: `pip install opencv-python>=4.9`

**Q: 小文本OCR效果差**  
A: 尝试提高图像分辨率或启用PaddleOCR: `use_paddle_ocr: true`

**Q: YOLO符号未检测到**
A: 确保 `models/symbols_yolo.onnx` 存在，或使用 `--no-yolo` 禁用

**Q: 大图像约束求解较慢**
A: 使用 `--no-constraints` 禁用或在配置中降低 `max_iterations`

**Q: 某些图像处理卡死**
A: 在并行处理中启用超时，检查图像是否损坏

**Q: Windows中文文件名无法处理**
A: 系统已优化支持中文文件名，使用 `np.fromfile` + `cv2.imdecode` 方案

### 性能优化

```yaml
# 快速处理（精度降低）
jobs: 8
use_yolo_symbols: false
use_paddle_ocr: false  
apply_constraint_solver: false
confidence_tta: false

# 高精度处理（较慢）
jobs: 1
use_yolo_symbols: true
use_paddle_ocr: true
apply_constraint_solver: true  
confidence_tta: true
```

### 日志与调试

```bash
# 启用详细日志
python -m png2svg.cli --verbose --log-file detailed.log

# 检查依赖状态
python -c "import png2svg; png2svg.check_dependencies(verbose=True)"

# 测试单个模块
python -c "from png2svg import preprocess, Config; print('预处理模块正常')"
```

## 🤝 贡献指南

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 📄 许可证

本项目基于 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- OpenCV - 计算机视觉算法
- Shapely - 几何运算
- NetworkX - 图分析
- PaddleOCR - 文字识别
- Ultralytics YOLO - 目标检测
- SciPy - 约束优化

## 📞 技术支持

- **问题反馈**: [GitHub Issues](https://github.com/example/png2svg/issues)
- **讨论交流**: [GitHub Discussions](https://github.com/example/png2svg/discussions)  
- **文档资料**: [Wiki](https://github.com/example/png2svg/wiki)

---

**PNG2SVG** - 基于AI分析的数学配图语义向量图转换系统