# 目标

把指定目录下的所有 **PNG 数学配图**自动转换为**带语义图层的 SVG**（并同时导出一个 `geo.json` 元数据），在**不训练数据**的情况下即可运行；若放入可选的预训练权重，则自动启用更强的符号识别。

---

# 1) 目录与文件布局

```
png2svg/
  __init__.py
  cli.py                  # 命令行入口
  config.py               # 读取/校验配置
  pipeline.py             # Orchestrator：按阶段串联
  io_utils.py             # 读写工具、并行、日志
  preprocess.py           # 预处理与纠偏
  detect_primitives.py    # 直线/圆/弧/虚线判定
  detect_symbols.py       # 可选：YOLO 小目标（直角标、等长刻度、平行、箭头、多弧）
  ocr_text.py             # 文本与标签识别（PaddleOCR/Tesseract）
  topology.py             # 拓扑关系与交点、箭头方向、线段合并
  constraints.py          # 几何约束构建与最小二乘求解（可选）
  svg_writer.py           # SVG 生成（含图层、样式、marker）
  geojson_writer.py       # 导出结构化语义
  models/
    symbols_yolo.onnx     # （可选）符号检测权重；没有则降级为规则法
  styles/
    default.css           # SVG 内联/外链样式（可选）
  examples/
    demo.png
requirements.txt
config.example.yaml
README.md
```

---

# 2) 依赖（requirements.txt）

```txt
opencv-python>=4.9
numpy>=1.24
scipy>=1.11            # 约束求解（可选）
shapely>=2.0
networkx>=3.2
svgwrite>=1.4
svgelements>=1.9
tqdm>=4.66
pydantic>=2.7          # 配置校验
PyYAML>=6.0
pytesseract>=0.3.10    # 作为 OCR 回退
# 可选：
ultralytics>=8.2       # YOLO 推理（或 onnxruntime）
onnxruntime>=1.17
paddleocr>=2.7.0       # 更稳健的小文字/OCR
scikit-image>=0.23     # 细化/骨架化（可选）
```

> 运行环境：Python 3.10+。没有 `paddleocr/ultralytics/onnxruntime` 时，自动降级为**纯传统算法 + Tesseract**。

---

# 3) 配置（config.example.yaml）

```yaml
input_dir: "./inputs"
output_dir: "./out"
jobs: 4                         # 并行处理进程数
deskew: true                    # 自动纠偏
min_line_len: 25
line_merge_angle_deg: 3         # 共线合并角阈
line_merge_gap_px: 6
use_yolo_symbols: true          # 若 models/symbols_yolo.onnx 不存在则自动=false
yolo_symbols_weights: "models/symbols_yolo.onnx"
use_paddle_ocr: true            # 若不可用则退化为 pytesseract
confidence_tta: true            # 旋转/镜像 TTA
apply_constraint_solver: true   # 约束最小二乘“回正”
svg:
  scale: 1.0
  stroke_main: 2
  stroke_aux: 1
  dash_pattern: "6,6"
export:
  write_geojson: true
  write_svg: true
log_level: "INFO"
```

---

# 4) CLI 规格（cli.py）

```bash
# 基本用法（读取 config.yaml）
python -m png2svg.cli --config config.yaml

# 指定输入/输出并覆盖配置
python -m png2svg.cli --input ./pngs --output ./out --jobs 8 --no-constraints

# 仅验证一张
python -m png2svg.cli --input ./pngs --output ./out --single demo.png
```

支持参数：

* `--config PATH`：加载 YAML 配置
* `--input DIR` / `--output DIR` / `--jobs N`
* `--no-yolo`, `--no-paddleocr`, `--no-constraints`：快速关闭模块
* `--single FILE`：仅处理单张
* 退出码：0 成功；≠0 表示处理过程中存在严重错误（并打印失败清单）

---

# 5) 核心流程（pipeline.py）

```python
def process_image(path: str, cfg: Cfg) -> Result:
    img = read_image(path)
    pre = preprocess.run(img, cfg)           # 灰度、去噪、二值、deskew
    prim = detect_primitives.run(pre, cfg)   # 线段/圆/弧 + 虚实/粗细
    sym  = detect_symbols.run(pre, cfg)      # 直角标、等长刻度、多弧、∥、箭头（可选）
    txt  = ocr_text.run(img, pre, cfg)       # A/B/C、30°、x+1 等
    topo = topology.build(prim, sym, txt, cfg)
    if cfg.apply_constraint_solver:
        topo = constraints.solve(topo, cfg)  # soft constraints：∥、⊥、共线、等长、点在圆
    svg_path = svg_writer.write(path, topo, cfg)
    geo_path = geojson_writer.write(path, topo, cfg)
    return Result(svg=svg_path, geo=geo_path)
```

---

# 6) 关键模块与函数签名（供代码生成）

### 6.1 预处理（preprocess.py）

```python
@dataclass
class PreprocessOut:
    img_gray: np.ndarray     # uint8
    img_bw: np.ndarray       # 二值（前景=255）
    transform: np.ndarray    # 2x3 仿射矩阵（deskew）

def run(img_bgr: np.ndarray, cfg: Cfg) -> PreprocessOut:
    """
    - 灰度 + 轻度高斯
    - 自适应阈值 (THRESH_BINARY_INV)
    - 可选：霍夫直线主方向估计 + 旋转纠偏
    - 返回二值图与 deskew 变换
    """
```

### 6.2 原语检测（detect\_primitives.py）

```python
@dataclass
class LineSeg:  # 坐标均为 float
    p1: Tuple[float,float]
    p2: Tuple[float,float]
    dashed: bool = False
    thickness: int = 2
    role: str = "main"       # main/aux/hidden (根据粗细/虚实)
    id: str = ""

@dataclass
class CircleArc:
    cx: float; cy: float; r: float
    theta1: float = 0.0; theta2: float = 360.0  # 弧段时使用
    kind: str = "circle"  # circle/arc
    id: str = ""

@dataclass
class Primitives:
    lines: list[LineSeg]
    circles: list[CircleArc]

def run(pre: PreprocessOut, cfg: Cfg) -> Primitives:
    """
    - LSD 优先，失败则 HoughLinesP
    - 合并共线（角度<cfg.line_merge_angle_deg 且端点距<cfg.line_merge_gap_px）
    - HoughCircles 检圆；对边缘拟合弧（通过 Canny + RANSAC/最小二乘）
    - is_dashed(): 采样线段像素值的 on/off 运行长度判断
    """
```

**is\_dashed 伪代码：**

```
sample L points along segment -> binarized values (0/255)
count runs -> on_runs >=2 and mean(on_run_len)>k && has off_runs -> dashed=True
```

### 6.3 符号检测（detect\_symbols.py）

```python
@dataclass
class Symbol:
    cls: str                 # right_angle / arc_1 / arc_2 / arc_3 / tick_1/2/3 / parallel / arrow_head / dot_filled / dot_hollow
    bbox: tuple[int,int,int,int]
    conf: float

@dataclass
class Symbols:
    items: list[Symbol]

def run(pre: PreprocessOut, cfg: Cfg) -> Symbols:
    """
    - 若找到 onnx 权重则用 onnxruntime 推理
    - 否则：回退规则：模板匹配/形状近似（如直角小方块=小矩形 + 与两线段近正交）
    - 输出统一的 Symbol 列表
    """
```

### 6.4 OCR（ocr\_text.py）

```python
@dataclass
class OCRItem:
    text: str
    bbox: tuple[int,int,int,int]
    conf: float

def run(img_bgr: np.ndarray, pre: PreprocessOut, cfg: Cfg) -> list[OCRItem]:
    """
    - 优先 PaddleOCR；不可用时用 pytesseract
    - 角度/单位字符（°、rad）、下标/撇号（A′）做正则清洗
    - 结果带 bbox 与置信度
    """
```

### 6.5 拓扑与关系（topology.py）

```python
@dataclass
class Node:   # 点（端点/交点/圆心/垂足候选等）
    x: float; y: float
    tag: str = ""           # A/B/C/O/H/M...
    kind: str = "point"     # point/mid/foot/center/vertex/...
    id: str = ""

@dataclass
class Edge:   # 线/弧的语义化
    geom: Union[LineSeg, CircleArc]
    role: str = "main"      # main/aux/hidden
    attrs: dict = field(default_factory=dict)

@dataclass
class Graph:
    nodes: list[Node]
    edges: list[Edge]
    relations: list[dict]   # {type: 'parallel'|'perp'|'equal_len'|'point_on', members:[...], conf:float}

def build(prim: Primitives, sym: Symbols, ocr: list[OCRItem], cfg: Cfg) -> Graph:
    """
    - 用 Shapely 计算所有交点/端点，生成节点
    - 绑定符号与最近线段/角：直角标->垂直关系；tick_n->等长分组；arc_k->等角标记；parallel->平行
    - 箭头与线的方向绑定
    - OCR 将点名/角度/长度分类并绑定到最近元素（带阈值）
    - 生成初始 relations：parallel/perp/point_on/equal_len/angle_group/...
    """
```

### 6.6 约束求解（constraints.py）

```python
def solve(g: Graph, cfg: Cfg) -> Graph:
    """
    - 构造变量：关键节点坐标
    - 目标：min Σ(data_term) + λ Σ(soft_constraints)
      data_term: 节点到观测几何（像素）的距离
      constraints: ∥, ⊥, 共线, 点在圆, 等长, 角度固定(若有数字)
    - SciPy least_squares + Huber loss
    - 解完回写节点坐标 & 派生边的端点（让 SVG 更“干净/正”）
    """
```

### 6.7 SVG 输出（svg\_writer.py）

```python
def write(img_path: str, g: Graph, cfg: Cfg) -> str:
    """
    - <g id="main"> 主图；<g id="aux"> 辅助虚线
    - 统一 style；虚线用 stroke-dasharray
    - 定义 <marker id="arrow"> 箭头；直角标/刻度用 path 或小 group
    - 所有元素写 data-* 语义：data-role, data-cls, data-conf, data-rel
    - 坐标以图像像素为单位（可乘 cfg.svg.scale）
    - 返回 SVG 文件路径
    """
```

### 6.8 GeoJSON 导出（geojson\_writer.py）

```python
def write(img_path: str, g: Graph, cfg: Cfg) -> str:
    """
    导出：
    - nodes: [{id,x,y,tag,kind}]
    - edges: [{id,type:'line'|'arc'|'circle', endpoints|center+radius, role}]
    - relations: [{type, members, conf}]
    - ocr: [{text,bbox,conf,bind_to}]
    """
```

---

# 7) 置信融合（可选 TTA）

在 `detect_symbols.run` 与 `ocr_text.run` 内部实现 TTA：

* 对输入图像执行 {原图, 水平翻转, 旋转 ±90/180} 推理
* 把检测/文本按坐标逆变换回原图坐标系
* 用 IoU/NMS（检测）与字符编辑距离（OCR）做合并与投票，得到 `conf` 提升后的结果

---

# 8) 并行与错误处理（io\_utils.py）

* 使用 `multiprocessing.Pool(jobs)`；每张图超时（例如 60s）即记录失败并继续
* 日志采用 `logging`，为每张图片打印阶段耗时与统计
* 遇到 OCR/YOLO 不可用时，打印 **WARN** 并自动降级

---

# 9) 最小可用版本（MVP）验收标准

1. **输入**：`input_dir` 中 ≥1 张 png
2. **输出**：每张图对应一个 `xxx.svg` 与 `xxx.geo.json`
3. **能力**：

   * 直线/圆基本可见（粗细/虚实区分）
   * 文本（A/B/C、长度/角度）被识别并写入 SVG 的 `data-*` 与 geojson
   * 生成少量基本关系（∥/⊥/point\_on）
   * 若放置 `models/symbols_yolo.onnx`，出现直角标/刻度/多弧/平行/箭头的解析与绑定
4. **稳定性**：对 100 张图整体处理不中断，失败个别样本但不影响批处理完成
5. **SVG 可读性**：在浏览器中清晰显示，虚线/箭头/文字位置合理

---

# 10) 代码片段（示例：CLI 与主流程骨架）

**cli.py**

```python
import argparse, sys, traceback
from png2svg.config import load_config
from png2svg.io_utils import list_pngs, run_parallel
from png2svg.pipeline import process_image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--input", type=str)
    ap.add_argument("--output", type=str)
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--single", type=str)
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--no-paddleocr", action="store_true")
    ap.add_argument("--no-constraints", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args)
    files = [args.single] if args.single else list_pngs(cfg.input_dir)
    results = run_parallel(files, process_image, cfg, jobs=cfg.jobs)
    failed = [r.path for r in results if not r.ok]
    if failed:
        print(f"[WARN] failed: {len(failed)} items", file=sys.stderr)
        for p in failed: print(" -", p, file=sys.stderr)
        sys.exit(2)
    sys.exit(0)

if __name__ == "__main__":
    main()
```

**detect\_primitives.py（关键函数示例）**

```python
def _detect_lines(gray, bw, cfg) -> list[LineSeg]:
    lines = []
    try:
        lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD)
        detected = lsd.detect(gray)[0]
        if detected is not None:
            for x1,y1,x2,y2 in detected.reshape(-1,4):
                lines.append(LineSeg((float(x1),float(y1)), (float(x2),float(y2))))
    except Exception:
        pass
    if not lines:
        edges = cv2.Canny(gray, 60, 120, apertureSize=3)
        hlines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=cfg.min_line_len, maxLineGap=4)
        if hlines is not None:
            for x1,y1,x2,y2 in hlines.reshape(-1,4):
                lines.append(LineSeg((float(x1),float(y1)), (float(x2),float(y2))))
    # merge & dashed
    lines = merge_collinear(lines, cfg)
    for ln in lines:
        ln.dashed = is_dashed(ln, bw)
        ln.role = "aux" if ln.dashed else "main"
    return lines
```

---

# 11) 扩展（可选，但接口已预留）

* **数据集推理评测**：增加 `eval/`，可在 PGDP5K/PGPS9K 上跑 P/R/F1 与结构一致性
* **局部矢量化精修**：接入 DiffVG/可微矢量化，把边缘拟合到 Bézier，提高弧/箭头边界质量
* **GUI 预览与人工校正**：用 Gradio/Streamlit 做快速标注与导出（SVG 双向编辑）

---

# 12) README 关键点（生成文档时包含）

* 安装步骤（Windows/Linux/Mac）与字体/OCR 依赖（Tesseract/PaddleOCR）
* 放置可选权重文件到 `models/` 的方法（无则自动降级）
* 配置项详解与常见问题（如 OCR 识别差可以放大图片或切换 PaddleOCR）
* 基准性能：单图 <1s（不含 YOLO），开启 YOLO 视模型大小而定
