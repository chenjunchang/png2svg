下面是一份**可直接交给 codex/cli 生成代码**的实现方案：把 `eval/PGDP5K`, `eval/PGPS9K`, `eval/Geometry3K` 里的标注**统一映射**为 YOLO 检测格式，训练 Ultralytics YOLO 小目标模型，并导出 `png2svg/models/symbols_yolo.onnx`。

---

# 目标产物

* 训练数据集：`datasets/pg_symbols/{images,labels}/{train,val,test}`
* 训练权重：`runs/detect/exp*/weights/best.pt`
* 最终文件：`png2svg/models/symbols_yolo.onnx`（ONNX，dynamic shape）

---

# 目录结构与文件清单

```
project-root/
  eval/
    PGDP5K/            # 原始数据（你已放好）
    PGPS9K/
    Geometry3K/
  png2svg/             # 你的矢量化工程（已存在）
    models/
      # ← 目标把 symbols_yolo.onnx 放到这里
  train_symbols/
    README.md
    requirements.txt
    configs/
      dataset.yaml.j2            # YOLO dataset.yaml 模板
      classes.yaml               # 统一类别&映射配置
      train.small.yaml           # 训练超参（小规模-快速版）
      train.full.yaml            # 训练超参（全量-高精度）
    converters/
      common.py                  # 通用工具：IO/切分/bbox归一化/可视化
      pgdp5k.py                  # 把 PGDP5K → YOLO
      pgps9k.py                  # 把 PGPS9K → YOLO
      geometry3k.py              # 把 Geometry3K → YOLO（仅其“diagram symbols”子集）
    scripts/
      00_build_dataset.py        # 扫描 eval/* → 统一映射 → 拆分 → 生成 YOLO 数据集
      01_train.py                # 训练（Ultralytics）
      02_export_onnx.py          # 导出 ONNX 并拷贝到 png2svg/models
      03_sanity_check.py         # 画预测结果（抽检 N 张）
      04_stats.py                # 类别分布与长尾统计
    tools/
      draw_box.py                # 辅助可视化（可选）
    Makefile
```

---

# 统一类别与映射（configs/classes.yaml）

> 结合数学题常见符号，建议 12 类（可按需增减）。各数据集的原始类名**映射**到这 12 类即可。

```yaml
# 目标检测类别（顺序=class id）
names:
  - right_angle         # 直角小方块
  - arc_1               # 角弧-单弧
  - arc_2               # 角弧-双弧
  - arc_3               # 角弧-三弧
  - tick_1              # 等长刻度-一道
  - tick_2              # 等长刻度-两道
  - tick_3              # 等长刻度-三道
  - parallel_mark       # 平行标记（斜杠对）
  - arrow_head          # 箭头头部（向量/指示）
  - dot_filled          # 实心点
  - dot_hollow          # 空心点
  - extend_mark         # 延长线标记（若数据没有，可保留空类）

# 各数据集→统一类的映射（多对一允许；缺失类自动忽略）
mapping:
  pgdp5k:
    perpendicular: right_angle
    angle-1: arc_1
    angle-2: arc_2
    angle-3: arc_3
    bar-1: tick_1
    bar-2: tick_2
    bar-3: tick_3
    parallel: parallel_mark
    arrow_head: arrow_head
    head: arrow_head
    dot_filled: dot_filled
    dot_hollow: dot_hollow
  pgps9k:
    perpendicular: right_angle
    angle-1: arc_1
    angle-2: arc_2
    angle-3: arc_3
    bar-1: tick_1
    bar-2: tick_2
    bar-3: tick_3
    parallel: parallel_mark
    arrow_head: arrow_head
    head: arrow_head
    dot_filled: dot_filled
    dot_hollow: dot_hollow
  geometry3k:
    # 按实际字段名映射（示例；以你本地的类名为准）
    right_angle_mark: right_angle
    angle_mark_1: arc_1
    angle_mark_2: arc_2
    angle_mark_3: arc_3
    equal_tick_1: tick_1
    equal_tick_2: tick_2
    equal_tick_3: tick_3
    parallel_mark: parallel_mark
    arrow_head: arrow_head
    point_filled: dot_filled
    point_hollow: dot_hollow
```

---

# 依赖（train\_symbols/requirements.txt）

```txt
ultralytics>=8.3
onnxruntime>=1.17
opencv-python>=4.9
numpy>=1.24
pillow>=10.0
pyyaml>=6.0
tqdm>=4.66
jinja2>=3.1
matplotlib>=3.8
```

安装：

```bash
cd train_symbols
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

# 数据集构建（scripts/00\_build\_dataset.py）

**职责**：扫描 `eval/PGDP5K`, `eval/PGPS9K`, `eval/Geometry3K`，调用各 `converters/*.py` 读取原始标注，按 `configs/classes.yaml` 进行**类映射**，输出 YOLO `images/` & `labels/`，并根据全局随机种子做 8/1/1（train/val/test）切分；生成 `dataset.yaml`。

关键点：

* 坐标一律写成 YOLO 归一化 `cx cy w h`。
* 若某数据集缺失某些类，记录日志，不中断。
* 同名图像去重；同图多框全部写入。
* 支持 `--max-per-dataset`（抽样加速初跑）。

示例代码（核心逻辑片段）：

```python
# scripts/00_build_dataset.py
import yaml, json, shutil, random
from pathlib import Path
from jinja2 import Template
from converters.pgdp5k import scan_pgdp5k
from converters.pgps9k import scan_pgps9k
from converters.geometry3k import scan_geometry3k
from converters.common import write_yolo_label, split_train_val_test, load_classes_cfg

random.seed(1234)

def main():
    root = Path(__file__).resolve().parents[1]
    eval_dir = root / "eval"
    out_dir  = root / "datasets" / "pg_symbols"
    out_img  = out_dir / "images"
    out_lbl  = out_dir / "labels"
    (out_img / "train").mkdir(parents=True, exist_ok=True)
    (out_img / "val").mkdir(parents=True, exist_ok=True)
    (out_img / "test").mkdir(parents=True, exist_ok=True)
    (out_lbl / "train").mkdir(parents=True, exist_ok=True)
    (out_lbl / "val").mkdir(parents=True, exist_ok=True)
    (out_lbl / "test").mkdir(parents=True, exist_ok=True)

    classes = load_classes_cfg(root / "configs" / "classes.yaml")
    names   = classes["names"]
    mapping = classes["mapping"]

    pool = []
    pool += scan_pgdp5k(eval_dir/"PGDP5K", mapping["pgdp5k"])
    pool += scan_pgps9k(eval_dir/"PGPS9K", mapping["pgps9k"])
    pool += scan_geometry3k(eval_dir/"Geometry3K", mapping["geometry3k"])

    # pool: list of dicts: {img_path, width, height, annos:[{name,x,y,w,h}]}
    items = split_train_val_test(pool, ratios=(0.8,0.1,0.1))
    for split, split_items in items.items():
        for rec in split_items:
            src = Path(rec["img_path"])
            dst_img = out_img/split/src.name
            shutil.copyfile(src, dst_img)
            yolo_lines = []
            W, H = rec["width"], rec["height"]
            for a in rec["annos"]:
                if a["name"] not in names: continue
                cls_id = names.index(a["name"])
                # 输入为 x1,y1,w,h（绝对像素）或 x1,y1,x2,y2，依据 converter 保证一致
                x1,y1,w,h = a["x"], a["y"], a["w"], a["h"]
                cx, cy = (x1 + w/2)/W, (y1 + h/2)/H
                nw, nh = w/W, h/H
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            write_yolo_label(out_lbl/split/(src.stem + ".txt"), yolo_lines)

    # 生成 dataset.yaml
    tmpl = Template(Path(root/"configs"/"dataset.yaml.j2").read_text(encoding="utf-8"))
    yml  = tmpl.render(root=str(out_dir.resolve()).replace("\\","/"), names=names)
    Path(out_dir/"dataset.yaml").write_text(yml, encoding="utf-8")
    print("[ok] dataset built at", out_dir)

if __name__ == "__main__":
    main()
```

`configs/dataset.yaml.j2`：

```yaml
path: "{{ root }}"
train: images/train
val: images/val
test: images/test
names:
{% for n in names %}  - {{ n }}
{% endfor %}
```

**各 converter 扫描函数约定**（以 PGDP5K 为例）：

```python
# converters/pgdp5k.py
from pathlib import Path
import json, cv2

def scan_pgdp5k(root: Path, mapping: dict):
    """
    返回 list[{
      "img_path": <str>,
      "width": <int>,
      "height": <int>,
      "annos": [{"name":<统一类名>, "x":<int>, "y":<int>, "w":<int>, "h":<int>}]
    }]
    """
    items = []
    # 依据你本地 PGDP5K 的实际结构，遍历图像与其 JSON/XML 标注
    # 伪代码：
    # for img in root.glob("images/**/*.png"):
    #     j = root/"annotations"/img.with_suffix(".json").name
    #     data = json.load(open(j, "r", encoding="utf-8"))
    #     im = cv2.imread(str(img)); H,W = im.shape[:2]
    #     annos=[]
    #     for s in data["symbols"]:
    #         raw = s["class"]
    #         if raw not in mapping: continue
    #         x,y,w,h = s["bbox"]  # 若是 x1,y1,x2,y2 则自行转换
    #         annos.append({"name": mapping[raw], "x":x, "y":y, "w":w, "h":h})
    #     items.append({"img_path": str(img), "width":W, "height":H, "annos":annos})
    return items
```

> `pgps9k.py` 与 `geometry3k.py` 同理：读各自标注，**按映射表改名**，输出一致结构。若某数据集无对应字段，**跳过并记录日志**。

---

# 训练（scripts/01\_train.py）

两套配置（都可改 CLI 参）：

* **small**：快速出结果，验证流程
* **full**：更佳召回/精度（小目标）

```python
# scripts/01_train.py
import argparse, yaml
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../datasets/pg_symbols/dataset.yaml")
    ap.add_argument("--model", default="yolo11n.pt")   # 或 yolo11s.pt
    ap.add_argument("--cfg", default="../configs/train.full.yaml")
    args = ap.parse_args()

    h = yaml.safe_load(Path(args.cfg).read_text())
    y = YOLO(args.model)
    y.train(
        data=args.data,
        imgsz=h.get("imgsz", 1024),
        epochs=h.get("epochs", 150),
        batch=h.get("batch", 16),
        optimizer=h.get("optimizer", "AdamW"),
        lr0=h.get("lr0", 1e-3),
        cos_lr=h.get("cos_lr", True),
        mosaic=h.get("mosaic", 0.7),
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.05, scale=0.9,
        fliplr=0.5,
        plots=True,
        patience=25
    )

if __name__ == "__main__":
    main()
```

`configs/train.small.yaml`：

```yaml
imgsz: 960
epochs: 60
batch: 16
optimizer: AdamW
lr0: 0.001
mosaic: 0.5
```

`configs/train.full.yaml`：

```yaml
imgsz: 1280         # 小目标建议更大分辨率
epochs: 160
batch: 16
optimizer: AdamW
lr0: 0.001
mosaic: 0.7
```

---

# 导出 ONNX（scripts/02\_export\_onnx.py）

```python
# scripts/02_export_onnx.py
import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    root = Path(__file__).resolve().parents[1]
    pt = max((root/"runs"/"detect").glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    onnx_out = pt.parent/"best.onnx"
    YOLO(str(pt)).export(format="onnx", dynamic=True, opset=12, simplify=True)
    dst = root.parent/"png2svg"/"models"/"symbols_yolo.onnx"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(onnx_out, dst)
    print("[ok] onnx exported to", dst)

if __name__ == "__main__":
    main()
```

---

# 训练后抽检（scripts/03\_sanity\_check.py）

```python
import random, cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO

def main():
    root = Path(__file__).resolve().parents[1]
    onnx = max((root/"runs"/"detect").glob("**/weights/best.onnx"), key=lambda p: p.stat().st_mtime)
    y = YOLO(str(onnx))
    imgs = list((root/"datasets"/"pg_symbols"/"images"/"val").glob("*.png"))
    random.shuffle(imgs)
    for img in imgs[:20]:
        r = y.predict(source=str(img), conf=0.2, iou=0.5, imgsz=1280)
        r[0].save(filename=str(img.with_suffix(".pred.png")))
    print("[ok] saved predictions to *.pred.png")

if __name__ == "__main__":
    main()
```

---

# 统计与类不均衡（scripts/04\_stats.py）

* 扫描 `labels/train/*.txt`，统计每类频次；
* 低频类打印告警，指导是否需要**重采样/Copy-Paste 合成**。

（留空由 codex 生成：读取 YOLO 标签累加 class id 计数 → matplotlib 画条形图）

---

# Makefile（快捷命令）

```makefile
.PHONY: build train export check

build:
\tpython scripts/00_build_dataset.py

train:
\tpython scripts/01_train.py --cfg ../configs/train.full.yaml

export:
\tpython scripts/02_export_onnx.py

check:
\tpython scripts/03_sanity_check.py
```

---

# 关键实现细节与约束

1. **解析适配**

   * 三个数据集标注字段名不完全一致。各 `converters/*.py` 封装为**纯函数**，读各自 JSON/XML，产出统一结构；遇到缺字段**跳过**并 `logging.warning`。
   * 如果某些图片没有符号标注，只复制图像，不生成标签文件（YOLO 会视为负样本）。

2. **小目标优化**

   * 分辨率≥1024（推荐 1280）。
   * `mosaic` 开启（0.5\~0.7），训练末期可自动关闭（Ultralytics 已内置策略）。
   * 推理时用 `conf≈0.15~0.25` + `iou≈0.5`，交给上游几何规则二次筛选（你的 `png2svg` 会做绑定）。

3. **合法合规**

   * `eval/*` 数据仅用于**学术/实验**；如需商用，请自行确认授权。脚本不内嵌下载逻辑，**仅离线读取本地数据**。

---

# 端到端执行流程

```bash
cd train_symbols
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) 构建数据集（会自动读取 eval 三个目录）
python scripts/00_build_dataset.py

# 2) 训练（可先 small，再 full）
python scripts/01_train.py --cfg ../configs/train.small.yaml
python scripts/01_train.py --cfg ../configs/train.full.yaml

# 3) 导出 ONNX 并拷贝到 png2svg/models
python scripts/02_export_onnx.py

# 4) 抽检可视化
python scripts/03_sanity_check.py
```

完成后，你将在 `png2svg/models/symbols_yolo.onnx` 得到**可直接被 png2svg 管线加载**的模型文件。如果后续你希望加入更多符号类别（如“相似/全等标记”、坐标轴箭头等），只需要在 `configs/classes.yaml` 中扩展 `names` 与各数据集的 `mapping`，重新执行 `build → train → export` 即可。
