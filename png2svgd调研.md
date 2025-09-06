下面给你一套**可落地**的技术方案（含原型级 Python 代码骨架），目标是把一批 PNG 数学配图自动“结构化矢量化”为**带语义的 SVG**（包含：斜线、虚线/点划线、直角标记、垂足/中点/圆心、等长刻度、角弧/多弧等）。方案融合「传统几何检测 + 轻量深度学习 + 约束求解 + 可迭代置信融合」，兼顾**先用起来**与**可持续升级**。

---

# 1) 总体蓝图（三层架构）

**输入**：批量 PNG（黑白或少色系配图最佳）
**输出**：与每张图对应的 `xxx.svg` + `xxx.geo.json`（语义/拓扑/置信度）

**Pipeline 六步：**

1. **预处理**：灰度化→去噪→自适应阈值→去伪影/细化；可校正旋转、统一 DPI。
2. **几何原语检测（Hybrid）**

   * 直线/线段：LSD/FLD 或 PHT（HoughLinesP）
   * 圆/弧：HoughCircles + 残差拟合（必要时 ellipse 拟合）
   * 箭头、直角小方块、等长刻度、角弧符号、平行/垂直标记等**离散符号**：用一套小型 **YOLO** 检测器（几十\~数百张合成数据即可起步）
   * 文本/标签（A、B、C、30°、x 等）：OCR（优先 PaddleOCR；或 Tesseract）
3. **分组与拓扑建模**：合并共线段、判定虚实/粗细、提取端点/交点、识别箭头朝向、建立**约束关系图**（incidence/parallel/perp/equal-length/point-on-arc…）。
4. **约束求解与几何“回正”**：以检测到的点为变量，构造带惩罚的最小二乘（并行/垂直/共线/等长/过圆心等约束），做**坐标优化**，得到干净、正交的几何。
5. **SVG 生成（带语义图层）**：用 `<g>` 分图层（主图/辅助线/符号/文字），统一样式与 marker（箭头、虚线、刻度、角弧…），并把每个元素的**语义与置信度**写入 `data-*` 属性，同时输出一个结构化 `geo.json`。
6. **置信融合与二次修正（可选）**：TTA（旋转/镜像）+ 多模型投票；必要时用可微矢量化（如 **DiffVG** / Bézier-splatting）对齐像素观测做末端精修（特别是弧线/箭头边界）。 ([people.csail.mit.edu][1], [GitHub][2], [arXiv][3])

> 关键第三方：OpenCV 的 LSD/FLD/Hough、PaddleOCR/Tesseract、Ultralytics YOLO、Shapely/NetworkX 做几何与图建模、`svgwrite`/`svgelements` 处理 SVG。([docs.opencv.org][4], [paddlepaddle.github.io][5], [GitHub][6], [docs.ultralytics.com][7], [shapely.readthedocs.io][8], [PyPI][9])

---

# 2) 元素→算法映射（针对你列的清单）

* **点/垂足/中点/圆心/重心等**：
  先由线段/圆弧交点 & 语义符号检测提出**候选点**；再用文本（如 O、H、M）与上下文关系确认**角色**。垂足= 点P 在直线l 上，且从顶点落线与 l 垂直（用约束自动判别）。
* **直线/斜线/虚线/延长线**：
  LSD/FLD/Hough 得到线段集合→DBSCAN 按方向聚类→判断虚线（线段间隙周期性）→超出端点方向延长（受文字/箭头/裁剪框限制）。 ([docs.opencv.org][4])
* **垂线/平行**：
  由方向夹角与符号检测双重判定；若有直角小方块/∥ 标记，直接提升约束权重（semantic prior）。
* **圆/弧/切线/割线/法线**：
  HoughCircles + 残差拟合；切/割线依据与圆的交点数量与切点几何判定。 ([docs.opencv.org][10])
* **角与角标记（单/双/三弧、直角标记）**：
  弧线与方块作为**对象检测类**处理（YOLO），角度数字交给 OCR；缺失标记可由夹角阈值与等角多弧标记推断。 ([docs.ultralytics.com][7])
* **等长刻度/勾号、底边刻度/比例标记**：
  统一当作**小目标类别**训练 YOLO；与邻近线段建立“归属”，并在约束中加入 equal-length。 ([docs.ultralytics.com][11])
* **箭头/向量符号/旋转中心**：
  小目标检测 + 与最近线段/圆弧绑定（端点方向一致性）。
* **坐标轴/网格/阴影区域/三维虚线**：
  规则纹理→频域/密度过滤得到网格；阴影区域可由阈值+连通域+填充；三维不可见边默认**虚线样式**导出。
* **文字与标签（A、B、30°、x+1 等）**：
  PaddleOCR（多语种小目标较稳），或 Tesseract；LaTeX 角度/表达式可接 Pix2Text/pix2tex。 ([paddlepaddle.github.io][5], [GitHub][6])

---

# 3) 训练数据与“可即刻使用”的冷启动

**冷启动（0 标注）**：

* 先用**传统方法**（LSD/Hough + OCR）运行整个管线，能可靠输出**直线/圆/点/文字/虚线**与基础 SVG。
* 对**直角小方块/多弧/等长刻度/平行标**等符号，初期可用**模板匹配/形状近似**启用（体验版）。

**快速提升（合成数据）**：

* 用 **TikZ/Manim/GeoGebra** 或自写脚本生成成千上万张**几何图 + 完整标注**，一晚训练 YOLO 小模型即可。
* 开源**PGDP5K**、PGPS9K 等图形解析数据集可做预训练或评测补充。 ([nlpr.ia.ac.cn][12])

**更进一步（研究向，可选）**：

* 引入 **SAM2** 做可提示分割，增强弱标注场景的元素提取；
* 用 **DiffVG / Bézier splatting** 做“像素到矢量”的可微拟合，改善弧/箭头细节与端点贴合。 ([arXiv][13], [ai.meta.com][14], [people.csail.mit.edu][1])

---

# 4) 置信融合与几何一致性

* **TTA 多次推理**（旋转 ±90/180、镜像），把检测与 OCR 结果做 NMS/字典投票，给出**元素级置信度**。
* **约束优化**：把“∥、⊥、点在线上、等长、过圆心”等作为 soft constraints，最小二乘+鲁棒损失（Huber）求解；优化后元素坐标**回写**到 SVG。
* **一致性校验**：把 SVG 渲染回位图，与原图做 SSIM/边缘重合率；若偏差超阈，再次迭代（可选用可微栅格器精修）。 ([people.csail.mit.edu][1])

---

# 5) 批处理原型代码（可直接改造成你的 MVP）

> 说明：此代码骨架**能跑出可用的 SVG**（线/圆/文字/虚线），并预留了 YOLO/PaddleOCR 等接口。安装依赖：`opencv-python shapely svgwrite networkx pytesseract`（如用 PaddleOCR/YOLO，按其官方文档安装）。
>
> * 直线：LSD（若无 `ximgproc`，回退到 HoughLinesP）
> * 圆：HoughCircles
> * OCR：优先 `pytesseract`，你可切换到 PaddleOCR
> * 输出：`out/<name>.svg` 与 `out/<name>.geo.json`

```python
# mvp_vectorize.py
import os, json, math, glob, cv2, numpy as np
import svgwrite
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
import networkx as nx

# ---------- Utils ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def binarize(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    bw = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 35, 15)
    return bw

def detect_lines(gray_or_bw):
    # Try LSD first
    lines = []
    try:
        lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD)
        detected = lsd.detect(gray_or_bw)[0]
        if detected is not None:
            for x1, y1, x2, y2 in detected.reshape(-1,4):
                lines.append(((float(x1),float(y1)), (float(x2),float(y2))))
    except Exception:
        pass
    # Fallback: HoughLinesP
    if not lines:
        edges = cv2.Canny(gray_or_bw, 60, 120, apertureSize=3)
        hlines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                 minLineLength=25, maxLineGap=4)
        if hlines is not None:
            for x1,y1,x2,y2 in hlines.reshape(-1,4):
                lines.append(((float(x1),float(y1)), (float(x2),float(y2))))
    return lines

def detect_circles(gray):
    circles = []
    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                         param1=120, param2=30, minRadius=8, maxRadius=0)
    if c is not None:
        for x,y,r in np.uint16(np.around(c[0,:])):
            circles.append((float(x),float(y),float(r)))
    return circles

def is_dashed(ls_pixels, img_bw):
    # sample along the segment to see on/off pattern; naive but works for MVP
    (x1,y1),(x2,y2)=ls_pixels
    L = int(math.hypot(x2-x1,y2-y1))
    if L < 20: return False
    xs = np.linspace(x1,x2,L).astype(int)
    ys = np.linspace(y1,y2,L).astype(int)
    vals = img_bw[np.clip(ys,0,img_bw.shape[0]-1), np.clip(xs,0,img_bw.shape[1]-1)]
    # Count runs
    runs, last, cnt = [], vals[0], 1
    for v in vals[1:]:
        if v==last: cnt+=1
        else: runs.append((last,cnt)); last=v; cnt=1
    runs.append((last,cnt))
    on_runs = [l for val,l in runs if val>0]
    off_runs= [l for val,l in runs if val==0]
    if len(on_runs)<2: return False
    return (np.mean(on_runs)>3) and (len(off_runs)>=1)

def ocr_text(img, boxes=None):
    # Minimal stub (pytesseract). Replace with PaddleOCR for better multi-lang & small text.
    try:
        import pytesseract
        if boxes is None:
            txt = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            results=[]
            for i in range(len(txt['text'])):
                s=txt['text'][i].strip()
                if s:
                    x,y,w,h = txt['left'][i],txt['top'][i],txt['width'][i],txt['height'][i]
                    results.append({'text':s,'bbox':[x,y,w,h],'conf':float(txt['conf'][i])})
            return results
        else:
            results=[]
            for (x,y,w,h) in boxes:
                crop = img[y:y+h, x:x+w]
                s = pytesseract.image_to_string(crop, config='--psm 7').strip()
                if s:
                    results.append({'text':s,'bbox':[x,y,w,h],'conf':1.0})
            return results
    except Exception:
        return []

def build_topology(lines, circles):
    G = nx.Graph()
    # add vertices at endpoints and intersections
    def add_point(pt):
        key = (round(pt[0],2), round(pt[1],2))
        if key not in G: G.add_node(key, kind='point')
        return key
    segs = [LineString([l[0], l[1]]) for l in lines]
    for i, s in enumerate(segs):
        p1 = add_point(lines[i][0])
        p2 = add_point(lines[i][1])
        G.add_edge(p1,p2, kind='segment', idx=i)
    U = unary_union(segs)
    # naive intersection: check pairwise
    for i in range(len(segs)):
        for j in range(i+1,len(segs)):
            if segs[i].intersects(segs[j]):
                inter = segs[i].intersection(segs[j])
                if "Point" in inter.geom_type:
                    k = add_point((inter.x, inter.y))
                    # connect with zero-length meta edges to store relation
                    G.add_edge(k, (round(lines[i][0][0],2),round(lines[i][0][1],2)), kind='incidence')
    return G

def save_svg(svg_path, w, h, lines, circles, dashed_flags, texts):
    dwg = svgwrite.Drawing(svg_path, size=(w,h))
    # style defs
    defs = dwg.defs
    arrow = defs.add(dwg.marker(id='arrow', insert=(4,2), size=(4,4), orient='auto'))
    arrow.add(dwg.path(d="M0,0 L4,2 L0,4 z"))
    dwg.add(defs)
    # layers
    g_main = dwg.g(id='main', stroke='black', fill='none', stroke_width=2)
    g_aux  = dwg.g(id='aux', stroke='black', fill='none', stroke_width=1)
    # draw lines
    for i, ((x1,y1),(x2,y2)) in enumerate(lines):
        style = {'stroke_dasharray':'6,6'} if dashed_flags[i] else {}
        g = g_aux if dashed_flags[i] else g_main
        g.add(dwg.line((x1,y1),(x2,y2), **style))
    # draw circles
    for (cx,cy,r) in circles:
        g_main.add(dwg.circle(center=(cx,cy), r=r))
    # draw texts
    for t in texts:
        x,y,w,h = t['bbox']
        dwg.add(dwg.text(t['text'], insert=(x, y-2), font_size=14))
    dwg.add(g_main); dwg.add(g_aux)
    dwg.save()

def process_one(png_path, out_dir):
    img = cv2.imread(png_path)
    h,w = img.shape[:2]
    bw = binarize(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = detect_lines(gray)
    circles = detect_circles(gray)
    dashed = [is_dashed(((int(x1),int(y1)),(int(x2),int(y2))), bw) for (x1,y1),(x2,y2) in lines]
    texts  = ocr_text(img)

    topo = build_topology(lines, circles)
    # TODO: 约束优化 + 小目标符号检测（YOLO）+ 角弧/直角/等长刻度归属

    base = os.path.splitext(os.path.basename(png_path))[0]
    ensure_dir(out_dir)
    save_svg(os.path.join(out_dir, f"{base}.svg"), w, h, lines, circles, dashed, texts)

    geo = {'image': base, 'size': [w,h],
           'lines': [{'p1':lines[i][0],'p2':lines[i][1],'dashed':bool(dashed[i])} for i in range(len(lines))],
           'circles':[{'c':(cx,cy),'r':r} for (cx,cy,r) in circles],
           'texts': texts,
           'constraints': [], 'version': 'mvp-0.1'}
    with open(os.path.join(out_dir, f"{base}.geo.json"), 'w', encoding='utf-8') as f:
        json.dump(geo, f, ensure_ascii=False, indent=2)

def batch_process(in_dir, out_dir='out'):
    for p in glob.glob(os.path.join(in_dir, '*.png')):
        process_one(p, out_dir)

if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('input_dir', help='folder of png images')
    ap.add_argument('--out', default='out')
    args = ap.parse_args()
    batch_process(args.input_dir, args.out)
```

> 参考实现要点：
>
> * **LSD/FLD/Hough**：直线/线段检测常用且稳健（OpenCV 自带）。([docs.opencv.org][4])
> * **圆/弧**：HoughCircles 很适合标准圆；弧可由边缘+最小二乘拟合得到。([docs.opencv.org][10])
> * **OCR**：演示用 pytesseract；生产建议换 **PaddleOCR** 提升小字母/多语言识别。([paddlepaddle.github.io][5])
> * **SVG 输出**：`svgwrite`/`svgelements` 常用稳定。([PyPI][9])

---

# 6) 小目标符号检测（YOLO）与数据标注建议

**类别建议（可按需裁剪）：**
`right_angle`（直角小方块）、`arc_1/arc_2/arc_3`（单/双/三弧）、`tick_1/tick_2/tick_3`（等长刻度道数）、`parallel_mark`、`perp_mark`、`arrow_head`、`dot_filled/dot_hollow`、`hatch_region`（阴影）、`extend_mark`（延长线符号）……

**标注来源**：

* 用 TikZ/Manim/GeoGebra 脚本自动合成（背景干净、风格一致，易学）
* 适量真实试卷/习题图像微调
* 参考 PGDP5K 的类别与关系定义以便通用化评测。([nlpr.ia.ac.cn][12])

**训练**：Ultralytics YOLO（v8/11 任一）自定义数据集即可，部署轻量且导出 ONNX/PyTorch 方便与本代码对接。([docs.ultralytics.com][11])

---

# 7) 约束求解（把“画得像”变成“几何正确”）

* 变量：关键点坐标 `P = {(xi, yi)}`
* 目标：最小化 ∑ 观测残差（像素-几何距离） + λ∑ 几何约束

  * ∥：两线段方向向量夹角 → 0
  * ⊥：方向向量点积 → 0
  * 共线/点在线上：点到直线距离 → 0
  * 等长：|AB| - |CD| → 0
  * 圆：|OP| - r → 0
* 解法：SciPy `least_squares` 或 cvxpy（软约束）。优化后回写坐标与**对齐后的 SVG**，显著减少锯齿/断裂。

---

# 8) 评测指标

* **元素级**：P/R/F1（线段/圆/符号/文字），平均端点误差（px）、角度误差（°）、圆心/半径误差。
* **关系级**：∥/⊥/等长 判定准确率；正确约束恢复率。
* **图像级**：SVG→位图 与原图 SSIM/边缘重合率（≥0.9 为佳）。
* **运行效能**：单图时延 & 吞吐（目标：单张 <1s，批量可多进程）。

---

# 9) 落地路线图（4\~6 周内可达成“好用”）

1. **第1周**：跑通 MVP（上面脚本）+ PaddleOCR 接入 + 导出 geo.json/语义属性。
2. **第2\~3周**：合成数据训练 YOLO 小模型（专攻直角/等长/多弧/箭头），集成到 Pipeline。
3. **第4周**：约束求解 & 置信融合（TTA + NMS）+ 基准评测集。
4. **第5\~6周**（可选）：接入 **SAM2**/DiffVG 精修弧线与复杂区域；补充 PGDP5K 对比评测。 ([arXiv][13], [people.csail.mit.edu][1])

---

# 10) 风险与规避

* **图片质量差/扫描噪声**：加强预处理与自适应阈值；尝试细化/骨架化提升线性要素提取稳定性。([scikit-image.org][15])
* **符号风格多样**：优先合成风格统一的训练集，逐步引入真实数据微调。
* **复杂三维/阴影密集**：先只抽主几何+关键虚线，把区域填充作为附加图层输出。
* **OCR 对小字母不稳**：切换 PaddleOCR 小模型 & 放大裁剪块；必要时加上字符级后验检查。([paddlepaddle.github.io][5])

---

## 参考与依据（核心外部要点）

* OpenCV 直线/线段与圆检测：LSD/FLD/HoughCircles。([docs.opencv.org][4])
* PaddleOCR / Tesseract：通用 OCR 与 Python 封装。([paddlepaddle.github.io][5], [GitHub][16])
* Ultralytics YOLO（自定义小目标检测）。([docs.ultralytics.com][11])
* SVG 生成与解析：`svgwrite` / `svgelements`。([PyPI][9])
* 高质量几何图数据集：PGDP5K / PGPS9K。([nlpr.ia.ac.cn][12])
* 可微矢量化与精修：DiffVG、Bézier Splatting（2025）。([people.csail.mit.edu][1], [arXiv][3])
* 骨架/细化等影像学工具：scikit-image skeletonize\&Hough。([scikit-image.org][15])

---


[1]: https://people.csail.mit.edu/tzumao/diffvg/?utm_source=chatgpt.com "Differentiable Vector Graphics Rasterization for Editing ..."
[2]: https://github.com/BachiLi/diffvg?utm_source=chatgpt.com "BachiLi/diffvg: Differentiable Vector Graphics Rasterization"
[3]: https://arxiv.org/abs/2503.16424?utm_source=chatgpt.com "Bézier Splatting for Fast and Differentiable Vector Graphics"
[4]: https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html?utm_source=chatgpt.com "cv::LineSegmentDetector Class Reference"
[5]: https://paddlepaddle.github.io/PaddleOCR/main/en/index.html?utm_source=chatgpt.com "Home - PaddleOCR Documentation"
[6]: https://github.com/tesseract-ocr/tesseract?utm_source=chatgpt.com "Tesseract Open Source OCR Engine (main repository)"
[7]: https://docs.ultralytics.com/tasks/detect/?utm_source=chatgpt.com "Object Detection - Ultralytics YOLO Docs"
[8]: https://shapely.readthedocs.io/?utm_source=chatgpt.com "Shapely — Shapely 2.1.1 documentation"
[9]: https://pypi.org/project/svgwrite/?utm_source=chatgpt.com "svgwrite"
[10]: https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html?utm_source=chatgpt.com "Hough Circle Transform"
[11]: https://docs.ultralytics.com/?utm_source=chatgpt.com "Home - Ultralytics YOLO Docs"
[12]: https://nlpr.ia.ac.cn/databases/CASIA-PGDP5K/index.html?utm_source=chatgpt.com "CASIA-PGDP5K: Plane Geometry Diagram Parsing Dataset"
[13]: https://arxiv.org/abs/2408.00714?utm_source=chatgpt.com "SAM 2: Segment Anything in Images and Videos"
[14]: https://ai.meta.com/sam2/?utm_source=chatgpt.com "Introducing Meta Segment Anything Model 2 (SAM 2)"
[15]: https://scikit-image.org/docs/0.25.x/auto_examples/edges/plot_skeleton.html?utm_source=chatgpt.com "Skeletonize — skimage 0.25.2 documentation"
[16]: https://github.com/madmaze/pytesseract?utm_source=chatgpt.com "A Python wrapper for Google Tesseract"
