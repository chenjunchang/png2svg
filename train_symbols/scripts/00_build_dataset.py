"""
数据集构建脚本
扫描三个数据集，统一转换为YOLO格式
"""
import sys
import yaml
import json
import shutil
import random
import argparse
import logging
from pathlib import Path
from jinja2 import Template

# 添加父目录到Python路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

from converters.pgdp5k import scan_pgdp5k
from converters.pgps9k import scan_pgps9k
from converters.geometry3k import scan_geometry3k
from converters.common import (
    write_yolo_label, 
    split_train_val_test, 
    load_classes_cfg,
    bbox_xywh_to_yolo,
    filter_duplicates
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Build YOLO dataset from multiple sources')
    parser.add_argument('--max-per-dataset', type=int, default=None, 
                       help='Maximum images per dataset (for debugging)')
    parser.add_argument('--seed', type=int, default=1234, 
                       help='Random seed for splitting')
    parser.add_argument('--ratios', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                       help='Train/Val/Test split ratios')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 项目根目录
    project_root = Path(__file__).resolve().parents[2]  # png2svg根目录
    train_root = Path(__file__).resolve().parents[1]   # train_symbols目录
    
    # 输入输出目录
    eval_dir = project_root / "eval"
    out_dir = project_root / "datasets" / "pg_symbols"
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    
    # 创建目录结构
    for split in ['train', 'val', 'test']:
        (out_img / split).mkdir(parents=True, exist_ok=True)
        (out_lbl / split).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {out_dir}")
    
    # 加载类别配置
    classes_cfg_path = train_root / "configs" / "classes.yaml"
    classes = load_classes_cfg(classes_cfg_path)
    names = classes["names"]
    mapping = classes["mapping"]
    
    logger.info(f"Loaded {len(names)} classes: {names}")
    
    # 收集所有数据
    pool = []
    
    # 1. 扫描PGDP5K
    pgdp5k_root = eval_dir / "PGDP5K"
    if pgdp5k_root.exists():
        logger.info(f"Scanning PGDP5K from {pgdp5k_root}")
        pgdp5k_items = scan_pgdp5k(pgdp5k_root, mapping["pgdp5k"], args.max_per_dataset)
        pool.extend(pgdp5k_items)
        logger.info(f"PGDP5K: Added {len(pgdp5k_items)} images")
    else:
        logger.warning(f"PGDP5K directory not found: {pgdp5k_root}")
    
    # 2. 扫描PGPS9K
    pgps9k_root = eval_dir / "PGPS9K"
    if pgps9k_root.exists():
        logger.info(f"Scanning PGPS9K from {pgps9k_root}")
        pgps9k_items = scan_pgps9k(pgps9k_root, mapping["pgps9k"], args.max_per_dataset)
        pool.extend(pgps9k_items)
        logger.info(f"PGPS9K: Added {len(pgps9k_items)} images")
    else:
        logger.warning(f"PGPS9K directory not found: {pgps9k_root}")
    
    # 3. 扫描Geometry3K
    geometry3k_root = eval_dir / "Geometry3K"
    if geometry3k_root.exists():
        logger.info(f"Scanning Geometry3K from {geometry3k_root}")
        geometry3k_items = scan_geometry3k(geometry3k_root, mapping["geometry3k"], args.max_per_dataset)
        pool.extend(geometry3k_items)
        logger.info(f"Geometry3K: Added {len(geometry3k_items)} images")
    else:
        logger.warning(f"Geometry3K directory not found: {geometry3k_root}")
    
    if not pool:
        logger.error("No data found! Please check your dataset directories.")
        return
    
    # 去重（基于图像路径）
    pool = filter_duplicates(pool)
    logger.info(f"Total unique images: {len(pool)}")
    
    # 分割数据集
    items = split_train_val_test(pool, ratios=tuple(args.ratios), seed=args.seed)
    
    # 处理每个split
    total_annotations = 0
    for split, split_items in items.items():
        logger.info(f"Processing {split} split with {len(split_items)} images")
        
        for rec in split_items:
            src = Path(rec["img_path"])
            
            # 生成唯一的文件名（避免冲突）
            # 使用数据集前缀+原始文件名
            dataset_prefix = src.parent.parent.name[:3].lower()  # pgd, pgp, geo
            dst_name = f"{dataset_prefix}_{src.name}"
            dst_img = out_img / split / dst_name
            
            # 复制图像
            try:
                shutil.copyfile(src, dst_img)
            except Exception as e:
                logger.error(f"Failed to copy {src} to {dst_img}: {e}")
                continue
            
            # 生成YOLO格式标签
            yolo_lines = []
            W, H = rec["width"], rec["height"]
            
            for anno in rec["annos"]:
                if anno["name"] not in names:
                    logger.warning(f"Unknown class {anno['name']}, skipping")
                    continue
                
                cls_id = names.index(anno["name"])
                
                # 转换为YOLO格式（归一化的cx,cy,w,h）
                cx, cy, nw, nh = bbox_xywh_to_yolo(
                    anno["x"], anno["y"], anno["w"], anno["h"], W, H
                )
                
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                total_annotations += 1
            
            # 写入标签文件
            if yolo_lines:
                label_file = out_lbl / split / (dst_name.rsplit('.', 1)[0] + ".txt")
                write_yolo_label(label_file, yolo_lines)
    
    logger.info(f"Total annotations written: {total_annotations}")
    
    # 生成dataset.yaml
    template_path = train_root / "configs" / "dataset.yaml.j2"
    with open(template_path, 'r', encoding='utf-8') as f:
        tmpl = Template(f.read())
    
    # Windows路径处理
    root_str = str(out_dir.resolve()).replace("\\", "/")
    yml_content = tmpl.render(root=root_str, names=names)
    
    dataset_yaml_path = out_dir / "dataset.yaml"
    with open(dataset_yaml_path, 'w', encoding='utf-8') as f:
        f.write(yml_content)
    
    logger.info(f"Dataset YAML created at: {dataset_yaml_path}")
    
    # 统计类别分布
    logger.info("\n" + "="*50)
    logger.info("Dataset Statistics:")
    logger.info("="*50)
    
    class_counts = {name: 0 for name in names}
    for item in pool:
        for anno in item['annos']:
            if anno['name'] in class_counts:
                class_counts[anno['name']] += 1
    
    # 按数量排序并显示
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in sorted_counts:
        if count > 0:
            logger.info(f"  {class_name:20s}: {count:6d}")
    
    logger.info("="*50)
    logger.info(f"Dataset successfully built at: {out_dir}")
    logger.info("Next step: Run 'python scripts/01_train.py' to start training")


if __name__ == "__main__":
    main()