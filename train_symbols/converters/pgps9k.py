"""
PGPS9K数据集转换器
"""
from pathlib import Path
import json
import logging
from typing import List, Dict
from .common import get_image_info, validate_bbox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scan_pgps9k(root: Path, mapping: dict, max_items: int = None) -> List[Dict]:
    """
    扫描PGPS9K数据集并转换为统一格式
    
    Args:
        root: PGPS9K根目录
        mapping: 类别映射字典
        max_items: 最大处理数量（用于调试）
    
    Returns:
        list[{
            "img_path": <str>,
            "width": <int>,
            "height": <int>,
            "annos": [{"name":<统一类名>, "x":<int>, "y":<int>, "w":<int>, "h":<int>}]
        }]
    """
    items = []
    
    # PGPS9K的标注文件
    anno_file = root / 'diagram_annotation.json'
    if not anno_file.exists():
        logger.error(f"Annotation file not found: {anno_file}")
        return items
    
    # 图像目录 - 可能在Diagram或Diagram_Visual目录
    img_dirs = [
        root / 'Diagram',
        root / 'Diagram_Visual',
        root / 'PGPS9K'  # 可能的其他位置
    ]
    
    # 找到存在的图像目录
    img_dir = None
    for d in img_dirs:
        if d.exists():
            img_dir = d
            logger.info(f"Using image directory: {img_dir}")
            break
    
    if img_dir is None:
        logger.error(f"No image directory found in {root}")
        return items
    
    # 读取标注
    with open(anno_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 处理每个图像
    for img_id, anno_data in annotations.items():
        if max_items and len(items) >= max_items:
            break
        
        # 构建图像路径
        img_name = anno_data.get('file_name')
        if not img_name:
            # 尝试使用img_id作为文件名
            img_name = f'{img_id}.png'
        
        img_path = img_dir / img_name
        
        # 如果在第一个目录找不到，尝试其他目录
        if not img_path.exists():
            for d in img_dirs:
                temp_path = d / img_name
                if temp_path.exists():
                    img_path = temp_path
                    break
        
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue
        
        # 获取图像尺寸
        width = anno_data.get('width')
        height = anno_data.get('height')
        
        if width is None or height is None:
            # 尝试从图像文件获取尺寸
            img_info = get_image_info(str(img_path))
            if img_info:
                width, height = img_info
            else:
                logger.warning(f"Failed to get image size for: {img_path}")
                continue
        
        # 处理符号标注
        annos = []
        symbols = anno_data.get('symbols', [])
        
        for symbol in symbols:
            # 获取原始类别
            raw_class = symbol.get('sym_class')
            if not raw_class:
                continue
            
            # 跳过text类别（不是几何符号）
            if raw_class == 'text':
                continue
            
            # 映射到统一类别
            if raw_class not in mapping:
                logger.debug(f"Class '{raw_class}' not in mapping, skipping")
                continue
            
            unified_class = mapping[raw_class]
            
            # 获取bbox (格式: [x, y, w, h])
            bbox = symbol.get('bbox')
            if not bbox or len(bbox) != 4:
                logger.warning(f"Invalid bbox for symbol in {img_path}")
                continue
            
            x, y, w, h = bbox
            
            # 验证bbox
            if not validate_bbox(x, y, w, h, width, height):
                logger.warning(f"Invalid bbox [{x},{y},{w},{h}] in image {img_path} ({width}x{height})")
                continue
            
            annos.append({
                'name': unified_class,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
        
        # 只有当有有效标注时才添加
        if annos:
            items.append({
                'img_path': str(img_path.absolute()),
                'width': width,
                'height': height,
                'annos': annos
            })
    
    logger.info(f"PGPS9K: Loaded {len(items)} images with annotations")
    
    # 统计类别分布
    class_counts = {}
    for item in items:
        for anno in item['annos']:
            class_name = anno['name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    logger.info(f"PGPS9K class distribution: {class_counts}")
    
    return items