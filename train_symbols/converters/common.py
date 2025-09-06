"""
通用工具模块：IO/切分/bbox归一化/可视化
"""
import json
import yaml
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_classes_cfg(yaml_path: Path) -> dict:
    """加载类别配置文件"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_yolo_label(label_path: Path, lines: List[str]):
    """写入YOLO格式标签文件"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def split_train_val_test(items: List[dict], 
                        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                        seed: int = 1234) -> Dict[str, List[dict]]:
    """
    将数据集分割为训练集、验证集和测试集
    
    Args:
        items: 数据项列表
        ratios: (train, val, test)比例
        seed: 随机种子
    
    Returns:
        {'train': [...], 'val': [...], 'test': [...]}
    """
    random.seed(seed)
    random.shuffle(items)
    
    total = len(items)
    train_size = int(total * ratios[0])
    val_size = int(total * ratios[1])
    
    result = {
        'train': items[:train_size],
        'val': items[train_size:train_size + val_size],
        'test': items[train_size + val_size:]
    }
    
    logger.info(f"Dataset split: train={len(result['train'])}, "
                f"val={len(result['val'])}, test={len(result['test'])}")
    
    return result


def bbox_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """
    将bbox从(x1,y1,x2,y2)格式转换为(x,y,w,h)格式
    
    Args:
        x1, y1: 左上角坐标
        x2, y2: 右下角坐标
    
    Returns:
        x, y, w, h: 左上角坐标和宽高
    """
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


def bbox_xywh_to_yolo(x: float, y: float, w: float, h: float,
                      img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    将bbox从(x,y,w,h)格式转换为YOLO格式(cx,cy,nw,nh)
    
    Args:
        x, y: 左上角坐标（像素）
        w, h: 宽高（像素）
        img_width, img_height: 图像尺寸
    
    Returns:
        cx, cy, nw, nh: 归一化的中心点坐标和宽高
    """
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    nw = w / img_width
    nh = h / img_height
    
    # 确保值在[0,1]范围内
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    nw = max(0, min(1, nw))
    nh = max(0, min(1, nh))
    
    return cx, cy, nw, nh


def visualize_bbox(img_path: str, bboxes: List[dict], output_path: str,
                  class_names: List[str] = None):
    """
    可视化边界框
    
    Args:
        img_path: 图像路径
        bboxes: 边界框列表 [{'name': str, 'x': float, 'y': float, 'w': float, 'h': float}]
        output_path: 输出路径
        class_names: 类别名称列表
    """
    img = cv2.imread(img_path)
    if img is None:
        logger.warning(f"Failed to read image: {img_path}")
        return
    
    # 为每个类别分配颜色
    colors = {}
    for i, bbox in enumerate(bboxes):
        name = bbox['name']
        if name not in colors:
            colors[name] = tuple(np.random.randint(0, 255, 3).tolist())
        
        color = colors[name]
        x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])
        
        # 画边界框
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # 画类别标签
        label = name
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x, y - label_size[1] - 4), 
                     (x + label_size[0], y), color, -1)
        cv2.putText(img, label, (x, y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(output_path, img)
    logger.info(f"Visualization saved to: {output_path}")


def validate_bbox(x: float, y: float, w: float, h: float, 
                 img_width: int, img_height: int) -> bool:
    """
    验证边界框是否有效
    
    Args:
        x, y: 左上角坐标
        w, h: 宽高
        img_width, img_height: 图像尺寸
    
    Returns:
        是否有效
    """
    # 检查边界框是否在图像内
    if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
        return False
    
    # 检查宽高是否有效
    if w <= 0 or h <= 0:
        return False
    
    # 检查面积是否太小（小于图像面积的0.0001%）
    min_area = img_width * img_height * 0.000001
    if w * h < min_area:
        return False
    
    return True


def get_image_info(img_path: str) -> Optional[Tuple[int, int]]:
    """
    获取图像尺寸
    
    Args:
        img_path: 图像路径
    
    Returns:
        (width, height) 或 None（如果失败）
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        height, width = img.shape[:2]
        return width, height
    except Exception as e:
        logger.error(f"Failed to get image info for {img_path}: {e}")
        return None


def filter_duplicates(items: List[dict], key_func=lambda x: x['img_path']) -> List[dict]:
    """
    根据指定的key去重
    
    Args:
        items: 数据项列表
        key_func: 获取key的函数
    
    Returns:
        去重后的列表
    """
    seen = set()
    result = []
    for item in items:
        key = key_func(item)
        if key not in seen:
            seen.add(key)
            result.append(item)
    
    if len(items) != len(result):
        logger.info(f"Removed {len(items) - len(result)} duplicate items")
    
    return result