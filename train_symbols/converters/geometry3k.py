"""
Geometry3K数据集转换器
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict
from .common import get_image_info, validate_bbox, bbox_xyxy_to_xywh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_xml_annotation(xml_path: Path, mapping: dict) -> List[Dict]:
    """
    解析Pascal VOC格式的XML标注文件
    
    Args:
        xml_path: XML文件路径
        mapping: 类别映射字典
    
    Returns:
        标注列表
    """
    annos = []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像尺寸
        size_elem = root.find('size')
        if size_elem is not None:
            width = int(size_elem.find('width').text)
            height = int(size_elem.find('height').text)
        else:
            width = height = None
        
        # 解析所有object
        for obj in root.findall('object'):
            # 获取类别名
            name_elem = obj.find('name')
            if name_elem is None:
                continue
            
            raw_class = name_elem.text
            
            # 跳过text类别
            if raw_class == 'text':
                continue
            
            # 映射到统一类别
            if raw_class not in mapping:
                logger.debug(f"Class '{raw_class}' not in mapping, skipping")
                continue
            
            unified_class = mapping[raw_class]
            
            # 获取边界框
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            
            try:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # 转换为xywh格式
                x, y, w, h = bbox_xyxy_to_xywh(xmin, ymin, xmax, ymax)
                
                # 验证bbox（如果有图像尺寸信息）
                if width and height:
                    if not validate_bbox(x, y, w, h, width, height):
                        logger.warning(f"Invalid bbox in {xml_path}")
                        continue
                
                annos.append({
                    'name': unified_class,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                })
            except (ValueError, AttributeError) as e:
                logger.warning(f"Error parsing bbox in {xml_path}: {e}")
                continue
        
        return annos, width, height
    
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML file {xml_path}: {e}")
        return [], None, None


def scan_geometry3k(root: Path, mapping: dict, max_items: int = None) -> List[Dict]:
    """
    扫描Geometry3K数据集并转换为统一格式
    
    Args:
        root: Geometry3K根目录
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
    
    # Geometry3K可能有两种目录结构
    # 1. symbols/目录包含所有标注的符号图像
    # 2. train/, val/, test/目录
    
    # 首先尝试symbols目录
    symbols_dir = root / 'symbols'
    if symbols_dir.exists():
        logger.info(f"Processing Geometry3K symbols directory: {symbols_dir}")
        
        # 遍历所有XML文件
        xml_files = list(symbols_dir.glob('*.xml'))
        logger.info(f"Found {len(xml_files)} XML files in symbols directory")
        
        for xml_path in xml_files:
            if max_items and len(items) >= max_items:
                break
            
            # 对应的图像文件
            img_path = xml_path.with_suffix('.png')
            if not img_path.exists():
                img_path = xml_path.with_suffix('.jpg')
            
            if not img_path.exists():
                logger.warning(f"Image not found for {xml_path}")
                continue
            
            # 解析XML获取标注
            annos, xml_width, xml_height = parse_xml_annotation(xml_path, mapping)
            
            if not annos:
                continue
            
            # 获取实际图像尺寸
            if xml_width is None or xml_height is None:
                img_info = get_image_info(str(img_path))
                if img_info:
                    width, height = img_info
                else:
                    logger.warning(f"Failed to get image size for: {img_path}")
                    continue
            else:
                width, height = xml_width, xml_height
            
            items.append({
                'img_path': str(img_path.absolute()),
                'width': width,
                'height': height,
                'annos': annos
            })
    
    # 然后尝试train/val/test目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        
        logger.info(f"Processing Geometry3K {split} directory: {split_dir}")
        
        # 遍历该目录下的所有XML文件
        xml_files = list(split_dir.glob('*.xml'))
        
        for xml_path in xml_files:
            if max_items and len(items) >= max_items:
                break
            
            # 对应的图像文件
            img_path = xml_path.with_suffix('.png')
            if not img_path.exists():
                img_path = xml_path.with_suffix('.jpg')
            
            if not img_path.exists():
                logger.warning(f"Image not found for {xml_path}")
                continue
            
            # 解析XML获取标注
            annos, xml_width, xml_height = parse_xml_annotation(xml_path, mapping)
            
            if not annos:
                continue
            
            # 获取实际图像尺寸
            if xml_width is None or xml_height is None:
                img_info = get_image_info(str(img_path))
                if img_info:
                    width, height = img_info
                else:
                    logger.warning(f"Failed to get image size for: {img_path}")
                    continue
            else:
                width, height = xml_width, xml_height
            
            items.append({
                'img_path': str(img_path.absolute()),
                'width': width,
                'height': height,
                'annos': annos
            })
    
    logger.info(f"Geometry3K: Loaded {len(items)} images with annotations")
    
    # 统计类别分布
    class_counts = {}
    for item in items:
        for anno in item['annos']:
            class_name = anno['name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    logger.info(f"Geometry3K class distribution: {class_counts}")
    
    return items