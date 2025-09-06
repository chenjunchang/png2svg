"""
抽检验证脚本
对训练好的模型进行可视化验证
"""
import sys
import random
import logging
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_class_names(config_path: Path) -> list:
    """加载类别名称"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('names', [])


def main():
    parser = argparse.ArgumentParser(description='Sanity check for trained YOLO model')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (.pt or .onnx)')
    parser.add_argument('--data', type=str, default='../datasets/pg_symbols',
                       help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                       help='Dataset split to test on')
    parser.add_argument('--num-images', type=int, default=20,
                       help='Number of images to test')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=1280,
                       help='Image size for inference')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--output-dir', type=str, default='../outputs/sanity_check',
                       help='Output directory for predictions')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 解析路径
    project_root = Path(__file__).resolve().parents[2]  # png2svg根目录
    train_root = Path(__file__).resolve().parents[1]   # train_symbols目录
    
    # 数据集路径
    if not Path(args.data).is_absolute():
        data_dir = train_root / args.data
    else:
        data_dir = Path(args.data)
    
    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        return
    
    # 模型路径
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = train_root / model_path
    else:
        # 尝试多种可能的模型位置
        possible_models = [
            project_root / "png2svg" / "models" / "symbols_yolo.onnx",
            train_root / "runs" / "detect" / "symbols" / "weights" / "best.pt",
            train_root / "runs" / "detect" / "symbols" / "weights" / "best.onnx"
        ]
        
        model_path = None
        for p in possible_models:
            if p.exists():
                model_path = p
                break
        
        if model_path is None:
            logger.error("No model found. Please specify --model or ensure model exists in expected locations")
            logger.info("Expected locations:")
            for p in possible_models:
                logger.info(f"  {p}")
            return
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    logger.info(f"Using model: {model_path}")
    
    # 输出目录
    if not Path(args.output_dir).is_absolute():
        output_dir = train_root / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 图像目录
    img_dir = data_dir / "images" / args.split
    if not img_dir.exists():
        logger.error(f"Image directory not found: {img_dir}")
        return
    
    # 获取所有图像文件
    image_extensions = ['.png', '.jpg', '.jpeg']
    images = []
    for ext in image_extensions:
        images.extend(img_dir.glob(f'*{ext}'))
    
    if not images:
        logger.error(f"No images found in {img_dir}")
        return
    
    logger.info(f"Found {len(images)} images in {args.split} split")
    
    # 随机选择图像
    if len(images) > args.num_images:
        images = random.sample(images, args.num_images)
    
    # 加载模型
    logger.info("Loading model...")
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # 加载类别名称
    try:
        dataset_yaml = data_dir / "dataset.yaml"
        if dataset_yaml.exists():
            class_names = load_class_names(dataset_yaml)
            logger.info(f"Loaded {len(class_names)} class names: {class_names}")
        else:
            logger.warning("dataset.yaml not found, using default class names")
            class_names = None
    except Exception as e:
        logger.warning(f"Failed to load class names: {e}")
        class_names = None
    
    # 推理参数
    predict_params = {
        'source': None,  # 每次设置
        'conf': args.conf,
        'iou': args.iou,
        'imgsz': args.imgsz,
        'device': args.device,
        'save': False,
        'show': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'line_width': 2
    }
    
    # 处理每张图像
    logger.info(f"Processing {len(images)} images...")
    
    success_count = 0
    for i, img_path in enumerate(images, 1):
        logger.info(f"Processing [{i}/{len(images)}]: {img_path.name}")
        
        try:
            # 执行推理
            predict_params['source'] = str(img_path)
            results = model.predict(**predict_params)
            
            if not results:
                logger.warning(f"No results for {img_path.name}")
                continue
            
            result = results[0]
            
            # 读取原图
            img = cv2.imread(str(img_path))
            if img is None:
                logger.error(f"Failed to read image: {img_path}")
                continue
            
            # 获取检测结果
            boxes = result.boxes
            detections = []
            
            if boxes is not None:
                for box in boxes:
                    # 获取坐标和置信度
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # 获取类别名称
                    if class_names and cls_id < len(class_names):
                        cls_name = class_names[cls_id]
                    else:
                        cls_name = f"class_{cls_id}"
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'conf': float(conf),
                        'class': cls_name,
                        'class_id': cls_id
                    })
            
            logger.info(f"  Found {len(detections)} detections")
            
            # 绘制检测结果
            img_vis = img.copy()
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0),
                (128, 255, 0), (0, 128, 255), (255, 0, 128), (128, 0, 255)
            ]
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['conf']
                cls_name = det['class']
                cls_id = det['class_id']
                
                # 选择颜色
                color = colors[cls_id % len(colors)]
                
                # 绘制边界框
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{cls_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                
                # 标签背景
                cv2.rectangle(img_vis, (x1, y1 - label_size[1] - 8), 
                             (x1 + label_size[0], y1), color, -1)
                
                # 标签文字
                cv2.putText(img_vis, label, (x1, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 保存结果
            output_path = output_dir / f"{img_path.stem}_pred.png"
            cv2.imwrite(str(output_path), img_vis)
            logger.info(f"  Saved to: {output_path}")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
    
    logger.info(f"\nSanity check completed!")
    logger.info(f"Successfully processed: {success_count}/{len(images)}")
    logger.info(f"Results saved to: {output_dir}")
    
    if success_count > 0:
        logger.info("\nNext step: Review the prediction images and run 'python scripts/04_stats.py' for detailed statistics")


if __name__ == "__main__":
    main()