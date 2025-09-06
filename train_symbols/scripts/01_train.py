"""
YOLO训练脚本
使用Ultralytics YOLO训练小目标检测模型
"""
import sys
import argparse
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO

# 添加父目录到Python路径
sys.path.append(str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for symbol detection')
    parser.add_argument('--data', type=str, 
                       default='../datasets/pg_symbols/dataset.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, 
                       default='yolo11n.pt',
                       help='Initial model (yolo11n.pt, yolo11s.pt, etc.)')
    parser.add_argument('--cfg', type=str,
                       default='../configs/train.full.yaml',
                       help='Training configuration file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    parser.add_argument('--device', type=str, default='0',
                       help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--project', type=str, default='../runs/detect',
                       help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='symbols',
                       help='Experiment name')
    args = parser.parse_args()
    
    # 解析路径
    project_root = Path(__file__).resolve().parents[2]
    train_root = Path(__file__).resolve().parents[1]
    
    # 数据集路径
    if not Path(args.data).is_absolute():
        data_path = (train_root / args.data).resolve()
    else:
        data_path = Path(args.data)
    
    if not data_path.exists():
        logger.error(f"Dataset YAML not found: {data_path}")
        logger.info("Please run 'python scripts/00_build_dataset.py' first to build the dataset")
        return
    
    # 配置文件路径
    if not Path(args.cfg).is_absolute():
        cfg_path = (train_root / args.cfg).resolve()
    else:
        cfg_path = Path(args.cfg)
    
    if not cfg_path.exists():
        logger.error(f"Configuration file not found: {cfg_path}")
        return
    
    # 加载训练配置
    with open(cfg_path, 'r', encoding='utf-8') as f:
        train_cfg = yaml.safe_load(f)
    
    logger.info(f"Training configuration:")
    for key, value in train_cfg.items():
        logger.info(f"  {key}: {value}")
    
    # 初始化YOLO模型
    logger.info(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # 项目输出目录
    if not Path(args.project).is_absolute():
        project_dir = (train_root / args.project).resolve()
    else:
        project_dir = Path(args.project)
    
    # 训练参数
    train_params = {
        'data': str(data_path),
        'imgsz': train_cfg.get('imgsz', 1024),
        'epochs': train_cfg.get('epochs', 150),
        'batch': train_cfg.get('batch', 16),
        'optimizer': train_cfg.get('optimizer', 'AdamW'),
        'lr0': train_cfg.get('lr0', 1e-3),
        'lrf': train_cfg.get('lrf', 0.01),
        'momentum': train_cfg.get('momentum', 0.937),
        'weight_decay': train_cfg.get('weight_decay', 0.0005),
        'cos_lr': train_cfg.get('cos_lr', True),
        'mosaic': train_cfg.get('mosaic', 0.7),
        'mixup': train_cfg.get('mixup', 0.0),
        'hsv_h': train_cfg.get('hsv_h', 0.0),  # 关闭颜色增强（几何符号不需要）
        'hsv_s': train_cfg.get('hsv_s', 0.0),
        'hsv_v': train_cfg.get('hsv_v', 0.0),
        'degrees': train_cfg.get('degrees', 0.0),  # 关闭旋转（几何符号方向重要）
        'translate': train_cfg.get('translate', 0.05),
        'scale': train_cfg.get('scale', 0.9),
        'fliplr': train_cfg.get('fliplr', 0.5),
        'flipud': train_cfg.get('flipud', 0.0),
        'perspective': train_cfg.get('perspective', 0.0),
        'plots': True,
        'save': True,
        'save_period': 10,
        'patience': train_cfg.get('patience', 25),
        'device': args.device,
        'project': str(project_dir),
        'name': args.name,
        'exist_ok': args.resume,
        'resume': args.resume,
        'verbose': True,
        'seed': 1234,
        'close_mosaic': 10,  # 最后10轮关闭mosaic
        'amp': True,  # 混合精度训练
        'fraction': 1.0,  # 使用全部数据
        'profile': False,
        'freeze': None,  # 不冻结任何层
        'multi_scale': False,  # 小目标建议关闭多尺度
        'single_cls': False,  # 多类检测
        'rect': False,  # 不使用矩形训练
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save_json': False,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'vid_stride': 1,
        'stream_buffer': False,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'retina_masks': False,
        'show': False,
        'save_frames': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'line_width': None
    }
    
    # 开始训练
    logger.info("Starting training...")
    logger.info(f"Results will be saved to: {project_dir / args.name}")
    
    try:
        results = model.train(**train_params)
        logger.info("Training completed successfully!")
        
        # 显示最佳结果
        if results:
            logger.info("\nBest results:")
            logger.info(f"  Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            logger.info(f"  Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        # 保存路径
        best_model_path = project_dir / args.name / 'weights' / 'best.pt'
        if best_model_path.exists():
            logger.info(f"\nBest model saved at: {best_model_path}")
            logger.info("Next step: Run 'python scripts/02_export_onnx.py' to export ONNX model")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()