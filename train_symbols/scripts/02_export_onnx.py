"""
ONNX导出脚本
将训练好的YOLO模型导出为ONNX格式并拷贝到png2svg/models目录
"""
import sys
import shutil
import logging
import argparse
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_best_model(runs_dir: Path, experiment_name: str = None) -> Path:
    """
    在runs目录中找到最新的best.pt模型
    
    Args:
        runs_dir: runs目录路径
        experiment_name: 实验名称（如果指定）
    
    Returns:
        best.pt文件路径
    """
    detect_dir = runs_dir / "detect"
    if not detect_dir.exists():
        raise FileNotFoundError(f"Detection runs directory not found: {detect_dir}")
    
    # 如果指定了实验名称，直接查找
    if experiment_name:
        exp_dir = detect_dir / experiment_name
        if exp_dir.exists():
            best_pt = exp_dir / "weights" / "best.pt"
            if best_pt.exists():
                return best_pt
            raise FileNotFoundError(f"best.pt not found in {exp_dir}")
    
    # 查找所有可能的权重文件
    weight_files = []
    for exp_dir in detect_dir.iterdir():
        if exp_dir.is_dir():
            weights_dir = exp_dir / "weights"
            if weights_dir.exists():
                best_pt = weights_dir / "best.pt"
                if best_pt.exists():
                    # 按修改时间排序
                    weight_files.append((best_pt.stat().st_mtime, best_pt))
    
    if not weight_files:
        raise FileNotFoundError(f"No best.pt found in {detect_dir}")
    
    # 返回最新的模型
    weight_files.sort(key=lambda x: x[0], reverse=True)
    latest_model = weight_files[0][1]
    logger.info(f"Found latest model: {latest_model}")
    return latest_model


def main():
    parser = argparse.ArgumentParser(description='Export YOLO model to ONNX format')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (.pt)')
    parser.add_argument('--runs', type=str, default='../runs',
                       help='Path to runs directory')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name (if not specified, use latest)')
    parser.add_argument('--output-name', type=str, default='symbols_yolo.onnx',
                       help='Output ONNX file name')
    parser.add_argument('--dynamic', action='store_true', default=True,
                       help='Export with dynamic batch size')
    parser.add_argument('--simplify', action='store_true', default=True,
                       help='Simplify ONNX model')
    parser.add_argument('--opset', type=int, default=12,
                       help='ONNX opset version')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640],
                       help='Image size for export (height width)')
    args = parser.parse_args()
    
    # 解析路径
    project_root = Path(__file__).resolve().parents[2]  # png2svg根目录
    train_root = Path(__file__).resolve().parents[1]   # train_symbols目录
    
    # 模型路径
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = train_root / model_path
    else:
        # 自动查找最新的best.pt
        runs_dir = train_root / args.runs if not Path(args.runs).is_absolute() else Path(args.runs)
        model_path = find_best_model(runs_dir, args.exp_name)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    logger.info(f"Loading model from: {model_path}")
    
    # 加载YOLO模型
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # 导出参数
    export_params = {
        'format': 'onnx',
        'dynamic': args.dynamic,
        'simplify': args.simplify,
        'opset': args.opset,
        'imgsz': args.imgsz,
        'half': False,  # 使用FP32以确保兼容性
        'int8': False,
        'nms': True,    # 包含NMS后处理
        'agnostic_nms': False,
        'device': 'cpu'  # 导出时使用CPU确保兼容性
    }
    
    # 执行导出
    logger.info("Starting ONNX export...")
    logger.info(f"Export parameters: {export_params}")
    
    try:
        export_path = model.export(**export_params)
        logger.info(f"Model exported to: {export_path}")
        
        # 检查导出的ONNX文件
        onnx_path = Path(export_path)
        if not onnx_path.exists():
            # 尝试在模型目录下查找
            model_dir = model_path.parent
            possible_onnx = model_dir / (model_path.stem + '.onnx')
            if possible_onnx.exists():
                onnx_path = possible_onnx
            else:
                logger.error("Exported ONNX file not found")
                return
        
        # 拷贝到png2svg/models目录
        target_dir = project_root / "png2svg" / "models"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = target_dir / args.output_name
        shutil.copyfile(onnx_path, target_path)
        
        logger.info(f"ONNX model copied to: {target_path}")
        
        # 验证文件大小
        file_size_mb = target_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model size: {file_size_mb:.1f} MB")
        
        # 显示模型信息
        logger.info("\nModel export completed successfully!")
        logger.info(f"  Source model: {model_path}")
        logger.info(f"  ONNX model: {target_path}")
        logger.info(f"  Dynamic batching: {args.dynamic}")
        logger.info(f"  ONNX opset: {args.opset}")
        logger.info(f"  Image size: {args.imgsz}")
        
        logger.info("\nNext step: Run 'python scripts/03_sanity_check.py' to verify the model")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


def verify_onnx_model(onnx_path: Path):
    """
    验证ONNX模型的基本信息
    """
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        logger.info(f"ONNX model verification:")
        logger.info(f"  Inputs: {[input.name for input in model.graph.input]}")
        logger.info(f"  Outputs: {[output.name for output in model.graph.output]}")
        logger.info(f"  Nodes: {len(model.graph.node)}")
    except ImportError:
        logger.warning("onnx package not available for verification")
    except Exception as e:
        logger.warning(f"ONNX verification failed: {e}")


if __name__ == "__main__":
    main()