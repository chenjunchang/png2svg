"""
统计分析脚本
分析数据集类别分布和长尾问题
"""
import sys
import yaml
import logging
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_yolo_label(label_path: Path) -> list:
    """
    解析YOLO格式的标签文件
    
    Args:
        label_path: 标签文件路径
    
    Returns:
        类别ID列表
    """
    if not label_path.exists():
        return []
    
    class_ids = []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:  # class_id cx cy w h
                        class_ids.append(int(parts[0]))
    except Exception as e:
        logger.warning(f"Error parsing {label_path}: {e}")
    
    return class_ids


def load_class_names(config_path: Path) -> list:
    """加载类别名称"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('names', [])


def analyze_dataset_distribution(data_dir: Path, class_names: list) -> dict:
    """
    分析数据集类别分布
    
    Args:
        data_dir: 数据集目录
        class_names: 类别名称列表
    
    Returns:
        分析结果字典
    """
    results = {
        'splits': {},
        'total': Counter(),
        'images_per_split': {},
        'annotations_per_split': {}
    }
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        label_dir = data_dir / 'labels' / split
        if not label_dir.exists():
            logger.warning(f"Label directory not found: {label_dir}")
            continue
        
        split_counter = Counter()
        label_files = list(label_dir.glob('*.txt'))
        
        images_with_labels = 0
        total_annotations = 0
        
        for label_file in label_files:
            class_ids = parse_yolo_label(label_file)
            if class_ids:
                images_with_labels += 1
                total_annotations += len(class_ids)
                for cls_id in class_ids:
                    if cls_id < len(class_names):
                        class_name = class_names[cls_id]
                        split_counter[class_name] += 1
                        results['total'][class_name] += 1
        
        results['splits'][split] = split_counter
        results['images_per_split'][split] = images_with_labels
        results['annotations_per_split'][split] = total_annotations
        
        logger.info(f"{split.upper()} split: {images_with_labels} images, {total_annotations} annotations")
    
    return results


def identify_long_tail_classes(class_counts: Counter, threshold_percentile: float = 10) -> dict:
    """
    识别长尾类别
    
    Args:
        class_counts: 类别计数
        threshold_percentile: 长尾阈值百分位数
    
    Returns:
        长尾分析结果
    """
    if not class_counts:
        return {'head': [], 'tail': [], 'threshold': 0}
    
    counts = list(class_counts.values())
    threshold = np.percentile(counts, threshold_percentile)
    
    head_classes = []
    tail_classes = []
    
    for class_name, count in class_counts.items():
        if count > threshold:
            head_classes.append((class_name, count))
        else:
            tail_classes.append((class_name, count))
    
    # 按数量排序
    head_classes.sort(key=lambda x: x[1], reverse=True)
    tail_classes.sort(key=lambda x: x[1], reverse=True)
    
    return {
        'head': head_classes,
        'tail': tail_classes,
        'threshold': threshold
    }


def plot_class_distribution(class_counts: Counter, class_names: list, output_dir: Path):
    """
    绘制类别分布图
    
    Args:
        class_counts: 类别计数
        class_names: 类别名称列表
        output_dir: 输出目录
    """
    if not class_counts:
        logger.warning("No class counts to plot")
        return
    
    # 确保所有类别都在统计中
    all_counts = []
    all_names = []
    for name in class_names:
        all_counts.append(class_counts.get(name, 0))
        all_names.append(name)
    
    # 创建条形图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(all_names)), all_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Annotations')
    plt.title('Class Distribution in Dataset')
    plt.xticks(range(len(all_names)), all_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # 在每个条上添加数值
    for i, (bar, count) in enumerate(zip(bars, all_counts)):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(all_counts) * 0.01,
                    str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = output_dir / 'class_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Class distribution plot saved to: {plot_path}")


def plot_split_comparison(split_results: dict, class_names: list, output_dir: Path):
    """
    绘制不同split的类别分布对比图
    """
    if not split_results:
        return
    
    splits = list(split_results.keys())
    x = np.arange(len(class_names))
    width = 0.8 / len(splits)
    
    plt.figure(figsize=(14, 8))
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, split in enumerate(splits):
        split_counts = [split_results[split].get(name, 0) for name in class_names]
        plt.bar(x + i * width, split_counts, width, 
                label=split.upper(), color=colors[i % len(colors)], alpha=0.7)
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Annotations')
    plt.title('Class Distribution Comparison Across Splits')
    plt.xticks(x + width * (len(splits) - 1) / 2, class_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plot_path = output_dir / 'split_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Split comparison plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset statistics')
    parser.add_argument('--data', type=str, default='../datasets/pg_symbols',
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='../outputs/stats',
                       help='Output directory for plots and reports')
    parser.add_argument('--tail-threshold', type=float, default=10,
                       help='Percentile threshold for long-tail classes')
    args = parser.parse_args()
    
    # 解析路径
    train_root = Path(__file__).resolve().parents[1]
    
    if not Path(args.data).is_absolute():
        data_dir = train_root / args.data
    else:
        data_dir = Path(args.data)
    
    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        return
    
    if not Path(args.output).is_absolute():
        output_dir = train_root / args.output
    else:
        output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载类别名称
    dataset_yaml = data_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        logger.error(f"Dataset YAML not found: {dataset_yaml}")
        return
    
    class_names = load_class_names(dataset_yaml)
    logger.info(f"Loaded {len(class_names)} classes: {class_names}")
    
    # 分析数据集分布
    logger.info("Analyzing dataset distribution...")
    results = analyze_dataset_distribution(data_dir, class_names)
    
    # 显示统计结果
    logger.info("\n" + "="*60)
    logger.info("DATASET STATISTICS")
    logger.info("="*60)
    
    # 总体统计
    total_images = sum(results['images_per_split'].values())
    total_annotations = sum(results['annotations_per_split'].values())
    
    logger.info(f"Total Images: {total_images}")
    logger.info(f"Total Annotations: {total_annotations}")
    logger.info(f"Average Annotations per Image: {total_annotations/total_images:.2f}")
    
    # 各split统计
    logger.info("\nSplit Statistics:")
    for split in ['train', 'val', 'test']:
        if split in results['images_per_split']:
            images = results['images_per_split'][split]
            annotations = results['annotations_per_split'][split]
            percentage = images / total_images * 100 if total_images > 0 else 0
            logger.info(f"  {split.upper():5s}: {images:5d} images ({percentage:5.1f}%), "
                       f"{annotations:6d} annotations")
    
    # 类别分布
    logger.info("\nClass Distribution (Total):")
    sorted_classes = results['total'].most_common()
    for class_name, count in sorted_classes:
        percentage = count / total_annotations * 100 if total_annotations > 0 else 0
        logger.info(f"  {class_name:20s}: {count:6d} ({percentage:5.1f}%)")
    
    # 空类别
    empty_classes = [name for name in class_names if results['total'].get(name, 0) == 0]
    if empty_classes:
        logger.warning(f"\nEmpty Classes: {empty_classes}")
    
    # 长尾分析
    long_tail = identify_long_tail_classes(results['total'], args.tail_threshold)
    
    logger.info(f"\n长尾分析 (Threshold: {long_tail['threshold']:.1f}):")
    logger.info(f"Head Classes ({len(long_tail['head'])}): {[x[0] for x in long_tail['head']]}")
    logger.info(f"Tail Classes ({len(long_tail['tail'])}): {[x[0] for x in long_tail['tail']]}")
    
    if long_tail['tail']:
        logger.warning("\n⚠️  Long-tail classes detected!")
        logger.info("Recommendations:")
        logger.info("  1. Consider data augmentation for tail classes")
        logger.info("  2. Use class weights during training")
        logger.info("  3. Apply focal loss for imbalanced datasets")
        logger.info("  4. Collect more data for underrepresented classes")
    
    # 生成图表
    logger.info("\nGenerating plots...")
    plot_class_distribution(results['total'], class_names, output_dir)
    plot_split_comparison(results['splits'], class_names, output_dir)
    
    # 生成报告文件
    report_path = output_dir / 'statistics_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Dataset Statistics Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Total Annotations: {total_annotations}\n")
        f.write(f"Average Annotations per Image: {total_annotations/total_images:.2f}\n\n")
        
        f.write("Split Statistics:\n")
        for split in ['train', 'val', 'test']:
            if split in results['images_per_split']:
                images = results['images_per_split'][split]
                annotations = results['annotations_per_split'][split]
                percentage = images / total_images * 100 if total_images > 0 else 0
                f.write(f"  {split.upper()}: {images} images ({percentage:.1f}%), "
                       f"{annotations} annotations\n")
        
        f.write("\nClass Distribution:\n")
        for class_name, count in sorted_classes:
            percentage = count / total_annotations * 100 if total_annotations > 0 else 0
            f.write(f"  {class_name}: {count} ({percentage:.1f}%)\n")
        
        if empty_classes:
            f.write(f"\nEmpty Classes: {empty_classes}\n")
        
        f.write(f"\nLong-tail Analysis (Threshold: {long_tail['threshold']:.1f}):\n")
        f.write(f"Head Classes: {[x[0] for x in long_tail['head']]}\n")
        f.write(f"Tail Classes: {[x[0] for x in long_tail['tail']]}\n")
    
    logger.info(f"Report saved to: {report_path}")
    logger.info(f"Plots saved to: {output_dir}")
    logger.info("\nStatistical analysis completed!")


if __name__ == "__main__":
    main()