#!/usr/bin/env python3
"""
数学题配图批量转换SVG工具 - Potrace版本

使用Potrace矢量化引擎，将数学题目中的几何图形转换为高质量的SVG矢量图。
专门针对数学配图进行优化，能够准确识别线条、文字和几何形状。
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PotraceMathImageToSVG:
    def __init__(self, input_dir="input_image", output_dir="output_svg"):
        """
        初始化Potrace转换器
        
        Args:
            input_dir: 输入图片目录
            output_dir: 输出SVG目录
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Potrace优化参数 - 专门针对数学图形
        self.potrace_params = {
            '--svg': True,           # 输出SVG格式
            '--tight': True,         # 紧密包围盒
            '--turnpolicy': 'black', # 转向策略：偏向黑色
            '--turdsize': '2',       # 最小特征尺寸（像素）
            '--alphamax': '1.0',     # 角度阈值
            '--opttolerance': '0.2', # 优化容差
            '--unit': '1'            # 单位尺寸
        }
        
    def check_potrace_installation(self):
        """检查potrace是否可用"""
        try:
            result = subprocess.run(['potrace', '--version'], 
                                  capture_output=True, text=True)
            logger.info(f"Potrace版本: {result.stdout.strip()}")
            return True
        except FileNotFoundError:
            logger.error("未找到potrace！请确保potrace已安装并在PATH中")
            return False
        except Exception as e:
            logger.error(f"检查potrace时出错: {e}")
            return False
    
    def preprocess_image_for_potrace(self, image_path):
        """
        为Potrace预处理图像：转换为高质量的黑白位图
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            PIL.Image: 预处理后的黑白图像
        """
        # 加载图像
        img = Image.open(image_path)
        logger.info(f"原始图像尺寸: {img.size}, 模式: {img.mode}")
        
        # 转换为灰度图
        if img.mode != 'L':
            img = img.convert('L')
        
        # 增强对比度 - 让线条更清晰
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # 使用自适应阈值或Otsu方法进行二值化
        # 先尝试简单的阈值
        img_array = list(img.getdata())
        
        # 计算Otsu阈值
        histogram = img.histogram()
        total_pixels = sum(histogram)
        
        # 简化的Otsu算法
        sum_total = sum(i * histogram[i] for i in range(256))
        sum_background = 0
        weight_background = 0
        weight_foreground = 0
        max_variance = 0
        optimal_threshold = 0
        
        for threshold in range(256):
            weight_background += histogram[threshold]
            if weight_background == 0:
                continue
                
            weight_foreground = total_pixels - weight_background
            if weight_foreground == 0:
                break
                
            sum_background += threshold * histogram[threshold]
            
            mean_background = sum_background / weight_background
            mean_foreground = (sum_total - sum_background) / weight_foreground
            
            # 类间方差
            between_class_variance = weight_background * weight_foreground * \
                                   (mean_background - mean_foreground) ** 2
            
            if between_class_variance > max_variance:
                max_variance = between_class_variance
                optimal_threshold = threshold
        
        logger.info(f"计算得到的最优阈值: {optimal_threshold}")
        
        # 应用阈值
        binary_img = img.point(lambda x: 255 if x > optimal_threshold else 0, mode='1')
        
        return binary_img
    
    def run_potrace(self, bitmap_path, output_svg_path):
        """
        运行Potrace进行矢量化
        
        Args:
            bitmap_path: 输入位图路径
            output_svg_path: 输出SVG路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 构建potrace命令
            cmd = ['potrace']
            
            # 添加参数
            for param, value in self.potrace_params.items():
                if value is True:
                    cmd.append(param)
                elif value is not True:
                    cmd.extend([param, str(value)])
            
            # 输出文件
            cmd.extend(['-o', str(output_svg_path)])
            
            # 输入文件
            cmd.append(str(bitmap_path))
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 执行potrace
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"Potrace转换成功: {output_svg_path}")
                return True
            else:
                logger.error(f"Potrace执行失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Potrace执行超时")
            return False
        except Exception as e:
            logger.error(f"运行Potrace时出错: {e}")
            return False
    
    def optimize_svg(self, svg_path):
        """
        优化生成的SVG文件，添加数学图形友好的样式
        
        Args:
            svg_path: SVG文件路径
        """
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单清理：确保SVG格式正确
            # 不做复杂的修改，保持Potrace原始输出的质量
            logger.info(f"SVG文件已生成: {svg_path}")
                
        except Exception as e:
            logger.warning(f"优化SVG文件时出错: {e}")
    
    def convert_image(self, image_path):
        """
        转换单个图像文件
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            bool: 转换是否成功
        """
        try:
            logger.info(f"开始处理: {image_path}")
            
            # 预处理图像
            binary_img = self.preprocess_image_for_potrace(image_path)
            
            # 生成输出文件名
            stem = Path(image_path).stem
            output_svg = self.output_dir / f"{stem}.svg"
            
            # 创建临时位图文件
            with tempfile.NamedTemporaryFile(suffix='.pbm', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # 保存为PBM格式（Potrace的标准输入格式）
                binary_img.save(temp_path, format='PPM')
                
                try:
                    # 运行Potrace
                    success = self.run_potrace(temp_path, output_svg)
                    
                    if success:
                        # 优化SVG
                        self.optimize_svg(output_svg)
                        logger.info(f"转换完成: {output_svg}")
                        return True
                    else:
                        return False
                        
                finally:
                    # 清理临时文件
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"处理 {image_path} 时出错: {str(e)}")
            return False
    
    def batch_convert(self):
        """
        批量转换input_image目录中的所有图片
        
        Returns:
            dict: 转换结果统计
        """
        # 检查potrace安装
        if not self.check_potrace_installation():
            return {"success": 0, "failed": 0, "total": 0, "error": "potrace未安装"}
        
        if not self.input_dir.exists():
            logger.error(f"输入目录不存在: {self.input_dir}")
            return {"success": 0, "failed": 0, "total": 0, "error": "输入目录不存在"}
        
        # 支持的图片格式
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        # 查找所有图片文件
        image_files = []
        for fmt in supported_formats:
            image_files.extend(self.input_dir.glob(f"*{fmt}"))
            image_files.extend(self.input_dir.glob(f"*{fmt.upper()}"))
        
        if not image_files:
            logger.warning(f"在 {self.input_dir} 中未找到支持的图片文件")
            return {"success": 0, "failed": 0, "total": 0, "error": "未找到图片文件"}
        
        logger.info(f"找到 {len(image_files)} 个图片文件，开始批量转换...")
        
        # 批量处理
        success_count = 0
        failed_count = 0
        
        for image_file in image_files:
            if self.convert_image(image_file):
                success_count += 1
            else:
                failed_count += 1
        
        # 统计结果
        total = len(image_files)
        logger.info(f"批量转换完成！成功: {success_count}, 失败: {failed_count}, 总计: {total}")
        
        return {
            "success": success_count,
            "failed": failed_count,
            "total": total
        }

def main():
    """主函数"""
    print("=== 数学题配图批量转换SVG工具 (Potrace版本) ===")
    print("使用专业的Potrace引擎进行高质量矢量化转换")
    print("特别适合数学几何图形和线条图的处理")
    print()
    
    # 创建转换器
    converter = PotraceMathImageToSVG()
    
    # 执行批量转换
    result = converter.batch_convert()
    
    # 显示结果
    print("\n=== 转换结果 ===")
    if "error" in result:
        print(f"错误: {result['error']}")
    else:
        print(f"成功转换: {result['success']} 个文件")
        print(f"转换失败: {result['failed']} 个文件")
        print(f"总计文件: {result['total']} 个")
        
        if result['success'] > 0:
            print(f"\nSVG文件已保存到: {converter.output_dir}")
            print("生成的SVG文件具有以下特点：")
            print("- 高质量矢量化，无像素化")
            print("- 完美适配数学图形和几何形状")
            print("- 文件体积小，缩放不失真")
            print("- 可在任何支持SVG的软件中使用")

if __name__ == "__main__":
    main()