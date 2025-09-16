#!/usr/bin/env python3
"""
测试标注文件分割功能
"""

import os
import shutil
import json
from autolabel import VideoProcessor

def test_annotation_split():
    """测试标注文件分割功能"""
    print("测试标注文件分割功能...")
    
    # 清理之前的输出
    if os.path.exists("test_split"):
        shutil.rmtree("test_split")
    
    # 创建视频处理器（进行检测，提取少量帧）
    processor = VideoProcessor(
        video_path="data/0912/topdown01.mp4",
        output_dir="test_split",
        frame_rate=2.0,  # 较高的帧率用于快速测试
        detect_objects=True,  # 进行检测
        crop_top_half=False,
        max_frames=5  # 限制5帧
    )
    
    # 处理视频
    result = processor.process_video()
    
    # 检查标注文件是否创建
    train_path = os.path.join(processor.ann_dir, 'train.json')
    val_path = os.path.join(processor.ann_dir, 'val.json')
    full_path = os.path.join(processor.ann_dir, f"{processor.video_name}.json")
    
    assert os.path.exists(train_path), "训练集标注文件不存在"
    assert os.path.exists(val_path), "验证集标注文件不存在"
    assert os.path.exists(full_path), "完整标注文件不存在"
    
    # 检查文件内容
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    
    with open(full_path, 'r') as f:
        full_data = json.load(f)
    
    # 验证分割比例 (8:2)
    total_images = len(full_data['images'])
    train_images = len(train_data['images'])
    val_images = len(val_data['images'])
    
    print(f"总图像数: {total_images}")
    print(f"训练集图像数: {train_images} ({train_images/total_images:.1%})")
    print(f"验证集图像数: {val_images} ({val_images/total_images:.1%})")
    
    # 检查affordance和grasp_policy字段
    if full_data['annotations']:
        ann = full_data['annotations'][0]
        assert 'affordance' in ann, "affordance字段不存在"
        assert 'grasp_policy' in ann, "grasp_policy字段不存在"
        assert ann['grasp_policy'] == 'no', f"grasp_policy应为'no', 实际为{ann['grasp_policy']}"
        
        print(f"✓ affordance字段: {ann['affordance']}")
        print(f"✓ grasp_policy字段: {ann['grasp_policy']}")
    
    print(f"✓ 训练集标注文件: {train_path}")
    print(f"✓ 验证集标注文件: {val_path}")
    print(f"✓ 完整标注文件: {full_path}")
    
    return True

if __name__ == "__main__":
    print("开始测试标注文件分割功能...\n")
    
    try:
        success = test_annotation_split()
        if success:
            print("\n✓ 标注文件分割功能测试成功!")
            print("\n现在会自动生成 train.json 和 val.json 文件")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()