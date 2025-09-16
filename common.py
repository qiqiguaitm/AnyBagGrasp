"""
Common utility functions for video processing, visualization, and geometric calculations.
"""

import cv2
import numpy as np
import math
import os
import json
from pycocotools import mask as coco_mask


class VisualizationUtils:
    """Utility functions for visualization"""
    
    @staticmethod
    def get_category_colors(category):
        """Get color scheme for different categories"""
        if 'handle' in category.lower():
            # Handle: cyan color scheme
            mask_color = (255, 200, 0)  # cyan
            edge_color = (255, 255, 0)  # bright cyan
            bbox_color = (255, 255, 0)  # yellow border
        else:
            # Bag: magenta color scheme
            mask_color = (255, 0, 200)  # magenta
            edge_color = (255, 0, 255)  # bright magenta
            bbox_color = (255, 100, 255)  # pink border
        return mask_color, edge_color, bbox_color
    
    @staticmethod
    def draw_mask(vis_img, mask_rle, mask_color, edge_color, alpha=0.3):
        """Draw segmentation mask with edge emphasis"""
        try:
            mask = coco_mask.decode(mask_rle)
            
            # 1. Draw semi-transparent mask
            overlay = vis_img.copy()
            overlay[mask > 0] = mask_color
            vis_img = cv2.addWeighted(vis_img, 1-alpha, overlay, alpha, 0)
            
            # 2. Draw mask edge contours
            contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours, -1, edge_color, 2)
            
            return vis_img, mask
        except Exception as e:
            print(f"Error drawing mask: {e}")
            return vis_img, None
    
    @staticmethod
    def draw_bbox_with_corners(vis_img, bbox, bbox_color, corner_len=20, thickness=2):
        """Draw bounding box with corner markers"""
        if len(bbox) < 4:
            return vis_img
            
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Draw solid border
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), bbox_color, thickness)
        
        # Draw corner markers
        corner_thickness = 3
        # Top-left corner
        cv2.line(vis_img, (x1, y1), (x1+corner_len, y1), bbox_color, corner_thickness)
        cv2.line(vis_img, (x1, y1), (x1, y1+corner_len), bbox_color, corner_thickness)
        # Top-right corner
        cv2.line(vis_img, (x2-corner_len, y1), (x2, y1), bbox_color, corner_thickness)
        cv2.line(vis_img, (x2, y1), (x2, y1+corner_len), bbox_color, corner_thickness)
        # Bottom-left corner
        cv2.line(vis_img, (x1, y2-corner_len), (x1, y2), bbox_color, corner_thickness)
        cv2.line(vis_img, (x1, y2), (x1+corner_len, y2), bbox_color, corner_thickness)
        # Bottom-right corner
        cv2.line(vis_img, (x2, y2-corner_len), (x2, y2), bbox_color, corner_thickness)
        cv2.line(vis_img, (x2-corner_len, y2), (x2, y2), bbox_color, corner_thickness)
        
        return vis_img
    
    @staticmethod
    def draw_label(vis_img, bbox, category, score, font_scale=0.7, thickness=2):
        """Draw label with background"""
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # Calculate area
        bbox_area = (x2 - x1) * (y2 - y1)
        label = f"{category} {score:.2f} A:{bbox_area:.0f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Label position (above or below box)
        label_y = y1 - 10 if y1 > 30 else y2 + 25
        
        # Draw label background (semi-transparent black)
        overlay = vis_img.copy()
        cv2.rectangle(overlay, 
                    (x1, label_y - label_size[1] - 8),
                    (x1 + label_size[0] + 10, label_y + 3),
                    (0, 0, 0), -1)
        vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)
        
        # Draw label text
        cv2.putText(vis_img, label, (x1 + 5, label_y - 2),
                   font, font_scale, (255, 255, 255), thickness)
        
        return vis_img
    
    @staticmethod
    def draw_rotated_rect(vis_img, affordance, color=(0, 255, 255), thickness=2):
        """Draw rotated rectangle"""
        xc, yc, w, h, angle = affordance
        rect = ((xc, yc), (w, h), math.degrees(angle))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(vis_img, [box], 0, color, thickness)
        return vis_img
    
    @staticmethod
    def draw_circle(vis_img, center, radius, color=(255, 0, 0), thickness=2):
        """Draw circle"""
        cv2.circle(vis_img, tuple(map(int, center)), int(radius), color, thickness)
        return vis_img
    
    @staticmethod
    def create_visualization(frame, result, vis_dir, base_filename, coco_data=None, image_id=None):
        """创建检测和分割的可视化结果，包括affordance和grasp_policy
        
        Args:
            frame: 原始图像
            result: 检测结果
            vis_dir: 可视化保存目录
            base_filename: 基础文件名
            coco_data: COCO数据(可选)
            image_id: 图像ID(可选)
            
        Returns:
            vis_img: 可视化后的图像
        """
        try:
            # 复制原始图像
            vis_img = frame.copy()
            
            if result and 'objects' in result:
                objects = result.get('objects', [])
                
                for obj in objects:
                    bbox = obj.get('bbox', [])
                    mask_rle = obj.get('mask', None)
                    category = obj.get('category', 'unknown').lower()
                    score = obj.get('score', 0.0)
                    
                    # 获取配色方案
                    mask_color, edge_color, bbox_color = VisualizationUtils.get_category_colors(category)
                    
                    # 绘制分割掩码（带边缘强调）
                    if mask_rle:
                        vis_img, _ = VisualizationUtils.draw_mask(vis_img, mask_rle, mask_color, edge_color)
                    
                    # 绘制边界框和标签
                    if len(bbox) >= 4:
                        # 绘制带角标记的边界框
                        vis_img = VisualizationUtils.draw_bbox_with_corners(vis_img, bbox, bbox_color)
                        
                        # 绘制标签
                        vis_img = VisualizationUtils.draw_label(vis_img, bbox, category, score)
            
            # 如果提供了image_id和coco_data，查找对应的标注并绘制affordance
            if image_id is not None and coco_data is not None:
                vis_img = VisualizationUtils.draw_affordance_and_policy(vis_img, image_id, coco_data)
            
            # 保存可视化结果
            vis_path = os.path.join(vis_dir, f"vis_{base_filename}")
            cv2.imwrite(vis_path, vis_img)
            
            return vis_img
            
        except Exception as e:
            print(f"创建可视化时发生错误: {e}")
            return frame
    
    @staticmethod
    def draw_affordance_and_policy(vis_img, image_id, coco_data):
        """在图像上绘制affordance、grasp_policy和IoU信息，根据affordance类型进行可视化
        
        Args:
            vis_img: 要绘制的图像
            image_id: 图像ID
            coco_data: COCO数据字典
            
        Returns:
            vis_img: 绘制后的图像
        """
        try:
            import math
            from pycocotools import mask as coco_mask
            
            # 查找该图像的所有标注
            image_annotations = [ann for ann in coco_data['annotations'] 
                               if ann['image_id'] == image_id]
            
            for ann in image_annotations:
                affordances = ann.get('affordance', [])  # 现在是list of lists
                grasp_policy = ann.get('grasp_policy', 'no')
                segmentation = ann.get('segmentation', None)
                handle_hole = ann.get('handle_hole', None)  # 获取handle_hole信息
                
                # 为多个affordance使用不同颜色
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                text_thickness = 4  # 加粗文字
                
                # 遍历所有affordances
                for aff_idx, affordance in enumerate(affordances):
                    if not affordance:  # 跳过空affordance
                        continue
                        
                    # 为每个affordance分配颜色
                    color = colors[aff_idx % len(colors)]
                    
                    # 计算affordance形状与mask的IoU
                    iou_value = 0.0
                    if segmentation and isinstance(segmentation, dict) and 'counts' in segmentation:
                        try:
                            mask = coco_mask.decode(segmentation)
                            # 确保mask是连续的numpy数组
                            mask = np.ascontiguousarray(mask, dtype=np.uint8)
                            
                            if len(affordance) == 3:  # circle类型
                                x, y, radius = affordance
                                # 创建圆形mask
                                affordance_mask = np.zeros(mask.shape, dtype=np.uint8)
                                cv2.circle(affordance_mask, (int(x), int(y)), int(radius), 1, -1)
                                # 计算IoU
                                iou_value = GeometryUtils.calculate_iou(mask, affordance_mask)
                                
                            elif len(affordance) >= 5:  # rot_bbox类型
                                xc, yc, w, h, angle = affordance
                                # 创建旋转矩形mask
                                affordance_mask = np.zeros(mask.shape, dtype=np.uint8)
                                rect = ((float(xc), float(yc)), (float(w), float(h)), float(angle)*180/math.pi)
                                box_points = cv2.boxPoints(rect).astype(int)
                                cv2.drawContours(affordance_mask, [box_points], 0, 1, -1)
                                # 计算IoU
                                iou_value = GeometryUtils.calculate_iou(mask, affordance_mask)
                        except Exception as e:
                            print(f"计算IoU时发生错误: {e}")
                    
                    # 根据affordance类型进行可视化
                    if len(affordance) == 3:  # circle类型 [x, y, radius]
                        x, y, radius = affordance
                        
                        # 绘制圆形（使用分配的颜色，加粗）
                        cv2.circle(vis_img, (int(x), int(y)), int(radius), color, 5)
                        
                        # 绘制中心点（使用分配的颜色）
                        cv2.circle(vis_img, (int(x), int(y)), 5, color, -1)
                        
                        # 显示affordance编号和参数（使用分配的颜色，加粗）
                        aff_text = f"Aff{aff_idx+1}: Circle({x:.1f},{y:.1f},r={radius:.1f})"
                        text_y = int(y) + 35 * aff_idx  # 垂直偏移避免重叠
                        cv2.putText(vis_img, aff_text, (int(x) + 10, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, text_thickness)
                        
                        # 显示IoU值（使用分配的颜色，加粗）
                        iou_text = f"IoU: {iou_value:.3f}"
                        cv2.putText(vis_img, iou_text, (int(x) + 10, text_y + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, text_thickness)
                        
                    elif len(affordance) >= 5:  # rot_bbox类型 [xc, yc, w, h, angle]
                        xc, yc, w, h, angle = affordance
                        
                        # 创建旋转矩形
                        rect = ((float(xc), float(yc)), (float(w), float(h)), float(angle)*180/math.pi)
                        box_points = cv2.boxPoints(rect).astype(int)
                        
                        # 绘制旋转矩形（使用分配的颜色，加粗）
                        cv2.drawContours(vis_img, [box_points], 0, color, 5)
                        
                        # 绘制中心点（使用分配的颜色）
                        cv2.circle(vis_img, (int(xc), int(yc)), 5, color, -1)
                        
                        # 显示affordance编号和参数（使用分配的颜色，加粗）
                        aff_text = f"Aff{aff_idx+1}: RotBox({xc:.1f},{yc:.1f},{w:.1f}x{h:.1f},θ={angle:.2f})"
                        text_y = int(yc) + 35 * aff_idx  # 垂直偏移避免重叠
                        cv2.putText(vis_img, aff_text, (int(xc) + 10, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, text_thickness)
                        
                        # 显示IoU值（使用分配的颜色，加粗）
                        iou_text = f"IoU: {iou_value:.3f}"
                        cv2.putText(vis_img, iou_text, (int(xc) + 10, text_y + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, text_thickness)
                
                # 绘制handle_hole圆形（如果存在）
                if handle_hole and len(handle_hole) == 3:
                    hh_x, hh_y, hh_radius = handle_hole
                    
                    # 使用橙色绘制handle_hole
                    hh_color = (0, 165, 255)  # 橙色 (BGR格式)
                    
                    # 绘制圆形轮廓
                    cv2.circle(vis_img, (int(hh_x), int(hh_y)), int(hh_radius), hh_color, 3)
                    
                    # 绘制中心点
                    cv2.circle(vis_img, (int(hh_x), int(hh_y)), 5, hh_color, -1)
                    
                    # 绘制标签
                    hh_text = f"Handle_Hole: ({hh_x:.1f},{hh_y:.1f},r={hh_radius:.1f})"
                    cv2.putText(vis_img, hh_text, (int(hh_x) + 15, int(hh_y) - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, hh_color, text_thickness)
                
                # 绘制rot_bbox_bag（如果存在）
                rot_bbox_bag = ann.get('rot_bbox_bag', None)
                if rot_bbox_bag and len(rot_bbox_bag) == 5:
                    bag_cx, bag_cy, bag_w, bag_h, bag_angle = rot_bbox_bag
                    
                    # 使用紫色绘制rot_bbox_bag
                    bag_color = (128, 0, 128)  # 紫色 (BGR格式)
                    
                    # 创建旋转矩形
                    rect = ((float(bag_cx), float(bag_cy)), (float(bag_w), float(bag_h)), float(bag_angle)*180/math.pi)
                    box_points = cv2.boxPoints(rect).astype(int)
                    
                    # 绘制旋转矩形轮廓（虚线效果：使用较细的线条）
                    cv2.drawContours(vis_img, [box_points], 0, bag_color, 2)
                    
                    # 绘制中心点
                    cv2.circle(vis_img, (int(bag_cx), int(bag_cy)), 3, bag_color, -1)
                    
                    # 绘制标签
                    bag_text = f"Bag_RotBox: ({bag_cx:.1f},{bag_cy:.1f},{bag_w:.1f}x{bag_h:.1f},θ={bag_angle:.2f})"
                    cv2.putText(vis_img, bag_text, (int(bag_cx) + 15, int(bag_cy) - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, bag_color, 2)
                
                # 在所有affordance绘制完后，显示grasp_policy信息（在合适的位置）
                if affordances:  # 如果有affordance才显示policy
                    # 找到第一个非空affordance的位置来显示policy
                    first_affordance = next((aff for aff in affordances if aff), None)
                    if first_affordance:
                        if len(first_affordance) >= 2:  # 确保有坐标
                            policy_x, policy_y = int(first_affordance[0]), int(first_affordance[1])
                            policy_text = f"Policy: {grasp_policy} ({len(affordances)} affordances)"
                            cv2.putText(vis_img, policy_text, (policy_x + 10, policy_y - 15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), text_thickness)
                elif handle_hole and len(handle_hole) == 3:
                    # 如果没有affordance但有handle_hole，在handle_hole位置显示policy
                    hh_x, hh_y = handle_hole[0], handle_hole[1]
                    policy_text = f"Policy: {grasp_policy} (handle_hole only)"
                    cv2.putText(vis_img, policy_text, (int(hh_x) + 10, int(hh_y) + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), text_thickness)
            
            return vis_img
                    
        except Exception as e:
            print(f"绘制affordance时发生错误: {e}")
            return vis_img


class GeometryUtils:
    """Utility functions for geometric calculations"""
    
    @staticmethod
    def calculate_iou(mask1, mask2):
        """Calculate IoU between two binary masks"""
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        if np.sum(union) == 0:
            return 0.0
        iou = np.sum(intersection) / np.sum(union)
        return float(iou)
    
    @staticmethod
    def calculate_rotated_bbox_intersection_ratio(affordance, bbox):
        """
        Calculate intersection ratio between rotated bbox (affordance) and normal bbox
        
        Args:
            affordance: [xc, yc, w, h, angle] rotated rectangle
            bbox: [x1, y1, x2, y2] normal rectangle
            
        Returns:
            ratio: intersection area / affordance area
        """
        if bbox is None:
            return 0.0
            
        # Create affordance rotated rectangle
        aff_center = (affordance[0], affordance[1])
        aff_size = (affordance[2], affordance[3])
        aff_angle = math.degrees(affordance[4])  # Convert to degrees
        
        # Get rotated rectangle vertices
        aff_rect = (aff_center, aff_size, aff_angle)
        aff_points = cv2.boxPoints(aff_rect)
        
        # Create normal bbox vertices
        bbox_points = np.array([
            [bbox[0], bbox[1]],  # top-left
            [bbox[2], bbox[1]],  # top-right
            [bbox[2], bbox[3]],  # bottom-right
            [bbox[0], bbox[3]]   # bottom-left
        ], dtype=np.float32)
        
        # Calculate polygon intersection using cv2
        try:
            intersection_area = cv2.intersectConvexConvex(aff_points, bbox_points)[0]
            affordance_area = affordance[2] * affordance[3]  # w * h
            
            if affordance_area > 0:
                ratio = intersection_area / affordance_area
                return float(ratio)
            else:
                return 0.0
        except:
            # If cv2.intersectConvexConvex fails, use backup method
            return GeometryUtils.calculate_polygon_intersection_ratio_backup(
                aff_points, bbox_points, affordance[2] * affordance[3])
    
    @staticmethod
    def calculate_polygon_intersection_ratio_backup(poly1_points, poly2_points, poly1_area):
        """
        Backup polygon intersection calculation method
        """
        try:
            from shapely.geometry import Polygon
            
            poly1 = Polygon(poly1_points)
            poly2 = Polygon(poly2_points)
            
            intersection = poly1.intersection(poly2)
            intersection_area = intersection.area
            
            if poly1_area > 0:
                return float(intersection_area / poly1_area)
            else:
                return 0.0
        except:
            # If shapely is not available, return conservative estimate
            return 0.0
    
    @staticmethod
    def line_intersect(p1, p2, p3, p4):
        """
        Calculate intersection point of two line segments using vector parameterization
        
        Line segment 1: p1 to p2
        Line segment 2: p3 to p4
        
        Returns: intersection point or None (if no intersection)
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        # Check for parallel lines
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is on both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate intersection point
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    @staticmethod
    def get_rotated_rect_from_contour(contour):
        """Get minimum area rotated rectangle from contour"""
        rect = cv2.minAreaRect(contour)
        (cx, cy), (width, height), angle = rect
        
        # OpenCV angle range is [-90, 0], need to convert
        angle_rad = math.radians(angle)
        
        # Ensure width is the longer side
        if width < height:
            width, height = height, width
            angle_rad = angle_rad - math.pi/2
            
        # Normalize angle to [-pi, pi]
        while angle_rad > math.pi:
            angle_rad -= 2*math.pi
        while angle_rad < -math.pi:
            angle_rad += 2*math.pi
            
        return cx, cy, width, height, angle_rad
    
    @staticmethod
    def get_circle_from_contour(contour):
        """Get minimum enclosing circle from contour"""
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        return cx, cy, radius


class MaskUtils:
    """Utility functions for mask operations"""
    
    @staticmethod
    def create_circle_mask(shape, center, radius):
        """Create a circular mask"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.circle(mask, tuple(map(int, center)), int(radius), 255, -1)
        return mask
    
    @staticmethod
    def create_rotated_rect_mask(shape, xc, yc, w, h, angle):
        """Create a rotated rectangle mask"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        rect = ((xc, yc), (w, h), math.degrees(angle))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.fillPoly(mask, [box], 255)
        return mask
    
    @staticmethod
    def get_mask_center(mask):
        """Get center of mass of a binary mask"""
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
        return None
    
    @staticmethod
    def get_mask_contours(mask):
        """Get contours from binary mask"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        return contours


class HoleProcessingUtils:
    """Utility functions for hole detection and processing"""
    
    @staticmethod
    def detect_holes_progressive(mask, area_threshold=3000, max_kernel_size=49):
        """
        Progressive hole detection using morphological operations
        
        Args:
            mask: Binary mask
            area_threshold: Minimum hole area threshold
            max_kernel_size: Maximum kernel size for morphological operations
            
        Returns:
            mask_processed: Processed mask
            optimal_kernel_size: Optimal kernel size found
            has_large_hole: Whether large holes were found
        """
        mask_processed = None
        optimal_kernel_size = 5
        
        for kernel_size in range(5, max_kernel_size + 1, 2):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            # Dilate-erode strategy for hole detection
            temp_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            temp_mask = cv2.erode(temp_mask, (5, 5), iterations=1)
            
            # Detect holes
            temp_contours, temp_hierarchy = cv2.findContours(
                temp_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check for large holes
            has_large_hole = False
            if temp_hierarchy is not None:
                temp_hierarchy = temp_hierarchy[0]
                for i, contour in enumerate(temp_contours):
                    # Inner contour (has parent)
                    if temp_hierarchy[i][3] != -1:
                        area = cv2.contourArea(contour)
                        if area > area_threshold:
                            has_large_hole = True
                            break
            
            if has_large_hole:
                mask_processed = temp_mask
                optimal_kernel_size = kernel_size
                return mask_processed, optimal_kernel_size, True
        
        # If no large holes found, use default
        if mask_processed is None:
            optimal_kernel_size = 25
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (optimal_kernel_size, optimal_kernel_size))
            mask_processed = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
            mask_processed = cv2.erode(mask_processed, kernel, iterations=1)
        
        return mask_processed, optimal_kernel_size, False
    
    @staticmethod
    def extract_hole_contours(mask_processed):
        """
        Extract hole contours from processed mask
        
        Args:
            mask_processed: Processed binary mask
            
        Returns:
            hole_contours: List of hole contours
            holes_mask: Mask of holes
        """
        # Find all contours with hierarchy
        contours_all, hierarchy = cv2.findContours(
            mask_processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create holes mask
        holes_mask = np.zeros(mask_processed.shape, dtype=np.uint8)
        holes_mask = np.ascontiguousarray(holes_mask)
        hole_contours = []
        
        # Find inner contours (holes)
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, contour in enumerate(contours_all):
                # If contour has parent, it's an inner contour (hole)
                if hierarchy[i][3] != -1:
                    holes_mask = np.ascontiguousarray(holes_mask, dtype=np.uint8)
                    cv2.drawContours(holes_mask, [contour], -1, 255, -1)
                    hole_contours.append(contour)
        
        return hole_contours, holes_mask
    
    @staticmethod
    def calculate_adaptive_kernel_size(mask, holes_mask):
        """
        Calculate adaptive kernel size based on hole-to-mask ratio
        
        Args:
            mask: Original binary mask
            holes_mask: Mask of detected holes
            
        Returns:
            kernel_size: Adaptive kernel size
            hole_ratio: Ratio of hole area to local mask area
        """
        h, w = mask.shape
        mask_size = min(h, w)
        hole_ratio = 0
        
        if not np.any(holes_mask):
            return 9, 0  # Default for no holes
        
        # Calculate initial hole area
        initial_hole_area = np.sum(holes_mask > 0)
        
        if initial_hole_area > 0:
            # Get hole contours for bounding box
            hole_contours_temp, _ = cv2.findContours(
                holes_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE)
            
            if hole_contours_temp:
                # Get bounding box of all holes
                x_min, y_min = h, w
                x_max, y_max = 0, 0
                for contour in hole_contours_temp:
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x + w_box)
                    y_max = max(y_max, y + h_box)
                
                # Expand bounding box by 20%
                expand_ratio = 0.2
                w_expand = int((x_max - x_min) * expand_ratio)
                h_expand = int((y_max - y_min) * expand_ratio)
                
                x_min = max(0, x_min - w_expand)
                y_min = max(0, y_min - h_expand)
                x_max = min(w, x_max + w_expand)
                y_max = min(h, y_max + h_expand)
                
                # Calculate local mask area
                local_mask = mask[y_min:y_max, x_min:x_max]
                local_mask_area = np.sum(local_mask > 0)
                
                hole_ratio = initial_hole_area / local_mask_area if local_mask_area > 0 else 0
            else:
                # Use entire mask if no contours found
                mask_area = np.sum(mask > 0)
                hole_ratio = initial_hole_area / mask_area if mask_area > 0 else 0
        
        # Determine kernel size based on ratio
        if hole_ratio > 0.40:
            kernel_size = max(17, int(mask_size * 0.07))
        elif hole_ratio > 0.25:
            kernel_size = max(13, int(mask_size * 0.05))
        elif hole_ratio > 0.15:
            kernel_size = max(11, int(mask_size * 0.04))
        elif hole_ratio > 0.08:
            kernel_size = max(9, int(mask_size * 0.03))
        else:
            kernel_size = max(7, int(mask_size * 0.02))
        
        # Ensure odd kernel size
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        return kernel_size, hole_ratio
    
    @staticmethod
    def fill_holes_adaptive(mask, holes_mask, kernel_size):
        """
        Fill holes using adaptive morphological operations
        
        Args:
            mask: Original binary mask
            holes_mask: Mask of detected holes
            kernel_size: Adaptive kernel size
            
        Returns:
            filled_mask: Mask with holes filled
            filled_holes_mask: Mask of filled holes
        """
        # Create adaptive kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilate holes to connect nearby regions
        filled_holes_mask = cv2.dilate(holes_mask, kernel, iterations=1)
        
        # Combine with original mask
        filled_mask = cv2.bitwise_or(mask, filled_holes_mask)
        
        # Apply morphological closing to smooth
        filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Clean up small artifacts
        clean_kernel_size = max(3, kernel_size // 3)
        clean_kernel_size = clean_kernel_size if clean_kernel_size % 2 == 1 else clean_kernel_size + 1
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_kernel_size, clean_kernel_size))
        filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, clean_kernel)
        
        return filled_mask, filled_holes_mask
    
    @staticmethod
    def find_handle_hole(mask, min_radius=15, max_radius=100):
        """
        Find handle hole (circular hole) in mask
        
        Args:
            mask: Binary mask
            min_radius: Minimum circle radius
            max_radius: Maximum circle radius
            
        Returns:
            handle_hole: [cx, cy, radius] or None
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        
        # Check if it's within reasonable radius range
        if min_radius <= radius <= max_radius:
            return [float(cx), float(cy), float(radius)]
        
        return None
    
    @staticmethod
    def visualize_hole_contours(frame, mask, hole_contours, vis_dir, video_name, frame_id=0, bag_bbox=None):
        """
        Visualize hole contours and save to file
        
        Args:
            frame: Original image
            mask: Bag mask
            hole_contours: List of detected hole contours
            vis_dir: Directory to save visualization
            video_name: Name of the video
            frame_id: Frame number
            bag_bbox: Bag bounding box (optional)
            
        Returns:
            str: Path to saved visualization or None if failed
        """
        try:
            import matplotlib.pyplot as plt
            import random
            
            # Create visualization figure
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 1. Original image
            axes[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # 2. Bag Mask with holes overlay
            mask_overlay = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
            # Mark holes in red
            holes_temp = np.zeros(mask.shape, dtype=np.uint8)
            holes_temp = np.ascontiguousarray(holes_temp)
            for contour in hole_contours:
                cv2.drawContours(holes_temp, [contour], -1, 255, -1)
            mask_overlay[holes_temp > 0] = [255, 0, 0]  # Red for holes
            axes[0, 1].imshow(cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(f'Bag Mask with Holes (red)')
            axes[0, 1].axis('off')
            
            # 3. Detected holes
            holes_vis = np.zeros(mask.shape, dtype=np.uint8)
            holes_vis = np.ascontiguousarray(holes_vis)
            for i, contour in enumerate(hole_contours):
                cv2.drawContours(holes_vis, [contour], -1, 255, -1)
            axes[0, 2].imshow(holes_vis, cmap='hot')
            axes[0, 2].set_title(f'Detected Holes (count: {len(hole_contours)})')
            axes[0, 2].axis('off')
            
            # 4. Individual hole contours
            contour_vis = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            hole_info = []
            for i, contour in enumerate(hole_contours):
                color = colors[i % len(colors)]
                cv2.drawContours(contour_vis, [contour], -1, color, 2)
                # Calculate contour center and area
                M = cv2.moments(contour)
                area = cv2.contourArea(contour)
                hole_info.append(f"Hole {i}: area={area:.0f}")
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(contour_vis, f"{i}", (cx-10, cy+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            axes[1, 0].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
            info_text = ', '.join(hole_info[:3])  # Show first 3 holes
            if len(hole_info) > 3:
                info_text += f', ... ({len(hole_info)} total)'
            axes[1, 0].set_title(f'Contours: {info_text}')
            axes[1, 0].axis('off')
            
            # 5. Largest hole
            if hole_contours:
                largest_hole = max(hole_contours, key=cv2.contourArea)
                largest_vis = np.zeros(mask.shape, dtype=np.uint8)
                largest_vis = np.ascontiguousarray(largest_vis)
                cv2.drawContours(largest_vis, [largest_hole], -1, 255, -1)
                axes[1, 1].imshow(largest_vis, cmap='hot')
                area = cv2.contourArea(largest_hole)
                # Calculate hole to mask ratio
                mask_area = np.sum(mask > 0)
                ratio = area / mask_area if mask_area > 0 else 0
                axes[1, 1].set_title(f'Largest Hole (area: {area:.0f}, ratio: {ratio:.3f})')
                axes[1, 1].axis('off')
                
                # 6. Circle fit for largest hole
                (cx, cy), radius = cv2.minEnclosingCircle(largest_hole)
                circle_vis = frame.copy()
                # Draw circle
                cv2.circle(circle_vis, (int(cx), int(cy)), int(radius), (0, 255, 255), 3)
                cv2.circle(circle_vis, (int(cx), int(cy)), 5, (0, 255, 255), -1)
                # Add text annotation
                cv2.putText(circle_vis, f"r={radius:.1f}", 
                           (int(cx-radius), int(cy-radius-10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                axes[1, 2].imshow(cv2.cvtColor(circle_vis, cv2.COLOR_BGR2RGB))
                axes[1, 2].set_title(f'Circle Fit (center: ({cx:.0f},{cy:.0f}), r: {radius:.1f})')
                axes[1, 2].axis('off')
            else:
                axes[1, 1].text(0.5, 0.5, 'No holes detected', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].axis('off')
                axes[1, 2].axis('off')
            
            # Set overall title
            frame_str = f"frame_{frame_id:06d}"
            plt.suptitle(f'Hole Contours - {video_name} - {frame_str}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Generate filename
            random_suffix = random.randint(1000, 9999)
            vis_filename = f"vis_hc_{video_name}_{frame_str}_{random_suffix}.png"
            vis_path = os.path.join(vis_dir, vis_filename)
            
            plt.savefig(vis_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"  Hole contours visualization saved: {vis_filename}")
            return vis_path
            
        except Exception as e:
            print(f"  Error visualizing hole_contours: {e}")
            return None


class VideoUtils:
    """Utility functions for video processing and COCO annotations"""
    
    @staticmethod
    def create_visualization_video(vis_dir, video_name, target_fps=0.5):
        """
        Create visualization video from images in vis directory
        
        Args:
            vis_dir: Directory containing visualization images
            video_name: Name for the output video
            target_fps: Target frame rate for the video
            
        Returns:
            str: Path to the created video or None if failed
        """
        try:
            import glob
            import re
            
            # Find all visualization images
            vis_pattern = os.path.join(vis_dir, f"vis_{video_name}_frame_*.jpg")
            vis_files = glob.glob(vis_pattern)
            
            if not vis_files:
                print(f"No visualization images found: {vis_pattern}")
                return None
            
            # Sort by frame number
            def get_frame_number(filepath):
                match = re.search(r'frame_(\d+)', filepath)
                return int(match.group(1)) if match else 0
            
            vis_files.sort(key=get_frame_number)
            
            print(f"\nGenerating visualization video...")
            print(f"  Found {len(vis_files)} visualization images")
            
            # Read first image to get dimensions
            first_img = cv2.imread(vis_files[0])
            if first_img is None:
                print(f"Cannot read image: {vis_files[0]}")
                return None
            
            height, width = first_img.shape[:2]
            
            # Adjust frame rate: ensure not too low for encoding
            adjusted_fps = max(1.0, target_fps)  # Minimum 1fps
            if adjusted_fps != target_fps:
                print(f"  Adjusting frame rate: {target_fps:.2f} -> {adjusted_fps:.2f}")
            
            # Try multiple codecs and formats - prefer MP4
            codecs_to_try = [
                ('mp4v', 'mp4'),
                ('H264', 'mp4'),
                ('avc1', 'mp4'),
                ('XVID', 'avi'),
                ('MJPG', 'avi'),
            ]
            
            output_path = None
            out = None
            
            for codec, ext in codecs_to_try:
                try:
                    temp_output_path = os.path.join(vis_dir, f"{video_name}_visualization.{ext}")
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    temp_out = cv2.VideoWriter(temp_output_path, fourcc, adjusted_fps, (width, height))
                    
                    if temp_out.isOpened():
                        output_path = temp_output_path
                        out = temp_out
                        print(f"  Using codec: {codec}, output format: {ext}")
                        break
                    else:
                        temp_out.release()
                        
                except Exception as e:
                    print(f"  Codec {codec} test failed: {e}")
            
            if out is None or not out.isOpened():
                print(f"  Cannot create video writer, tried codecs: {[c for c, _ in codecs_to_try]}")
                return None
            
            # Write all frames
            frame_written = 0
            for vis_file in vis_files:
                img = cv2.imread(vis_file)
                if img is not None:
                    # Check size consistency
                    h, w = img.shape[:2]
                    if (w, h) != (width, height):
                        img = cv2.resize(img, (width, height))
                    
                    out.write(img)
                    frame_written += 1
                    if frame_written % 10 == 0:
                        print(f"  Processed {frame_written}/{len(vis_files)} frames...")
                else:
                    print(f"  Warning: Cannot read image {vis_file}")
            
            # Release video writer
            out.release()
            
            # Get output file size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                
                # Verify generated video
                test_cap = cv2.VideoCapture(output_path)
                if test_cap.isOpened():
                    frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = test_cap.get(cv2.CAP_PROP_FPS)
                    test_cap.release()
                    print(f"  Visualization video generated: {output_path}")
                    print(f"  Video info: {width}x{height}, {fps:.2f}fps, {frame_count} frames, {file_size:.2f}MB")
                else:
                    print(f"  Warning: Video file generated but cannot verify")
                
                return output_path
            else:
                print(f"  Video generation failed")
                return None
                
        except Exception as e:
            print(f"Error generating visualization video: {e}")
            import traceback
            traceback.print_exc()
            return None
    

class COCOUtils:
    """Utility functions for COCO format operations"""
    
    @staticmethod
    def xywh_to_xyxy(bbox):
        """Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]"""
        x, y, w, h = bbox
        return [x, y, x + w, y + h]
    
    @staticmethod
    def xyxy_to_xywh(bbox):
        """Convert bbox from [x1, y1, x2, y2] to [x, y, w, h]"""
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]
    
    @staticmethod
    def encode_mask(mask):
        """Encode binary mask to RLE format"""
        return coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    
    @staticmethod
    def decode_mask(mask_rle):
        """Decode RLE format to binary mask"""
        return coco_mask.decode(mask_rle)
    
    @staticmethod
    def mask_to_polygon(mask):
        """Convert binary mask to polygon points"""
        contours = MaskUtils.get_mask_contours(mask)
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Simplify contour
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            # Convert to list of points
            polygon = approx.reshape(-1, 2).tolist()
            return polygon
        return []
    
    @staticmethod
    def save_coco_annotations(coco_data, ann_dir, video_name, train_ratio=0.8):
        """
        Save COCO format annotation files, split into train.json and val.json
        
        Args:
            coco_data: COCO format data dictionary
            ann_dir: Directory to save annotation files
            video_name: Name of the video (for filename)
            train_ratio: Ratio for train/val split (default 0.8)
            
        Returns:
            dict: Paths to saved files
        """
        if not coco_data['images']:
            return {}
        
        # Get all image IDs
        image_ids = [img['id'] for img in coco_data['images']]
        
        # Split by ratio
        split_idx = int(len(image_ids) * train_ratio)
        train_image_ids = set(image_ids[:split_idx])
        val_image_ids = set(image_ids[split_idx:])
        
        # Create train and val data
        train_data = {
            'info': coco_data['info'],
            'licenses': coco_data['licenses'],
            'categories': coco_data['categories'],
            'images': [],
            'annotations': []
        }
        
        val_data = {
            'info': coco_data['info'],
            'licenses': coco_data['licenses'],
            'categories': coco_data['categories'],
            'images': [],
            'annotations': []
        }
        
        # Split image data
        for img in coco_data['images']:
            if img['id'] in train_image_ids:
                train_data['images'].append(img)
            else:
                val_data['images'].append(img)
        
        # Split annotation data
        for ann in coco_data['annotations']:
            if ann['image_id'] in train_image_ids:
                train_data['annotations'].append(ann)
            else:
                val_data['annotations'].append(ann)
        
        # Save train set
        train_path = os.path.join(ann_dir, 'train.json')
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        # Save val set
        val_path = os.path.join(ann_dir, 'val.json')
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        # Save full annotation file (optional)
        full_path = os.path.join(ann_dir, f"{video_name}.json")
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        print(f"Train annotation file saved: {train_path} ({len(train_data['images'])} images, {len(train_data['annotations'])} annotations)")
        print(f"Val annotation file saved: {val_path} ({len(val_data['images'])} images, {len(val_data['annotations'])} annotations)")
        print(f"Full annotation file saved: {full_path} ({len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations)")
        
        return {
            'train': train_path,
            'val': val_path,
            'full': full_path
        }