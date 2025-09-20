import os
import base64
from openai import OpenAI
from typing import List, Dict, Union
import json
import re
import cv2
import numpy as np
from mmengine.config import Config

# Import DINO APIs
try:
    from .xdino import DetectionAPI, GraspAnythingAPI
except ImportError:
    try:
        from xdino import DetectionAPI, GraspAnythingAPI
    except ImportError:
        print("Warning: xdino module not found. Detection and grasp features will be disabled.")
        DetectionAPI = None
        GraspAnythingAPI = None


class Xbrain:
    """Robot Agent Brain for task planning and control"""
    
    def __init__(self, api_key: str = None, detection_config: Dict = None, grasp_config: Dict = None, 
                 use_default_configs: bool = True, swap_tmp_area: tuple = (0, 0)):
        """
        Initialize Xbrain with API key and DINO configs
        
        Args:
            api_key: DashScope API key. If None, will try to get from environment
            detection_config: Configuration for DetectionAPI
            grasp_config: Configuration for GraspAnythingAPI
            use_default_configs: Whether to use default configs from xdino.py
            swap_tmp_area: Temporary area coordinates (x, y) for object swapping, default (0, 0)
        """
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                # Fallback to hardcoded key for testing
                api_key = "sk-e622f19390f646dab2a083ee16deb64c"
        
        if not api_key:
            raise ValueError("API key not available")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # Store temporary swap area coordinates
        self.swap_tmp_area = swap_tmp_area
        
        # Initialize DINO APIs
        self.detection_api = None
        self.grasp_api = None
        
        # Use default configs from xdino.py if not provided
        if use_default_configs:
            if detection_config is None:
                detection_config = self._get_default_detection_config()
            if grasp_config is None:
                grasp_config = self._get_default_grasp_config()
        
        if DetectionAPI is not None and detection_config:
            try:
                cfg = Config()
                for key, value in detection_config.items():
                    setattr(cfg, key, value)
                self.detection_api = DetectionAPI(cfg)
                print("DetectionAPI initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize DetectionAPI: {e}")
        
        if GraspAnythingAPI is not None and grasp_config:
            try:
                cfg = Config()
                for key, value in grasp_config.items():
                    setattr(cfg, key, value)
                self.grasp_api = GraspAnythingAPI(cfg)
                print("GraspAnythingAPI initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize GraspAnythingAPI: {e}")
    
    def _get_default_detection_config(self):
        """Get default detection config from xdino.py main function"""
        return {
            'uri': '/v2/task/dinox/detection',
            'status_uri': '/v2/task_status', 
            'token': 'c4cdacb48bc4d1a1a335c88598a18e8c',
            'model_name': 'DINO-X-1.0'
        }
    
    def _get_default_grasp_config(self):
        """Get default grasp config from xdino.py main function"""
        return {
            'server_list': '/Users/didi/Documents/AgileX-DINO/AnyBagGrasp/xagent/server_grasp.json',
            'model_name': 'full'
        }
        
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode local image to base64 data URL"""
        try:
            with open(image_path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{encoded}"
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
    
    def _parse_action_list(self, response_text: str) -> List[Dict]:
        """
        Parse the VLM response to extract structured action list and detailed object analysis
        
        Args:
            response_text: Raw response from VLM with structured object analysis
            
        Returns:
            List of action dictionaries with enhanced object information
        """
        import re
        actions = []
        
        # Parse detailed object information from the structured response
        object_analysis = self._parse_object_analysis(response_text)
        spatial_analysis = self._parse_spatial_analysis(response_text)
        
        # Look for pick_n_place function calls - handle both inline and code block formats
        # First pattern: backtick-wrapped function calls (may include nested parentheses)
        pick_pattern1 = r'`pick_n_place\s*\(([^`]+)\)`'
        # Second pattern: regular function calls (may include nested parentheses)
        pick_pattern2 = r'pick_n_place\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)'
        
        matches = []
        # Try backtick pattern first (more specific)
        backtick_matches = re.findall(pick_pattern1, response_text, re.IGNORECASE)
        matches.extend(backtick_matches)
        # Then try regular pattern but avoid duplicates
        regular_matches = re.findall(pick_pattern2, response_text, re.IGNORECASE)
        # Filter out matches that were already found with backticks
        for rm in regular_matches:
            if rm not in matches:
                matches.append(rm)
        
        print(f"DEBUG: Found {len(matches)} pick_n_place function calls")
        
        for i, match in enumerate(matches):
            # Parse parameters from the function call
            params = {}
            description = f"pick_n_place({match})"
            
            # Extract object_id - handle both quoted and unquoted, including Chinese characters
            object_match = re.search(r'object_id\s*=\s*"([^"]+)"', match)
            if not object_match:
                object_match = re.search(r"object_id\s*=\s*'([^']+)'", match)
            if not object_match:
                # Try without quotes
                object_match = re.search(r'object_id\s*=\s*([^,\)]+)', match)
            
            if object_match:
                obj_id = object_match.group(1).strip()
                # Clean up object ID but preserve Chinese characters
                if obj_id and obj_id != 'object_id':  # Filter out placeholder values
                    params['object_id'] = obj_id
            
            # Extract source_position
            src_match = re.search(r'source_position\s*=\s*\(([^)]+)\)', match)
            if src_match:
                coords = src_match.group(1).split(',')
                if len(coords) >= 2:
                    try:
                        x = float(coords[0].strip())
                        y = float(coords[1].strip())
                        # Validate coordinates are reasonable
                        if 0 <= x <= 10000 and 0 <= y <= 10000:
                            params['source_position'] = (x, y)
                    except ValueError:
                        pass
            
            # Extract target_position - handle coordinates and special values
            if 'swap_tmp_area' in match:
                params['target_position'] = 'swap_tmp_area'
                params['use_temporary_area'] = True
            else:
                # Try to match coordinate format first
                pos_match = re.search(r'target_position\s*=\s*\(([^)]+)\)', match)
                if pos_match:
                    coords = pos_match.group(1).split(',')
                    if len(coords) >= 2:
                        try:
                            x = float(coords[0].strip())
                            y = float(coords[1].strip())
                            # Validate coordinates are reasonable
                            if 0 <= x <= 10000 and 0 <= y <= 10000:
                                params['target_position'] = (x, y)
                            else:
                                # Coordinates out of reasonable range
                                params['target_position'] = pos_match.group(1).strip()
                        except ValueError:
                            params['target_position'] = pos_match.group(1).strip()
                else:
                    # Try to match any value after target_position=
                    target_match = re.search(r'target_position\s*=\s*([^,\)]+)', match)
                    if target_match:
                        params['target_position'] = target_match.group(1).strip()
            
            # Extract color and object information for fallback
            if not params.get('object_id'):
                # Look for object IDs in the match
                object_id_patterns = [r'object_(\d+)', r'(\w+)_bag', r'(\w+)色', r'(pink|yellow|blue|粉色|黄色|蓝色)']
                for pattern in object_id_patterns:
                    obj_match = re.search(pattern, match, re.IGNORECASE)
                    if obj_match:
                        params['object_id'] = obj_match.group(1)
                        break
            
            # Add detailed object information if available
            if params.get('object_id') and object_analysis:
                for obj_id, obj_info in object_analysis.items():
                    if params['object_id'] in obj_id or obj_id in params['object_id']:
                        params['object_details'] = obj_info
                        break
            
            # Only add action if it has valid parameters (both object_id and target_position)
            if params.get('object_id') and params.get('target_position'):
                actions.append({
                    'type': 'pick_n_place',
                    'description': description,
                    'params': params,
                    'object_analysis': object_analysis,
                    'spatial_analysis': spatial_analysis
                })
        
        # If no function calls found, look for numbered action sequences
        if not actions:
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for numbered actions or key action words
                if (re.match(r'^\d+\.', line) and 
                    any(keyword in line.lower() for keyword in ['pick_n_place', '抓取', '移动', '放置', 'object_'])):
                    
                    action = {
                        'type': 'pick_n_place',
                        'description': line,
                        'params': {},
                        'object_analysis': object_analysis,
                        'spatial_analysis': spatial_analysis
                    }
                    
                    # Extract object information from the line
                    object_patterns = [
                        r'object_(\w+)', r'(\w+)_bag', r'(\w+)色袋', 
                        r'(pink|yellow|blue|粉色|黄色|蓝色)', r'object_(\d+)'
                    ]
                    for pattern in object_patterns:
                        obj_match = re.search(pattern, line, re.IGNORECASE)
                        if obj_match:
                            action['params']['object_id'] = obj_match.group(1)
                            break
                    
                    # Extract coordinate information
                    coord_match = re.search(r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)', line)
                    if coord_match:
                        try:
                            x, y = float(coord_match.group(1)), float(coord_match.group(2))
                            action['params']['target_position'] = (x, y)
                        except ValueError:
                            pass
                    
                    # Extract relative position information
                    if not action['params'].get('target_position'):
                        if '左' in line or 'left' in line.lower():
                            action['params']['target_position'] = 'left'
                        elif '中' in line or 'middle' in line.lower():
                            action['params']['target_position'] = 'middle'
                        elif '右' in line or 'right' in line.lower():
                            action['params']['target_position'] = 'right'
                        elif 'swap_tmp_area' in line:
                            action['params']['target_position'] = 'swap_tmp_area'
                            action['params']['use_temporary_area'] = True
                    
                    actions.append(action)
        
        # If still no specific actions found, create a comprehensive analysis action
        if not actions:
            actions = [
                {
                    'type': 'scene_analysis',
                    'description': '详细场景分析和物体识别',
                    'params': {},
                    'object_analysis': object_analysis,
                    'spatial_analysis': spatial_analysis
                }
            ]
            
            # Add a plan execution action with the full response
            if len(response_text) > 100:  # Only add if there's substantial content
                actions.append({
                    'type': 'plan_execution',
                    'description': response_text[:300] + '...' if len(response_text) > 300 else response_text,
                    'params': {},
                    'full_analysis': response_text
                })
        
        # Remove duplicate actions but preserve detailed information
        seen = set()
        unique_actions = []
        for action in actions:
            action_key = (action['type'], action['params'].get('object_id', ''), 
                         str(action['params'].get('target_position', '')))
            if action_key not in seen:
                seen.add(action_key)
                unique_actions.append(action)
        
        # Insert reset_to_home after each pick_n_place operation
        enhanced_actions = []
        for action in unique_actions:
            enhanced_actions.append(action)
            
            # Add reset_to_home after each pick_n_place
            if action['type'] == 'pick_n_place':
                reset_action = {
                    'type': 'reset_to_home',
                    'description': '返回初始位置',
                    'params': {}
                }
                enhanced_actions.append(reset_action)
        
        return enhanced_actions[:20]  # Limit to prevent excessive actions
    
    def _parse_object_analysis(self, response_text: str) -> Dict:
        """
        Parse detailed object analysis from the structured VLM response
        
        Args:
            response_text: Raw response text containing object analysis
            
        Returns:
            Dictionary with object IDs as keys and detailed info as values
        """
        import re
        objects = {}
        
        # Look for object analysis sections - updated to handle Chinese ID format
        # Pattern to match object ID with possible markdown formatting and full Chinese names
        object_pattern = r'物体ID:\s*\**\s*([^*\n]+?)\s*\**\s*(.*?)(?=物体ID:|\*\*第[二三四]步|---\n\*\*物体ID:|$)'
        matches = re.findall(object_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        for object_id, content in matches:
            # Clean up object_id (remove trailing spaces and special chars but keep full name)
            object_id = object_id.strip().rstrip('*').strip()
            # Remove any extra colons or asterisks
            object_id = object_id.rstrip(':').strip()
            obj_info = {}
            
            # Parse each attribute - handle both [] and non-bracketed formats
            attributes = {
                '类别': [r'类别:\s*\[([^\]]+)\]', r'类别:\s*([^\n]+)'],
                '颜色': [r'颜色:\s*\[([^\]]+)\]', r'颜色:\s*([^\n]+)'],
                '大小': [r'大小:\s*\[([^\]]+)\]', r'大小:\s*([^\n]+)'],
                '位置': [r'位置:\s*\[([^\]]+)\]', r'位置:\s*([^\n]+)'],
                '形状': [r'形状:\s*\[([^\]]+)\]', r'形状:\s*([^\n]+)'],
                '材质属性': [r'材质属性:\s*\[([^\]]+)\]', r'材质属性:\s*([^\n]+)'],
                '朝向角度': [r'朝向角度:\s*\[([^\]]+)\]', r'朝向角度:\s*([^\n]+)'],
                '稳定性': [r'稳定性:\s*\[([^\]]+)\]', r'稳定性:\s*([^\n]+)'],
                '可抓取点': [r'可抓取点:\s*\[([^\]]+)\]', r'可抓取点:\s*([^\n]+)']
            }
            
            for attr_name, patterns in attributes.items():
                for pattern in patterns:
                    match = re.search(pattern, content)
                    if match:
                        value = match.group(1).strip()
                        # Clean up the value
                        value = value.rstrip('-').strip()
                        if value and value != '[' and not value.startswith('*'):
                            obj_info[attr_name] = value
                            break
            
            # Only add object if it has some attributes
            if obj_info:
                objects[object_id] = obj_info
        
        return objects
    
    def _parse_spatial_analysis(self, response_text: str) -> Dict:
        """
        Parse spatial relationship analysis from the VLM response
        
        Args:
            response_text: Raw response text containing spatial analysis
            
        Returns:
            Dictionary with spatial relationship information
        """
        import re
        spatial_info = {}
        
        # Look for spatial analysis sections - handle both [] and non-bracketed formats, including multiline content
        sections = {
            '整体布局': [r'整体布局:\s*\[([^\]]+)\]', r'整体布局:\s*([^*\n]+(?:\n[^*\n-]+)*)'],
            '左右顺序': [r'左右顺序:\s*\[([^\]]+)\]', r'左右顺序:\s*((?:[^*\n]+(?:\n\s*-[^\n]+)*)+)'],
            '前后关系': [r'前后关系:\s*\[([^\]]+)\]', r'前后关系:\s*([^*\n]+(?:\n[^*\n-]+)*)'],
            '相对距离': [r'相对距离:\s*\[([^\]]+)\]', r'相对距离:\s*([^*\n]+(?:\n[^*\n-]+)*)'],
            '空间占用': [r'空间占用:\s*\[([^\]]+)\]', r'空间占用:\s*([^*\n]+(?:\n[^*\n-]+)*)'],
            '自由空间': [r'自由空间:\s*\[([^\]]+)\]', r'自由空间:\s*([^*\n]+(?:\n[^*\n-]+)*)'],
            '相邻接触': [r'相邻接触:\s*\[([^\]]+)\]', r'相邻接触:\s*([^*\n]+(?:\n[^*\n-]+)*)'],
            '重叠情况': [r'重叠情况:\s*\[([^\]]+)\]', r'重叠情况:\s*([^*\n]+(?:\n[^*\n-]+)*)'],
            '支撑关系': [r'支撑关系:\s*\[([^\]]+)\]', r'支撑关系:\s*([^*\n]+(?:\n[^*\n-]+)*)']
        }
        
        for section_name, patterns in sections.items():
            for pattern in patterns:
                match = re.search(pattern, response_text, re.DOTALL | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value - preserve line breaks for multi-line content
                    lines = value.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and line != '[' and not line.startswith('*'):
                            # Keep lines that start with dash (list items)
                            if line.startswith('-'):
                                cleaned_lines.append(line)
                            elif cleaned_lines and cleaned_lines[-1].startswith('-'):
                                # This might be a continuation of previous list item
                                cleaned_lines[-1] += ' ' + line
                            else:
                                cleaned_lines.append(line)
                    
                    if cleaned_lines:
                        spatial_info[section_name] = '\n'.join(cleaned_lines)
                        break
        
        return spatial_info
    
    def _format_grasp_info(self, grasp_points: List[Dict]) -> str:
        """
        Format grasp detection results for VLM prompt
        
        Args:
            grasp_points: List of grasp detection results
            
        Returns:
            Formatted string describing grasp information
        """
        if not grasp_points or 'error' in grasp_points[0]:
            return "抓取检测: 未检测到可抓取对象"
        
        grasp_info = []
        for i, grasp in enumerate(grasp_points):
            # Remove mask field and format key information
            grasp_desc = f"对象{i+1}:"
            
            # Add category information if available
            category = grasp.get('category', '未知')
            if category and category != '未知':
                grasp_desc += f" 类别({category})"
            
            bbox = grasp.get('bbox', [])
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                grasp_desc += f" 位置({center_x:.0f},{center_y:.0f})"
            
            affordances = grasp.get('affordances', [])
            if affordances:
                grasp_desc += f" 抓取候选点{len(affordances)}个"
                # Format first few affordances
                for j, aff in enumerate(affordances[:2]):
                    if len(aff) >= 5:
                        x, y, w, h, angle = aff[:5]
                        grasp_desc += f" 点{j+1}:({x:.0f},{y:.0f},角度{angle:.1f})"
            
            touching_points = grasp.get('touching_points', [])
            if touching_points:
                grasp_desc += f" 接触点{len(touching_points)}个"
            
            scores = grasp.get('scores', [])
            if scores:
                avg_score = sum(scores) / len(scores)
                grasp_desc += f" 置信度{avg_score:.2f}"
            
            grasp_info.append(grasp_desc)
        
        return "抓取检测结果:\n" + "\n".join(grasp_info)

    def _format_object_info(self, objects: List[Dict]) -> str:
        """
        Format object detection results for VLM prompt
        
        Args:
            objects: List of object detection results
            
        Returns:
            Formatted string describing object information
        """
        if not objects or 'error' in objects[0]:
            return "对象检测: 未检测到对象"
        
        object_info = []
        for i, obj in enumerate(objects):
            # Remove mask field and format key information
            obj_desc = f"对象{i+1}:"
            
            # Add category information if available
            category = obj.get('category', '未知')
            if category and category != '未知':
                obj_desc += f" 类别({category})"
            
            bbox = obj.get('bbox', [])
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                obj_desc += f" 位置({center_x:.0f},{center_y:.0f}) 尺寸({width:.0f}x{height:.0f})"
            
            # Add detection confidence score
            score = obj.get('score', 0.0)
            if score > 0:
                obj_desc += f" 置信度{score:.2f}"
            
            object_info.append(obj_desc)
        
        return "对象检测结果:\n" + "\n".join(object_info)

    def get_plan(self, image: Union[str, bytes], text: str, model: str = "qwen-vl-max-latest") -> List[Dict]:
        """
        Get action plan from image and text input, enhanced with grasp detection
        
        Args:
            image: Image file path (str) or image bytes
            text: Task description text
            model: VLM model to use
            
        Returns:
            List of action dictionaries with structure:
            [
                {
                    'type': 'action_type',
                    'description': 'human readable description',
                    'params': {'key': 'value'}
                }
            ]
        """
        try:
            # Handle image input
            if isinstance(image, str):
                if image.startswith('http'):
                    image_url = image
                else:
                    image_url = self._encode_image_to_base64(image)
            else:
                encoded = base64.b64encode(image).decode('utf-8')
                image_url = f"data:image/jpeg;base64,{encoded}"
            
            # Get object detection information
            objects = self.get_objects(image)
            #objects = self.get_grasp_points(image)
            #print(f"DEBUG: Detection API initialized: {self.detection_api is not None}")
            #print(f"DEBUG: Objects result: {objects[:2] if objects else 'None'}")
            
            # Enhance task description with object information
            object_info_text = self._format_object_info(objects)
            #object_info_text = self._format_object_info(objects)
            enhanced_text = f"{object_info_text}\n\n {text}"
            
            # Prepare system prompt for structured output
            system_prompt = f"""你是智能机器人Agent大脑，具备视觉理解、空间推理和精确操作能力。请按照以下结构化格式逐步分析并输出：

**第一步：逐个物体详细分析**
请仔细分析图像中的每个物体，逐个输出详细信息：

**物体分析格式（逐个输出）：**
物体ID: [根据物体特征生成有意义的中文名称，格式为：颜色+大小+类别+序号，如"粉色中等手提袋1"、"黄色小型手提袋2"、"蓝色大型盒子3"]
- 类别: [具体类别名称，如"手提袋"、"盒子"等]
- 颜色: [主要颜色和辅助颜色，如"粉色主体，黑色手柄"]
- 大小: [相对大小描述和像素尺寸，如"中等大小，约120x80像素"]
- 位置: [精确坐标和区域描述，如"(245, 180)，位于图像左侧区域"]
- 形状: [几何形状特征，如"矩形袋状，略扁平"]
- 材质属性: [表面材质，如"光滑塑料材质"、"编织布料"]
- 朝向角度: [物体朝向，如"正面朝向相机"、"侧面45度角"]
- 稳定性: [抓取稳定性评估，如"底部平稳，易于抓取"]
- 可抓取点: [推荐抓取位置，如"顶部手柄，边缘位置"]

**物体ID命名规则：**
- 颜色：粉色、黄色、蓝色、红色、绿色、白色、黑色、橙色、紫色等
- 大小：小型、中等、大型、微小、巨大
- 类别：手提袋、盒子、瓶子、杯子、书本、玩具等
- 序号：如果同类物体较多时添加数字后缀
- 示例：粉色中等手提袋1、黄色小型盒子2、蓝色大型书本3

**第二步：空间关系和布局分析**
详细描述场景中各物体的空间关系：

**布局分析：**
- 整体布局: [描述物体在场景中的总体分布模式]
- 左右顺序: [从左到右列出物体排列，如"左侧：粉色袋，中间：黄色袋，右侧：蓝色袋"]
- 前后关系: [描述物体的前后层次，如"所有物体位于同一平面，无前后遮挡"]
- 相对距离: [物体间距离，如"粉色袋与黄色袋间距约50像素"]
- 空间占用: [每个物体占用的空间范围]
- 自由空间: [识别可用于移动的空白区域]

**接触关系：**
- 相邻接触: [哪些物体相互接触或临近]
- 重叠情况: [是否存在物体重叠或遮挡]
- 支撑关系: [物体的支撑情况和稳定性]

**第三步：任务理解和冲突分析**
基于详细的物体和空间分析：

**目标状态分析：**
- 当前状态: [详细描述当前物体排列]
- 目标状态: [详细描述期望的最终排列]
- 变化需求: [明确哪些物体需要移动到哪里]

**移动冲突识别：**
- 直接冲突: [哪些移动会直接产生空间冲突]
- 路径冲突: [移动路径上的潜在障碍]
- 临时区域需求: [是否需要使用临时放置区域：swap_tmp_area=({self.swap_tmp_area[0]},{self.swap_tmp_area[1]})]

**第四步：精确动作规划**
基于深度分析，输出结构化动作序列：

**可用API：**
- pick_n_place(object_id, source_position, target_position) - 从源位置拾取物体并放置到目标位置
- reset_to_home() - 返回初始位置（自动插入）

**注意事项：**
- object_id：使用第一步分析中的物体ID
- source_position：物体当前位置坐标(x,y)，从第一步分析的位置信息中获取
- target_position：目标位置，可以是精确坐标(x,y)或swap_tmp_area临时区域

**执行策略：**
[详细说明移动顺序的逻辑依据，冲突避免方案，效率优化考虑]

**动作序列：**
1. pick_n_place(object_id="粉色中等手提袋1", source_position=(当前x,y), target_position=(目标x,y)) - [详细操作描述和理由]
2. pick_n_place(object_id="黄色小型手提袋2", source_position=(当前x,y), target_position=swap_tmp_area) - [移至临时区的具体原因]
3. pick_n_place(object_id="蓝色大型手提袋3", source_position=(当前x,y), target_position=(目标x,y)) - [详细操作描述和理由]

**质量要求：**
- 坐标必须精确到像素级别，避免空间冲突
- 物体标识要与前面分析的物体ID一致
- 考虑操作的物理可行性和安全性
- 优化移动次数，提高整体效率
- 每个动作都要有明确的执行理由"""
            print("=== VLM Prompt ===")
            print("System Prompt:\n", system_prompt)
            print("User Prompt:\n", enhanced_text)
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                },
                            },
                            {"type": "text", "text": enhanced_text},
                        ],
                    },
                ],
            )
            
            response_text = completion.choices[0].message.content
            print("=== VLM Response ===")
            print(response_text)
            action_list = self._parse_action_list(response_text)
            
            # Print detailed analysis results
            print("\n=== 解析结果 ===")
            
            # Extract and print object information
            if action_list and len(action_list) > 0 and 'object_analysis' in action_list[0]:
                object_info = action_list[0]['object_analysis']
                if object_info:
                    print(f"\n【物体信息 Object Info】: 检测到{len(object_info)}个物体")
                    for obj_id, details in object_info.items():
                        print(f"  ► {obj_id}:")
                        if details:
                            for attr, value in details.items():
                                print(f"      {attr}: {value}")
                        else:
                            print(f"      (无详细属性信息)")
                        print()
                else:
                    print("\n【物体信息 Object Info】: 解析结果为空")
            else:
                print("\n【物体信息 Object Info】: 未解析到详细物体信息")
            
            # Extract and print spatial information  
            if action_list and len(action_list) > 0 and 'spatial_analysis' in action_list[0]:
                spatial_info = action_list[0]['spatial_analysis']
                if spatial_info:
                    print(f"\n【空间信息 Spatial Info】: {len(spatial_info)}个空间属性")
                    for section, content in spatial_info.items():
                        print(f"  ► {section}: {content}")
                else:
                    print("\n【空间信息 Spatial Info】: 解析结果为空")
            else:
                print("\n【空间信息 Spatial Info】: 未解析到空间关系信息")
            
            # Print action list
            print(f"\n【规划信息 Action List】: 共{len(action_list)}个动作")
            for i, action in enumerate(action_list, 1):
                # Format action display based on type
                if action['type'] == 'pick_n_place':
                    obj_id = action['params'].get('object_id', 'unknown')
                    source = action['params'].get('source_position', None)
                    target = action['params'].get('target_position', 'unknown')
                    
                    # Format source position
                    if source and isinstance(source, tuple) and len(source) == 2:
                        source_str = f"({source[0]:.0f}, {source[1]:.0f})"
                    else:
                        source_str = "当前位置"
                    
                    # Format target position
                    if isinstance(target, tuple) and len(target) == 2:
                        target_str = f"({target[0]:.0f}, {target[1]:.0f})"
                    else:
                        target_str = str(target)
                    
                    print(f"  {i}. pick_n_place: '{obj_id}' 从{source_str} -> 移动到{target_str}")
                elif action['type'] == 'reset_to_home':
                    print(f"  {i}. reset_to_home: 返回初始位置")
                else:
                    print(f"  {i}. {action['type']}: {action.get('description', '')}")
            
            return action_list
            
        except Exception as e:
            return [
                {
                    'type': 'error',
                    'description': f'执行失败: {str(e)}',
                    'params': {'error': str(e)}
                }
            ]

    
    def get_objects(self, image: Union[str, np.ndarray], prompt_text: str = "pink bag. yellow bag. blue bag", 
                   bbox_threshold: float = 0.25, iou_threshold: float = 0.8) -> List[Dict]:
        """
        Detect objects in the image using DetectionAPI
        
        Args:
            image: Image file path or numpy array
            prompt_text: Natural language description of objects to detect
            bbox_threshold: Bounding box confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detected objects with bbox, category, score, etc.
        """
        if self.detection_api is None:
            return [{'error': 'DetectionAPI not initialized'}]
        
        try:
            # Handle different image input types
            if isinstance(image, str):
                if image.startswith('http'):
                    # Download image for processing
                    import requests
                    response = requests.get(image)
                    rgb = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                else:
                    # Load local image
                    rgb = cv2.imread(image)
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            elif isinstance(image, np.ndarray):
                rgb = image
            else:
                return [{'error': f'Unsupported image type: {type(image)}'}]
            
            # Perform detection
            result = self.detection_api.detect_objects(
                rgb=rgb,
                prompt_text=prompt_text,
                bbox_threshold=bbox_threshold,
                iou_threshold=iou_threshold
            )
            
            # Extract objects from result
            objects = result.get('objects', [])
            
            # Format objects for easier use
            formatted_objects = []
            for obj in objects:
                formatted_obj = {
                    'bbox': obj.get('bbox', []),
                    'category': obj.get('category', 'unknown'),
                    'score': obj.get('score', 0.0),
                    'mask': obj.get('mask', None)
                }
                formatted_objects.append(formatted_obj)
            
            return formatted_objects
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def get_grasp_points(self, image: Union[str, np.ndarray], use_touching_points: bool = True, 
                        bag_mode: bool = True) -> List[Dict]:
        """
        Get grasp points for objects in the image using GraspAnythingAPI
        
        Args:
            image: Image file path or numpy array
            use_touching_points: Whether to use touching points optimization
            bag_mode: Whether to use bag-specific processing
            
        Returns:
            List of grasp information with affordances, masks, touching points, etc.
        """
        if self.grasp_api is None:
            return [{'error': 'GraspAnythingAPI not initialized'}]
        
        try:
            # Handle different image input types
            if isinstance(image, str):
                if image.startswith('http'):
                    # Download image for processing
                    import requests
                    response = requests.get(image)
                    rgb = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                else:
                    # Load local image
                    rgb = cv2.imread(image)
            elif isinstance(image, np.ndarray):
                rgb = image
            else:
                return [{'error': f'Unsupported image type: {type(image)}'}]
            
            # Perform grasp detection
            result, padded_img = self.grasp_api.forward(
                rgb=rgb,
                bag=bag_mode,
                use_touching_points=use_touching_points
            )
            
            # Extract grasp information from result
            grasp_objects = result[0] if result and len(result) > 0 else []
            
            formatted_grasps = []
            for obj in grasp_objects:
                formatted_obj = {
                    'bbox': obj.get('dt_bbox', []),
                    'mask': obj.get('dt_mask', None),
                    'affordances': obj.get('affs', []),
                    'scores': obj.get('scores', []),
                    'touching_points': obj.get('touching_points', [])
                }
                formatted_grasps.append(formatted_obj)
            
            return formatted_grasps
            
        except Exception as e:
            return [{'error': str(e)}]
    
   

def main():
    """Test the Xbrain class"""
    print("=== Testing Enhanced Xbrain (with object detection) ===")
    # Test with DINO APIs
    brain_basic = Xbrain(use_default_configs=True)
    
    text_prompt = "要求将抓取和交换小手提袋，实现从左到右依次排列pink，yellow，blue排列，你打算如何进行。"
    
    try:
        action_list = brain_basic.get_plan("example.jpg", text_prompt)
        print(f"VLM规划生成 {len(action_list)} 个动作步骤")
        
        # Show all actions with full details
        for i, action in enumerate(action_list):
            if action['type'] == 'pick_n_place':
                obj_id = action['params'].get('object_id', 'unknown')
                source = action['params'].get('source_position', None)
                target = action['params'].get('target_position', 'unknown')
                
                # Format source position
                if source and isinstance(source, tuple) and len(source) == 2:
                    source_str = f"({source[0]:.0f}, {source[1]:.0f})"
                else:
                    source_str = "当前位置"
                
                # Format target position
                if isinstance(target, tuple) and len(target) == 2:
                    target_str = f"({target[0]:.0f}, {target[1]:.0f})"
                else:
                    target_str = str(target)
                
                print(f"{i}. pick_n_place: '{obj_id}' 从{source_str} -> 移动到{target_str}")
            elif action['type'] == 'reset_to_home':
                print(f"{i}. reset_to_home: 返回初始位置")
            else:
                # Print full description without truncation
                print(f"{i}. {action['type']}: {action['description']}")
            
    except Exception as e:
        print(f"Basic test failed: {e}")
    
    
    '''
    print("\n" + "="*50)
    print("=== Testing Enhanced Xbrain (with default DINO configs) ===")
    
    # Test with default configs from xdino.py
    try:
        brain_enhanced = Xbrain(use_default_configs=True)
        
        print("\n--- Enhanced Planning with Grasp Detection ---")
        enhanced_actions = brain_enhanced.get_plan("example.jpg", text_prompt)
        print(f"增强规划生成 {len(enhanced_actions)} 个动作步骤")
        
        # Show first few actions to see if grasp info is integrated
        for i, action in enumerate(enhanced_actions[:3], 1):
            print(f"{i}. {action['type']}: {action['description'][:80]}...")
        
        # Test object detection
        print("\n--- Object Detection ---")
        objects = brain_enhanced.get_objects("example.jpg", "bags, objects")
        if objects and 'error' not in objects[0]:
            print(f"检测到 {len(objects)} 个对象")
            for obj in objects[:2]:  # Show first 2 objects
                print(f"  - {obj['category']}: {obj['score']:.2f}")
        else:
            print(f"检测结果: {objects[0] if objects else 'No results'}")
        
        # Test grasp detection
        print("\n--- Grasp Point Detection ---")
        grasps = brain_enhanced.get_grasp_points("example.jpg")
        if grasps and 'error' not in grasps[0]:
            print(f"检测到 {len(grasps)} 个抓取候选")
            for grasp in grasps[:2]:  # Show first 2 grasps
                print(f"  - 抓取点数量: {len(grasp.get('touching_points', []))}")
        else:
            print(f"抓取检测结果: {grasps[0] if grasps else 'No results'}")
        
        # Test combined analysis
        print("\n--- Combined Analysis ---")
        combined = brain_enhanced.get_combined_analysis("example.jpg", text_prompt)
        print(f"分析状态: {combined['status']}")
        print(f"检测到对象: {len(combined['detected_objects'])}")
        print(f"抓取点: {len(combined['grasp_points'])}")
        print(f"动作计划: {len(combined['action_plan'])} 步")
        
    except Exception as e:
        print(f"Enhanced test failed: {e}")
        
    '''



if __name__ == "__main__":
    main()