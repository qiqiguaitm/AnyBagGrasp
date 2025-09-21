import os
import base64
from typing import List, Dict, Union, Optional
import json
import re
import cv2
import numpy as np
from mmengine.config import Config


try:
    from .xbrain import Xbrain
except ImportError:
    from xbrain import Xbrain

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


class XAgent:
    """Robot Agent Brain for task planning and control"""
    
    def __init__(self, vlm_platform: str = "dashscope", vlm_api_key: str = None, detection_config: Dict = None, grasp_config: Dict = None, 
                 use_default_configs: bool = True, swap_tmp_area: tuple = (0, 0),
                 ):
        """
        Initialize XAgent with API key and DINO configs
        
        Args:
            vlm_platform: VLM platform to use ("dashscope" or "siliconflow")
            vlm_api_key: API key for VLM platform (uses api_key if not provided)
            detection_config: Configuration for DetectionAPI
            grasp_config: Configuration for GraspAnythingAPI
            use_default_configs: Whether to use default configs from xdino.py
            swap_tmp_area: Temporary area coordinates (x, y) for object swapping, default (0, 0)
            
            
        """
        
        if vlm_api_key is None:
            if vlm_platform == "dashscope":
                vlm_api_key = os.getenv("DASHSCOPE_API_KEY", "sk-e622f19390f646dab2a083ee16deb64c")
            if vlm_platform == "siliconflow":
                vlm_api_key = os.getenv("SILICONFLOW_API_KEY", "sk-mwbqpbrgcpsraguhphsjddkftuqmkhvpwuxwzsvyucbqggvo")
        if vlm_api_key is None:
            raise ValueError("API key not available")
        
        self.xbrain = Xbrain(platform=vlm_platform, api_key=vlm_api_key)
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
        return self.xbrain.encode_image_to_base64(image_path)
    
    def _parse_action_list(self, response_text: str) -> List[Dict]:
        """
        Simple parser for VLM action list
        """
        import re
        actions = []
        
        # Find action sequence section
        match = re.search(r'\*\*动作序列：\*\*(.*?)(?:\*\*|$)', response_text, re.DOTALL)
        action_text = match.group(1) if match else response_text
        
        # Find all pick_n_place calls
        for line in action_text.split('\n'):
            if 'pick_n_place' not in line:
                continue
                
            # Extract parameters using simple regex
            params = {}
            
            # object_id
            m = re.search(r'object_id\s*=\s*["\']([^"\']+)["\']', line)
            if m:
                params['object_id'] = m.group(1)
            
            # source_position
            if 'source_position=swap_tmp_area' in line:
                params['source_position'] = 'swap_tmp_area'
            else:
                m = re.search(r'source_position\s*=\s*\(([^)]+)\)', line)
                if m:
                    try:
                        params['source_position'] = tuple(map(float, m.group(1).split(',')))
                    except:
                        pass
            
            # target_position  
            if 'target_position=swap_tmp_area' in line:
                params['target_position'] = 'swap_tmp_area'
            else:
                m = re.search(r'target_position\s*=\s*\(([^)]+)\)', line)
                if m:
                    try:
                        params['target_position'] = tuple(map(float, m.group(1).split(',')))
                    except:
                        pass
            
            # Add action if valid
            if params.get('object_id') and params.get('target_position'):
                actions.append({
                    'type': 'pick_n_place',
                    'params': params
                })
        
        # Add reset_to_home after each pick_n_place
        final_actions = []
        for action in actions:
            final_actions.append(action)
            final_actions.append({
                'type': 'reset_to_home',
                'params': {}
            })
        
        return final_actions
    
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
        
        # Try multiple patterns to match different response formats
        patterns = [
            # Pattern 1: **物体1: Name**
            (r'\*\*物体\d+:\s*([^*\n]+?)\*\*\s*(.*?)(?=\n\*\*物体\d+:|\n###|$)', True),
            # Pattern 2: **物体ID: xxx** or **物体ID：xxx**
            (r'\*\*物体ID[：:]\s*([^*\n]+?)\*\*\s*(.*?)(?=\n\*\*物体ID|\*\*空间分析|$)', True),
            # Pattern 3: 物体ID: xxx (without asterisks)
            (r'(?:^|\n)物体ID[：:]\s*([^\n]+?)\n(.*?)(?=\n物体ID[：:]|\n\*\*空间分析|$)', True),
            # Pattern 4: - 物体ID: xxx (with dash)
            (r'-\s*物体ID[：:]\s*([^\n]+?)\n(.*?)(?=\n-\s*物体ID[：:]|\n\*\*空间分析|$)', True),
            # Pattern 5: 对象1: xxx format
            (r'(?:^|\n)(对象\d+)[：:]\s*([^\n]*?)(?=\n对象\d+[：:]|\n\*\*|$)', False)
        ]
        
        for pattern, has_content in patterns:
            matches = re.findall(pattern, response_text, re.DOTALL | re.MULTILINE)
            if matches:
                print(f"Found {len(matches)} matches with pattern {patterns.index((pattern, has_content)) + 1}")
                break
        
        # Process matches based on pattern type
        if matches and has_content:
            for object_id, content in matches:
                # Clean up object_id
                object_id = object_id.strip().rstrip('*:：').strip()
                obj_info = {}
            
            # Parse each attribute - handle both [] and non-bracketed formats
            attributes = {
                '类别': [r'-\s*类别:\s*(.+?)(?=\n|$)', r'类别:\s*(.+?)(?=\n|$)'],
                '颜色': [r'-\s*颜色:\s*(.+?)(?=\n|$)', r'颜色:\s*(.+?)(?=\n|$)'],
                '大小': [r'-\s*大小:\s*(.+?)(?=\n|$)', r'大小:\s*(.+?)(?=\n|$)'],
                '位置': [r'-\s*位置:\s*(.+?)(?=\n|$)', r'位置:\s*(.+?)(?=\n|$)'],

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
        elif matches and not has_content:
            # Handle simple object format (对象1: 类别(xxx)...)
            for obj_id, description in matches:
                obj_info = {}
                # Parse inline description
                if '类别(' in description:
                    category_match = re.search(r'类别\(([^)]+)\)', description)
                    if category_match:
                        obj_info['类别'] = category_match.group(1)
                if '位置(' in description:
                    pos_match = re.search(r'位置\((\d+),(\d+)\)', description)
                    if pos_match:
                        obj_info['位置'] = f"({pos_match.group(1)},{pos_match.group(2)})"
                if obj_info:
                    objects[obj_id] = obj_info
        
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
        
        # Look for spatial analysis sections - handle various formats including the **Title** format
        sections = {
            '整体布局': [
                r'\*\*整体布局\*\*\s*\n-\s*([^\n]+(?:\n(?![\*#])[^\n]+)*)',
                r'整体布局[：:]\s*\[([^\]]+)\]', 
                r'整体布局[：:]\s*([^*\n]+(?:\n[^*\n-]+)*)', 
                r'-\s*整体布局[：:]\s*([^\n]+)'
            ],
            '左右顺序': [
                r'\*\*左右顺序\*\*\s*\n-\s*([^\n]+(?:\n(?![\*#])[^\n]+)*)',
                r'左右顺序[：:]\s*\[([^\]]+)\]', 
                r'左右顺序[：:]\s*((?:[^*\n]+(?:\n\s*-[^\n]+)*)+)', 
                r'-\s*左右顺序[：:]\s*([^\n]+)'
            ],
            '前后关系': [
                r'\*\*前后关系\*\*\s*\n-\s*([^\n]+(?:\n(?![\*#])[^\n]+)*)',
                r'前后关系[：:]\s*\[([^\]]+)\]', 
                r'前后关系[：:]\s*([^*\n]+(?:\n[^*\n-]+)*)', 
                r'-\s*前后关系[：:]\s*([^\n]+)'
            ],
            '相对距离': [
                r'\*\*相对距离\*\*\s*\n-\s*([^\n]+(?:\n-[^\n]+)*)',
                r'相对距离[：:]\s*\[([^\]]+)\]', 
                r'相对距离[：:]\s*([^*\n]+(?:\n[^*\n-]+)*)', 
                r'-\s*相对距离[：:]\s*([^\n]+)'
            ],
            '群组关系': [
                r'\*\*群组关系\*\*\s*\n-\s*([^\n]+(?:\n-[^\n]+)*)',
                r'群组关系[：:]\s*\[([^\]]+)\]', 
                r'群组关系[：:]\s*([^*\n]+(?:\n[^*\n-]+)*)', 
                r'-\s*群组关系[：:]\s*([^\n]+)'
            ],
            '对称性': [
                r'\*\*对称性\*\*\s*\n-\s*([^\n]+(?:\n-[^\n]+)*)',
                r'对称性[：:]\s*\[([^\]]+)\]', 
                r'对称性[：:]\s*([^*\n]+(?:\n[^*\n-]+)*)', 
                r'-\s*对称性[：:]\s*([^\n]+)'
            ]
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

    def get_plan(self, image_path: str, text: str, model: Optional[str] = None) -> List[Dict]:
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
            print("=== Starting get_plan ===")
            # Get object detection information
            print("Getting object detection...")
            objects = self.get_objects(image_path)
            print(f"Objects detected: {len(objects)}")
            #objects = self.get_grasp_points(image)
            #print(f"DEBUG: Detection API initialized: {self.detection_api is not None}")
            #print(f"DEBUG: Objects result: {objects[:2] if objects else 'None'}")
            
            # Enhance task description with object information
            object_info_text = self._format_object_info(objects)
            #object_info_text = self._format_object_info(objects)
            enhanced_text = f"【重要】以下是检测系统提供的精确物体信息，请优先使用这些数据：\n{object_info_text}\n\n任务要求：{text}"
            
            # Simplified system prompt to avoid timeout issues
            system_prompt = f"""你是智能机器人Agent大脑，具备视觉理解、空间推理和精确操作能力。

【重要指示】
1. 系统已提供精确的物体检测信息，包括准确的位置坐标、边界框和置信度
2. 在生成动作序列时，必须优先使用检测系统提供的坐标，而非视觉估计
3. 确保所有坐标值直接引用检测数据，提高操作精度

请按照以下结构化格式输出：

**物体分析：**
根据检测信息分析每个物体，使用格式：物体ID: [颜色+大小+类别+序号]
- 类别: [具体类别]
- 位置: [使用检测坐标]

**空间分析：**
描述物体左右顺序和布局

**动作规划：**
使用pick_n_place(object_id, source_position, target_position)生成动作序列
- object_id：使用分析的物体ID
- source_position：物体当前位置坐标
- target_position：目标位置坐标或swap_tmp_area

**位置分配原则：**
- 优先使用现有物体位置或swap_tmp_area进行位置交换
- 目标位置必须从已有的source_position或swap_tmp_area中选择

示例动作序列：
1. pick_n_place(object_id="粉色手提袋1", source_position=(432,308), target_position=swap_tmp_area)
2. pick_n_place(object_id="黄色手提袋2", source_position=(710,300), target_position=(432,308))
3. pick_n_place(object_id="蓝色手提袋3", source_position=(162,309), target_position=(710,300))
4. pick_n_place(object_id="粉色手提袋1", source_position=swap_tmp_area, target_position=(162,309))

请基于检测信息生成简洁的动作序列。"""



            print("=== VLM Prompt ===")
            print("System Prompt:\n", system_prompt)
            print("User Prompt:\n", enhanced_text)
            if model:
                print(f"Model: {model}")
            
            # Use VLM client for chat completion with image size limit to prevent "File name too long" errors
            try:
                response_text = self.xbrain.chat_with_image(
                    image=image_path,
                    text=enhanced_text,
                    system_prompt=system_prompt,
                    model=model,
                )
            except Exception as e:
                ret = [{
                    'type': 'error', 
                    'description': f'VLM请求失败: {str(e)}',
                    'params': {'error': str(e)}
                }]
                print(ret)
                return ret
            
            print("=== VLM Response ===")
            print(response_text)
            
            # 1. Parse and print object analysis independently
            print("\n=== 物体分析 Object Analysis ===")
            object_analysis = self._parse_object_analysis(response_text)
            if object_analysis:
                print(f"检测到 {len(object_analysis)} 个物体:")
                for obj_id, details in object_analysis.items():
                    print(f"  ► {obj_id}:")
                    if details:
                        for attr, value in details.items():
                            print(f"      {attr}: {value}")
                    print()
            else:
                print("未解析到物体信息")
            
            # 2. Parse and print spatial analysis independently  
            print("\n=== 空间分析 Spatial Analysis ===")
            spatial_analysis = self._parse_spatial_analysis(response_text)
            if spatial_analysis:
                print(f"解析到 {len(spatial_analysis)} 个空间关系:")
                for section, content in spatial_analysis.items():
                    print(f"  ► {section}: {content}")
            else:
                print("未解析到空间关系")
            
            # 3. Parse and print action list independently
            print("\n=== 动作序列 Action List ===")
            action_list = self._parse_action_list(response_text)
            print(f"生成 {len(action_list)} 个动作:")
            for i, action in enumerate(action_list, 1):
                if action['type'] == 'pick_n_place':
                    params = action['params']
                    obj_id = params.get('object_id', 'unknown')
                    source = params.get('source_position', 'unknown')
                    target = params.get('target_position', 'unknown')
                    
                    # Format positions
                    if isinstance(source, tuple):
                        source_str = f"({source[0]:.0f},{source[1]:.0f})"
                    else:
                        source_str = str(source)
                    
                    if isinstance(target, tuple):
                        target_str = f"({target[0]:.0f},{target[1]:.0f})"
                    else:
                        target_str = str(target)
                    
                    print(f"  {i}. pick_n_place: '{obj_id}' 从{source_str} -> 移动到{target_str}")
                elif action['type'] == 'reset_to_home':
                    print(f"  {i}. reset_to_home: 返回初始位置")
                else:
                    print(f"  {i}. {action['type']}")
            
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
    
    def describe_objects(self, image_path: str, model: Optional[str] = None) -> Dict[str, str]:
        """
        Describe objects in an image with detailed object and spatial analysis
        
        Args:
            image_path: Path to the image file
            model: Optional VLM model to use
            
        Returns:
            Dictionary containing:
                - 'object_info_text': Updated detailed object information
                - 'spatial_info_text': Spatial relationship analysis
        """
        # Step 1: Get objects using detection API
        objects = self.get_objects(image_path)
        
        if not objects or 'error' in objects[0]:
            return {
                'object_info_text': "未检测到物体",
                'spatial_info_text': "无空间信息"
            }
        
        # Format initial object detection results
        object_info_text = self._format_object_info(objects)
        
        # Step 2: Create prompt for VLM to analyze objects and spatial relationships
        analysis_prompt = f"""请详细分析图像中的物体和空间关系。

已检测到的物体信息：
{object_info_text}

针对已经检测到的物体，进行以下分析：

**物体分析：**
对每个检测到的物体进行详细描述，包括：
- 物体ID：[ID编号+使用有意义的可区分的中文名称，如"粉色手提袋"]
- 类别：[具体类别]
- 颜色：[主要颜色和辅助颜色]
- 大小：[相对大小和像素尺寸，使用已经提供的检测到的精确尺寸]
- 位置：[使用已经提供的检测到的精确坐标]
- 材质：[如果可见]

**空间分析：**
- 整体布局：[描述物体在场景中的总体分布]
- 左右顺序：[从左到右列出物体]
- 前后关系：[描述深度层次]
"""

        try:
            # Step 3: Call VLM for detailed analysis
            print("\n=== Calling VLM for analysis ===")
            response_text = self.xbrain.chat_with_image(
                image=image_path,
                text=analysis_prompt,
                system_prompt="你是一个专业的视觉分析系统，能够精确识别和描述物体及其空间关系。请严格按照要求的格式输出。",
                model=model
            )
            
            print("\n=== VLM Raw Response ===")
            print(response_text)
            
            # Step 4: Parse the VLM response
            object_analysis = self._parse_object_analysis(response_text)
            spatial_analysis = self._parse_spatial_analysis(response_text)
            
            print(f"\nParsed {len(object_analysis) if object_analysis else 0} objects")
            print(f"Parsed {len(spatial_analysis) if spatial_analysis else 0} spatial sections")
            
            # Step 5: Format the results
            # Enhanced object information
            enhanced_object_info = []
            enhanced_object_info.append("=== 物体详细信息 ===\n")
            
            if object_analysis:
                for obj_id, details in object_analysis.items():
                    enhanced_object_info.append(f"【{obj_id}】")
                    for attr, value in details.items():
                        enhanced_object_info.append(f"  {attr}: {value}")
                    enhanced_object_info.append("")
            else:
                # If parsing failed but we have VLM response, extract the object analysis section
                if "物体分析" in response_text or "**物体" in response_text:
                    # Extract the object analysis section from raw response
                    import re
                    obj_section = re.search(r'###?\s*物体分析(.*?)(?=###?\s*空间分析|$)', response_text, re.DOTALL)
                    if obj_section:
                        enhanced_object_info.append("VLM分析结果：\n")
                        enhanced_object_info.append(obj_section.group(1).strip())
                    else:
                        enhanced_object_info.append(object_info_text)
                else:
                    enhanced_object_info.append(object_info_text)
            
            # Spatial information
            spatial_info_parts = []
            spatial_info_parts.append("=== 空间关系分析 ===\n")
            
            if spatial_analysis:
                for section, content in spatial_analysis.items():
                    spatial_info_parts.append(f"【{section}】")
                    spatial_info_parts.append(f"  {content}")
                    spatial_info_parts.append("")
            else:
                # If parsing failed but we have VLM response, extract the spatial analysis section
                if "空间分析" in response_text:
                    import re
                    spatial_section = re.search(r'###?\s*空间分析(.*?)(?=$)', response_text, re.DOTALL)
                    if spatial_section:
                        spatial_info_parts.append("VLM分析结果：\n")
                        spatial_info_parts.append(spatial_section.group(1).strip())
                    else:
                        spatial_info_parts.append("未能解析空间关系信息")
                else:
                    spatial_info_parts.append("未能解析空间关系信息")
            
            return {
                'object_info_text': "\n".join(enhanced_object_info),
                'spatial_info_text': "\n".join(spatial_info_parts)
            }
            
        except Exception as e:
            print(f"VLM analysis error: {e}")
            # Return basic detection results if VLM fails
            return {
                'object_info_text': f"基础检测信息：\n{object_info_text}",
                'spatial_info_text': "VLM分析失败，无法提供空间关系信息"
            }
    
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

if __name__ == "__main__":
    xagent = XAgent(use_default_configs=True)
    
    # Test with an example image
    test_image = "example.jpg"  # or "example_s.jpg"
    
    # Test with Dashscope
    #result = xagent.describe_objects(test_image, model="qwen-vl-plus")
    #result = xagent.describe_objects(test_image, model="qwen2.5-vl-72b-instruct")
    result = xagent.describe_objects(test_image, model="qwen2.5-vl-3b-instruct")
    print("\nObject Information:")
    print(result['object_info_text'])
    print("\nSpatial Information:")
    print(result['spatial_info_text'])
    
    
    '''
    # Test with SiliconFlow
    print("\n2. Testing with SiliconFlow platform:")
    xagent.switch_vlm_platform("siliconflow")
    result_sf = xagent.describe_objects(test_image, model="zai-org/GLM-4.5V")
    print("\nObject Information (SiliconFlow):")
    print(result_sf['object_info_text'])
    print("\nSpatial Information (SiliconFlow):")
    print(result_sf['spatial_info_text'])
    '''
    # Original test code (commented out)
    # text_prompt = "要求交换和调整纯色手提袋位置，实现从左到右依次排列pink，yellow，blue排列，你打算如何进行。"
    # action_list = xagent.get_plan("example.jpg", text_prompt, model="qwen-vl-plus")
   
    
