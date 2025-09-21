import os
import base64
import requests
from typing import List, Dict, Union, Optional
from abc import ABC, abstractmethod
from openai import OpenAI


class VLMPlatform(ABC):
    """Abstract base class for VLM platforms"""
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> str:
        """Send chat completion request to the platform"""
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Get list of supported models for this platform"""
        pass


class DashscopePlatform(VLMPlatform):
    """Dashscope platform implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                api_key = "sk-e622f19390f646dab2a083ee16deb64c"
        
        if not api_key:
            raise ValueError("Dashscope API key not available")
        
        self.xbrain = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        self.supported_models = [
            # VLM models
            "qwen-vl-max-latest",
            "qwen2.5-vl-72b-instruct",
            'qwen2.5-vl-32b-instruct',
            "qwen-vl-max",
            "qwen-vl-plus",
            'qwen2.5-vl-3b-instruct',
             "qwen2.5-vl-7b-instruct",
            # Text-only LLM models
            "qwen3-4b",
            "qwen3-8b",
            "qwen3-0.6b",
            "qwen-max",
            "qwen-turbo",
        ]
    
    def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> str:
        if model not in self.supported_models:
            raise ValueError(f"Model {model} not supported by Dashscope. Supported models: {self.supported_models}")
        
        completion = self.xbrain.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        return completion.choices[0].message.content
    
    def get_supported_models(self) -> List[str]:
        return self.supported_models


class SiliconFlowPlatform(VLMPlatform):
    """SiliconFlow platform implementation"""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("SILICONFLOW_API_KEY", "sk-mwbqpbrgcpsraguhphsjddkftuqmkhvpwuxwzsvyucbqggvo")
        
        if not api_key:
            raise ValueError("SiliconFlow API key not available")
        
        self.api_key = api_key
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        
        self.supported_models = [
            "zai-org/GLM-4.5V",
            "THUDM/GLM-4.1V-9B-Thinking",
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "Pro/THUDM/GLM-4.1V-9B-Thinking",
        ]
    
    def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> str:
        if model not in self.supported_models:
            raise ValueError(f"Model {model} not supported by SiliconFlow. Supported models: {self.supported_models}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": self._format_messages(messages),
            "stream": False,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.7),
            "top_k": kwargs.get("top_k", 50),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "n": kwargs.get("n", 1),
            "stop": kwargs.get("stop", [])
        }
        
        if model in ["THUDM/GLM-4.1V-9B-Thinking", "Pro/THUDM/GLM-4.1V-9B-Thinking"]:
            payload["thinking_budget"] = kwargs.get("thinking_budget", 4096)
        
        response = requests.post(self.base_url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _format_messages(self, messages: List[Dict]) -> List[Dict]:
        """Format messages for SiliconFlow API"""
        formatted_messages = []
        
        for msg in messages:
            formatted_msg = {"role": msg["role"]}
            
            if isinstance(msg["content"], str):
                formatted_msg["content"] = msg["content"]
            elif isinstance(msg["content"], list):
                content_parts = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        content_parts.append({
                            "type": "text",
                            "text": part["text"]
                        })
                    elif part["type"] == "image_url":
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": part["image_url"]["url"],
                                "detail": part["image_url"].get("detail", "auto")
                            }
                        })
                formatted_msg["content"] = content_parts
            
            formatted_messages.append(formatted_msg)
        
        return formatted_messages
    
    def get_supported_models(self) -> List[str]:
        return self.supported_models


class Xbrain:
    """Unified VLM brain supporting multiple platforms"""
    
    def __init__(self, platform: str = "dashscope", api_key: Optional[str] = None):
        """
        Initialize VLM brain with specified platform
        
        Args:
            platform: Platform name ("dashscope" or "siliconflow")
            api_key: API key for the platform
        """
        self.platform_name = platform.lower()
        
        if self.platform_name == "dashscope":
            self.platform = DashscopePlatform(api_key)
        elif self.platform_name == "siliconflow":
            self.platform = SiliconFlowPlatform(api_key)
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode local image to base64 data URL, with optional resizing to reduce size"""
        try:
            # Try to open and potentially resize the image
            try:
                from PIL import Image
                # Open image
                img = Image.open(image_path)
                
                # Convert RGBA to RGB if necessary (JPEG doesn't support alpha channel)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create a white background image
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    # Paste the image on the background using the alpha channel as mask
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                # Save to bytes
                import io
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            except ImportError:
                # PIL not available, read file directly
                with open(image_path, 'rb') as image_file:
                    encoded = base64.b64encode(image_file.read()).decode('utf-8')
            
            return f"data:image/jpeg;base64,{encoded}"
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
    
    def chat(self, 
             messages: List[Dict], 
             model: Optional[str] = None,
             **kwargs) -> str:
        """
        Send chat completion request.
        
        Note: For image inputs, use chat_with_image() method instead.
        
        Args:
            messages: List of message dictionaries. For text-only messages, 
                     each message should have the format:
                     {"role": "user/system/assistant", "content": "text content"}
            model: Model name (if None, uses default for platform)
            **kwargs: Additional parameters for the platform
            
        Returns:
            Response text from the model
            
        Raises:
            ValueError: If messages contain complex formats not supported by this method
        """
        # Validate message format for text-only content
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Check if this is a complex message format (with images)
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        raise ValueError(
                            "Complex message formats with images are not supported by chat(). "
                            "Use chat_with_image() method instead for image inputs."
                        )
        
        if model is None:
            model = self.get_default_model()
        
        return self.platform.chat_completion(messages, model, **kwargs)
    
    def _chat_internal(self, messages: List[Dict], model: Optional[str] = None, **kwargs) -> str:
        """
        Internal chat method that bypasses validation for use by other methods in this class.
        
        Args:
            messages: List of message dictionaries
            model: Model name (if None, uses default for platform)
            **kwargs: Additional parameters for the platform
            
        Returns:
            Response text from the model
        """
        if model is None:
            model = self.get_default_model()
        
        return self.platform.chat_completion(messages, model, **kwargs)
    
    def chat_with_image(self,
                       image: Union[str, bytes],
                       text: str,
                       system_prompt: str = "",
                       model: Optional[str] = None,
                       **kwargs) -> str:
        """
        Convenient method for chatting with image input
        
        Args:
            image: Image file path or bytes
            text: User prompt text
            system_prompt: System prompt
            model: Model name
            max_image_size: Maximum size for image (width, height) - images will be resized to fit
            **kwargs: Additional parameters
            
        Returns:
            Response text from the model
        """
        if isinstance(image, str):
            if image.startswith('http'):
                image_url = image
            else:
                image_url = self.encode_image_to_base64(image)
        else:
            # For bytes, we can't resize, so we'll encode directly
            encoded = base64.b64encode(image).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{encoded}"
        
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                },
                {"type": "text", "text": text}
            ]
        })
        
        return self._chat_internal(messages, model, **kwargs)
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models for current platform"""
        return self.platform.get_supported_models()
    
    def get_default_model(self) -> str:
        """Get default model for current platform"""
        models = self.platform.get_supported_models()
        return models[0] if models else None
    
    def switch_platform(self, platform: str, api_key: Optional[str] = None):
        """Switch to a different platform"""
        self.platform_name = platform.lower()
        
        if self.platform_name == "dashscope":
            self.platform = DashscopePlatform(api_key)
        elif self.platform_name == "siliconflow":
            self.platform = SiliconFlowPlatform(api_key)
        else:
            raise ValueError(f"Unsupported platform: {platform}")


def test_xbrain():
    """Test function for VLM xbrain"""
    
    print("Testing VLM xbrain...")
    
    # Test Dashscope
    print("\n1. Testing Dashscope platform:")
    xbrain = Xbrain(platform="dashscope")
    print(f"   Supported models: {xbrain.get_supported_models()}")
    print(f"   Default model: {xbrain.get_default_model()}")
    
    # Test SiliconFlow
    print("\n2. Testing SiliconFlow platform:")
    xbrain.switch_platform("siliconflow")
    print(f"   Supported models: {xbrain.get_supported_models()}")
    print(f"   Default model: {xbrain.get_default_model()}")
    
    # Example usage with image
    print("\n3. Example usage:")
   
    xbrain = Xbrain(platform="dashscope")
    response = xbrain.chat_with_image(
        image="example.png",
        text="What objects do you see in this image?",
        model="qwen2.5-vl-72b-instruct"
    )
    print("Response from Dashscope:", response)
    
    # Using SiliconFlow with thinking model
    xbrain = Xbrain(platform="siliconflow")
    response = xbrain.chat_with_image(
        image="example.png",
        text="Analyze this image step by step",
        model="zai-org/GLM-4.5V",
        thinking_budget=4096
    )
    print("Response from SiliconFlow:", response)

if __name__ == "__main__":
    test_xbrain()