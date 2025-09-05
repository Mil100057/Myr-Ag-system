"""
Ollama Client for local LLM integration.
"""
import json
import time
from typing import Dict, Any, List, Optional, Generator
import requests
from loguru import logger

from config.settings import settings


class OllamaClient:
    """Client for interacting with local Ollama server."""
    
    def __init__(self, base_url: str = None, model: str = None):
        """Initialize the Ollama client."""
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.default_model = model or settings.OLLAMA_MODEL
        self.current_model = self.default_model
        self.session = requests.Session()
        
        logger.info(f"Ollama client initialized with base URL: {self.base_url}")
        logger.info(f"Default model: {self.default_model}")
        logger.info(f"Current model: {self.current_model}")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=30)
            if response.status_code == 200:
                logger.info("✅ Ollama server connection successful")
                return True
            else:
                logger.error(f"❌ Ollama server returned status: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ Cannot connect to Ollama server at {self.base_url}")
            logger.error("Please ensure Ollama is running: ollama serve")
            return False
        except Exception as e:
            logger.error(f"❌ Error testing Ollama connection: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models on the Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=30)
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(f"Found {len(models)} models on Ollama server")
                return models
            else:
                logger.error(f"Failed to list models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def check_model_availability(self, model_name: str = None) -> bool:
        """Check if a specific model is available."""
        model_name = model_name or self.current_model
        models = self.list_models()
        
        for model in models:
            if model.get("name") == model_name:
                logger.info(f"✅ Model {model_name} is available")
                return True
        
        logger.warning(f"⚠️ Model {model_name} not found on Ollama server")
        logger.info("Available models:")
        for model in models:
            logger.info(f"  - {model.get('name', 'Unknown')}")
        return False
    
    def pull_model(self, model_name: str = None) -> bool:
        """Pull a model to the Ollama server."""
        model_name = model_name or self.current_model
        
        if self.check_model_availability(model_name):
            logger.info(f"Model {model_name} is already available")
            return True
        
        try:
            logger.info(f"Pulling model {model_name}...")
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Successfully pulled model {model_name}")
                return True
            else:
                logger.error(f"❌ Failed to pull model {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error pulling model {model_name}: {e}")
            return False
    
    def generate_response(self, prompt: str, model: str = None,
                         system_prompt: str = None, temperature: float = 0.7,
                         max_tokens: int = 2048) -> str:
        """Generate a response using the specified model."""
        model = model or self.current_model
        
        # Ensure model is available
        if not self.check_model_availability(model):
            logger.error(f"Model {model} not available")
            return ""
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            logger.info(f"Generating response with model {model}")
            logger.debug(f"Prompt: {prompt[:100]}...")
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                logger.info(f"✅ Generated response ({len(generated_text)} characters)")
                logger.debug(f"Response: {generated_text[:100]}...")
                
                return generated_text
            else:
                logger.error(f"❌ Generation failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return ""
    
    def generate_response_streaming(self, prompt: str, model: str = None,
                                  system_prompt: str = None, temperature: float = 0.7,
                                  max_tokens: int = 2048) -> Generator[str, None, None]:
        """Generate a streaming response using the specified model."""
        model = model or self.current_model
        
        # Ensure model is available
        if not self.check_model_availability(model):
            logger.error(f"Model {model} not available")
            return
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            logger.info(f"Generating streaming response with model {model}")
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=300
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'response' in data:
                                yield data['response']
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                logger.error(f"❌ Streaming generation failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Error in streaming generation: {e}")
    
    def get_model_info(self, model: str = None) -> Dict[str, Any]:
        """Get information about a specific model."""
        model = model or self.current_model
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json={"name": model},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get model info: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check the health of the Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=30)
            return response.status_code == 200
        except Exception:
            return False
    
    def change_model(self, model_name: str) -> bool:
        """Change the current model to a new one."""
        try:
            # Check if the new model is available
            if not self.check_model_availability(model_name):
                logger.error(f"Cannot change to model {model_name}: not available")
                return False
            
            # Update the current model
            old_model = self.current_model
            self.current_model = model_name
            
            logger.info(f"✅ Model changed from {old_model} to {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error changing model to {model_name}: {e}")
            return False
