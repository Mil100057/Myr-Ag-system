"""
Device Manager for MPS/CPU optimization
Manages device selection based on component compatibility
"""

import os
import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class DeviceManager:
    """Intelligent device manager for MPS/CPU optimization."""
    
    def __init__(self):
        self.mps_available = torch.backends.mps.is_available()
        self.mps_built = torch.backends.mps.is_built() if self.mps_available else False
        self.device_cache = {}
        
        # Configure MPS environment
        self._configure_mps_environment()
        
        # Test MPS functionality
        self.mps_working = self._test_mps_functionality()
        
        logger.info(f"Device Manager initialized - MPS available: {self.mps_available}, MPS working: {self.mps_working}")
    
    def _configure_mps_environment(self):
        """Configure MPS environment variables for optimal performance."""
        if self.mps_available:
            # Disable MPS memory limit to prevent out-of-memory errors
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            
            # Enable MPS optimizations
            if hasattr(torch.backends.mps, 'allow_tf32'):
                torch.backends.mps.allow_tf32 = True
            if hasattr(torch.backends.mps, 'allow_fp16'):
                torch.backends.mps.allow_fp16 = True
    
    def _test_mps_functionality(self) -> bool:
        """Test if MPS is actually working with a simple operation."""
        if not self.mps_available:
            return False
        
        try:
            # Test with a small tensor
            test_tensor = torch.tensor([1.0, 2.0, 3.0], device='mps')
            result = test_tensor * 2
            result.cpu()  # Move back to CPU
            torch.mps.empty_cache()  # Clean up
            return True
        except Exception as e:
            logger.warning(f"MPS test failed: {e}")
            return False
    
    def get_device(self, component: str = "general") -> str:
        """
        Get optimal device for a specific component.
        
        Args:
            component: Component type ("embedding", "docling", "general", "cpu_only")
        
        Returns:
            Device string ("mps" or "cpu")
        """
        # Force CPU for certain components
        cpu_only_components = ["docling", "easyocr", "ocr", "layout", "table"]
        
        if component.lower() in cpu_only_components:
            return "cpu"
        
        # Use MPS for embeddings if available and working
        if component.lower() in ["embedding", "sentence_transformer"] and self.mps_working:
            return "mps"
        
        # Use MPS for general PyTorch operations if available and working
        if component.lower() == "general" and self.mps_working:
            return "mps"
        
        # Default to CPU
        return "cpu"
    
    def get_optimal_batch_size(self, component: str = "general", base_batch_size: int = 32) -> int:
        """
        Get optimal batch size based on device and component.
        
        Args:
            component: Component type
            base_batch_size: Base batch size for CPU
        
        Returns:
            Optimized batch size
        """
        device = self.get_device(component)
        
        if device == "mps":
            # Reduce batch size for MPS to prevent memory issues
            if component.lower() in ["embedding", "sentence_transformer"]:
                return min(base_batch_size, 16)
            else:
                return min(base_batch_size, 8)
        
        return base_batch_size
    
    def cleanup_memory(self, device: Optional[str] = None):
        """Clean up memory for specified device."""
        if device == "mps" and self.mps_working:
            try:
                torch.mps.empty_cache()
                torch.mps.synchronize()
            except Exception as e:
                logger.warning(f"Failed to cleanup MPS memory: {e}")
        elif device == "cpu" or device is None:
            # CPU cleanup (if needed)
            pass
    
    def get_mps_info(self) -> Dict[str, Any]:
        """Get detailed MPS information."""
        info = {
            "mps_available": self.mps_available,
            "mps_built": self.mps_built,
            "mps_working": self.mps_working,
            "device_cache": self.device_cache
        }
        
        if self.mps_working:
            try:
                memory_allocated = torch.mps.current_allocated_memory()
                memory_reserved = torch.mps.driver_allocated_memory()
                info.update({
                    "memory_allocated": int(memory_allocated) if memory_allocated is not None else 0,
                    "memory_reserved": int(memory_reserved) if memory_reserved is not None else 0,
                    "max_memory": int(memory_reserved + memory_allocated) if memory_reserved is not None and memory_allocated is not None else 0
                })
            except Exception as e:
                info["memory_error"] = str(e)
        
        return info
    
    def safe_tensor_operation(self, operation_func, *args, device: Optional[str] = None, **kwargs):
        """
        Safely execute tensor operation with fallback to CPU.
        
        Args:
            operation_func: Function to execute
            *args: Arguments for the function
            device: Target device (if None, will be determined automatically)
            **kwargs: Keyword arguments for the function
        
        Returns:
            Result of the operation
        """
        if device is None:
            device = self.get_device("general")
        
        try:
            if device == "mps" and self.mps_working:
                return operation_func(*args, **kwargs)
            else:
                # Force CPU operation
                return operation_func(*args, **kwargs)
        except RuntimeError as e:
            if "MPS" in str(e) and device == "mps":
                logger.warning(f"MPS operation failed, falling back to CPU: {e}")
                # Fallback to CPU
                return operation_func(*args, **kwargs)
            else:
                raise e

# Global device manager instance
device_manager = DeviceManager()

def get_device(component: str = "general") -> str:
    """Convenience function to get device for component."""
    return device_manager.get_device(component)

def get_optimal_batch_size(component: str = "general", base_batch_size: int = 32) -> int:
    """Convenience function to get optimal batch size."""
    return device_manager.get_optimal_batch_size(component, base_batch_size)

def cleanup_memory(device: Optional[str] = None):
    """Convenience function to cleanup memory."""
    device_manager.cleanup_memory(device)

def get_mps_info() -> Dict[str, Any]:
    """Convenience function to get MPS info."""
    return device_manager.get_mps_info()
