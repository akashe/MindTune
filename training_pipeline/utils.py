# utils.py - Create utility functions
import torch
import importlib.util

def check_flash_attention_availability():
    """Check if Flash Attention 2 is available and working"""
    
    try:
        # Check if flash_attn package is installed
        flash_attn_spec = importlib.util.find_spec("flash_attn")
        if flash_attn_spec is None:
            return False, "flash_attn package not installed"
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # Check GPU compute capability (Flash Attention 2 requires >= 8.0)
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            compute_capability = major + minor / 10
            
            if compute_capability < 8.0:
                return False, f"GPU compute capability {compute_capability} < 8.0 (required for Flash Attention 2)"
        
        # Try to import flash attention
        try:
            from flash_attn import flash_attn_func
            return True, "Flash Attention 2 available and compatible"
        except ImportError as e:
            return False, f"Flash Attention import failed: {e}"
            
    except Exception as e:
        return False, f"Error checking Flash Attention: {e}"

def get_optimal_attention_implementation():
    """Get the best attention implementation for current setup"""
    
    is_available, reason = check_flash_attention_availability()
    
    if is_available:
        print(f"✅ Using Flash Attention 2: {reason}")
        return "flash_attention_2"
    else:
        print(f"ℹ️  Using eager attention: {reason}")
        return "eager"

# GPU capability check function
def check_gpu_info():
    """Print detailed GPU information"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    compute_capability = major + minor / 10
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🖥️  GPU: {gpu_name}")
    print(f"🔢 Compute Capability: {compute_capability}")
    print(f"💾 Memory: {memory_gb:.1f} GB")
    
    # Flash Attention compatibility
    if compute_capability >= 8.0:
        print("✅ Flash Attention 2 compatible")
    else:
        print(f"❌ Flash Attention 2 not compatible (needs >= 8.0, got {compute_capability})")