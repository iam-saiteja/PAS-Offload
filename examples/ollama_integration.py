import os
import sys
import time
import json
import requests
import torch
from pynvml import *

# Add root folder to python path to import pas_offload
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pas_offload.engine import PASOffloadEngine

# Initialize NVML for hardware tracking
try:
    nvmlInit()
    nvml_handle = nvmlDeviceGetHandleByIndex(0)
    has_nvml = True
except Exception:
    has_nvml = False

def run_hybrid_inference():
    # 1. Connect to local Ollama
    url = "http://localhost:11434/api/generate"
    model_name = "qwen2.5:0.5b"
    
    print("==========================================")
    print("Ollama + PAS-Offload Hybrid Example")
    print("==========================================")
    
    try:
        r = requests.get("http://localhost:11434/api/tags")
        r.raise_for_status()
    except Exception as e:
        print(f"Error: Could not connect to Ollama server. Make sure it is running: {e}")
        return
        
    # 2. Initialize the PAS-Offload Engine for an offloaded 7B layer
    # Standard Llama FFN layer dimensions: 4096 -> 11008
    in_features = 4096
    out_features = 11008
    
    print(f"Initializing PAS-Offload Engine for a 7B FFN Layer ({in_features}x{out_features})...")
    engine = PASOffloadEngine(in_features, out_features, rank=16)
    
    # Load random weights to represent the offloaded FFN layer
    mock_weights = torch.randn(out_features, in_features, dtype=torch.float16)
    engine.load_weights(mock_weights)
    
    # 3. Define prompt and payload
    prompt = "Explain in one sentence why memory bandwidth limits supercomputers."
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": 2048
        }
    }
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Streaming tokens and executing offloaded weight-streaming on GPU...")
    print("-" * 80)
    print(f"{'Token':<18} | {'Active Cols':<12} | {'Slicing + Tx':<15} | {'GPU VRAM':<10} | {'GPU Power':<10}")
    print("-" * 80)
    
    # Hidden state simulation tensor on GPU
    mock_hidden_state = torch.randn(1, in_features, device='cuda', dtype=torch.float16)
    
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        token_times = []
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                token = chunk.get("response", "")
                
                # Clean up token rendering for the table
                token_clean = token.replace("\n", "\\n").replace("\t", "\\t")
                if len(token_clean) > 15:
                    token_clean = token_clean[:12] + "..."
                if not token_clean.strip():
                    token_clean = repr(token)
                
                # Execute PAS-Offload streaming forward pass for this token
                start_t = time.perf_counter()
                out, indices = engine.forward(mock_hidden_state, threshold=0.15)
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start_t) * 1000
                token_times.append(elapsed_ms)
                
                # Read NVML stats if available
                vram_str = "N/A"
                power_str = "N/A"
                if has_nvml:
                    try:
                        mem_info = nvmlDeviceGetMemoryInfo(nvml_handle)
                        vram_str = f"{mem_info.used / (1024 * 1024):.0f} MB"
                        power_str = f"{nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0:.1f} W"
                    except Exception:
                        pass
                
                print(f"{token_clean:<18} | {len(indices):<12} | {elapsed_ms:8.3f} ms | {vram_str:<10} | {power_str:<10}")
                
                if chunk.get("done", False):
                    break
                    
        print("-" * 80)
        print("Inference Complete.")
        print(f"Average PAS-Offload latency per token: {sum(token_times)/len(token_times):.3f} ms")
        
    except Exception as e:
        print(f"Error during generation: {e}")
    finally:
        if has_nvml:
            nvmlShutdown()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA must be available to run this hybrid GPU example.")
        sys.exit(1)
    run_hybrid_inference()
