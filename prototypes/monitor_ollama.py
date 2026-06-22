# /// script
# dependencies = [
#   "requests",
#   "psutil",
#   "nvidia-ml-py",
#   "matplotlib",
#   "numpy",
# ]
# ///

import os
import sys
import time
import json
import threading
import requests
import psutil
import numpy as np
import matplotlib.pyplot as plt
from pynvml import *

# Initialize NVML
try:
    nvmlInit()
    nvml_handle = nvmlDeviceGetHandleByIndex(0)
    device_name = nvmlDeviceGetName(nvml_handle)
    if isinstance(device_name, bytes):
        device_name = device_name.decode('utf-8')
    has_gpu_monitoring = True
except Exception as e:
    print(f"Warning: NVML initialization failed (GPU monitoring disabled): {e}")
    has_gpu_monitoring = False

# Global metrics storage
monitoring = True
time_stamps = []
cpu_util = []
ram_usage = []
gpu_util = []
gpu_vram = []
gpu_power = []

def monitor_system_stats():
    global monitoring
    start_t = time.perf_counter()
    
    while monitoring:
        current_t = time.perf_counter() - start_t
        time_stamps.append(current_t)
        
        # CPU & RAM (System)
        cpu_util.append(psutil.cpu_percent())
        ram_usage.append(psutil.virtual_memory().percent)
        
        # GPU stats via NVML
        if has_gpu_monitoring:
            try:
                util = nvmlDeviceGetUtilizationRates(nvml_handle)
                gpu_util.append(util.gpu)
                
                mem_info = nvmlDeviceGetMemoryInfo(nvml_handle)
                gpu_vram.append((mem_info.used / mem_info.total) * 100)
                
                power = nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
                gpu_power.append(power)
            except Exception:
                gpu_util.append(0)
                gpu_vram.append(0)
                gpu_power.append(0)
        else:
            gpu_util.append(0)
            gpu_vram.append(0)
            gpu_power.append(0)
            
        time.sleep(0.05)  # Sample every 50ms

def run_ollama_request():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:0.5b",
        "prompt": "Write a highly detailed, 4-paragraph history of the development of high-performance computing, focusing on supercomputers and memory bandwidth bottlenecks.",
        "stream": True
    }
    
    print(f"\nSending generation request to Ollama (model: qwen2.5:0.5b)...")
    
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                
                # Capture time to first token
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    ttft_ms = (first_token_time - start_time) * 1000
                    print(f"Time to First Token (TTFT): {ttft_ms:.2f} ms")
                
                token_count += 1
                sys.stdout.write(chunk.get("response", ""))
                sys.stdout.flush()
                
                if chunk.get("done", False):
                    break
                    
        end_time = time.perf_counter()
        total_time = end_time - start_time
        decode_time = end_time - first_token_time if first_token_time else total_time
        
        print("\n\n------------------------------------------")
        print("Ollama Inference Results")
        print("------------------------------------------")
        print(f"Total Tokens Generated: {token_count}")
        print(f"Total Time:             {total_time:.2f} s")
        print(f"Overall Throughput:     {token_count / total_time:.2f} tokens/s")
        print(f"Decode Throughput:      {token_count / decode_time:.2f} tokens/s")
        
    except Exception as e:
        print(f"\nError contacting Ollama API: {e}")

def main():
    global monitoring
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_system_stats)
    monitor_thread.start()
    
    # Run Ollama inference
    run_ollama_request()
    
    # Stop monitoring
    monitoring = False
    monitor_thread.join()
    
    # NVML Shutdown
    if has_gpu_monitoring:
        nvmlShutdown()
        
    # Analyze and Print Hardware Trends
    print("\n------------------------------------------")
    print("CPU & GPU Hardware Trends During Inference")
    print("------------------------------------------")
    if time_stamps:
        print(f"CPU Utilization:   Mean: {np.mean(cpu_util):.1f}% | Max: {np.max(cpu_util):.1f}%")
        print(f"System RAM Usage:  Mean: {np.mean(ram_usage):.1f}% | Max: {np.max(ram_usage):.1f}%")
        if has_gpu_monitoring:
            print(f"GPU Name:          {device_name}")
            print(f"GPU Utilization:   Mean: {np.mean(gpu_util):.1f}% | Max: {np.max(gpu_util):.1f}%")
            print(f"GPU VRAM Usage:    Mean: {np.mean(gpu_vram):.1f}% | Max: {np.max(gpu_vram):.1f}%")
            print(f"GPU Power Draw:    Mean: {np.mean(gpu_power):.1f} W | Max: {np.max(gpu_power):.1f} W")
            
        # Plot and save trends
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot CPU/RAM
        ax1.plot(time_stamps, cpu_util, label="CPU Util (%)", color="dodgerblue", alpha=0.8)
        ax1.plot(time_stamps, ram_usage, label="System RAM (%)", color="navy", alpha=0.8)
        ax1.set_ylabel("System Metrics (%)")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper left")
        ax1.set_title("Ollama Inference Hardware Performance Trends")
        
        # Plot GPU
        if has_gpu_monitoring:
            ax2.plot(time_stamps, gpu_util, label="GPU Util (%)", color="limegreen", alpha=0.8)
            ax2.plot(time_stamps, gpu_vram, label="VRAM Util (%)", color="darkgreen", alpha=0.8)
            ax2_power = ax2.twinx()
            ax2_power.plot(time_stamps, gpu_power, label="GPU Power (W)", color="crimson", alpha=0.7)
            ax2_power.set_ylabel("Power (Watts)")
            
            # Combine legends
            lines, labels = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_power.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="upper left")
            
        ax2.set_ylabel("GPU Metrics (%)")
        ax2.set_xlabel("Time (seconds)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        artifact_dir = r"C:\Users\iamsa\.gemini\antigravity\brain\3f36aee9-38b2-4855-81a7-7d26e2a5ef6f"
        os.makedirs(artifact_dir, exist_ok=True)
        plot_path = os.path.join(artifact_dir, "ollama_trends.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"\nSaved hardware trends plot to {plot_path}")
    else:
        print("No monitoring data collected.")

if __name__ == "__main__":
    main()
