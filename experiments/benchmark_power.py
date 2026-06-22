import os
import sys
import time
import threading
import torch
import matplotlib.pyplot as plt
from pynvml import *

# Initialize NVML
try:
    nvmlInit()
    nvml_handle = nvmlDeviceGetHandleByIndex(0)
    has_nvml = True
    device_name = nvmlDeviceGetName(nvml_handle)
    if isinstance(device_name, bytes):
        device_name = device_name.decode('utf-8')
except Exception as e:
    print(f"Warning: NVML initialization failed (Power monitoring disabled): {e}")
    has_nvml = False

# Global telemetry storage
monitoring = True
time_stamps = []
power_samples = []

def monitor_power_worker(start_time):
    global monitoring
    while monitoring:
        if has_nvml:
            try:
                # Get current power draw in Watts
                power = nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
                power_samples.append(power)
                time_stamps.append(time.perf_counter() - start_time)
            except Exception:
                pass
        time.sleep(0.01)  # sample power every 10ms

def run_workload(transfer_size_mb, duration_sec):
    # Ensure element count is an integer
    elements = int(transfer_size_mb * 1024 * 1024 // 2)  # float16 is 2 bytes
    dummy_cpu = torch.zeros(elements, dtype=torch.float16, pin_memory=True)
    
    start_t = time.perf_counter()
    while time.perf_counter() - start_t < duration_sec:
        # Constantly stream the tensor to GPU to simulate active bus loading
        _ = dummy_cpu.cuda(non_blocking=True)
        # Execute a mock matrix multiplication to draw processing power
        torch.cuda.synchronize()
        time.sleep(0.01)  # brief pause to simulate token interval

def run_power_experiment():
    global monitoring, time_stamps, power_samples
    
    print("==================================================")
    print("EXPERIMENT: GPU Power and Thermal Profile Comparison")
    print("==================================================")
    
    if not has_nvml:
        print("Error: NVIDIA NVML is required to capture GPU power draw.")
        return
        
    print(f"Tracking GPU: {device_name}")
    
    # --------------------------------------------------
    # Phase 1: Benchmark Standard Offload (Full Weight Streaming)
    # --------------------------------------------------
    print("\nStarting Phase 1: Simulating Standard Full Weight Streaming (90MB)...")
    time_stamps = []
    power_samples = []
    monitoring = True
    
    start_time = time.perf_counter()
    monitor_thread = threading.Thread(target=monitor_power_worker, args=(start_time,))
    monitor_thread.start()
    
    # Simulate streaming 90MB layers continuously for 4 seconds
    run_workload(90, 4)
    
    monitoring = False
    monitor_thread.join()
    
    std_times = list(time_stamps)
    std_powers = list(power_samples)
    print(f"Phase 1 Complete. Avg Power: {sum(std_powers)/len(std_powers):.2f} W | Peak: {max(std_powers):.2f} W")
    
    # Let GPU cool down
    print("Cooling down GPU for 3 seconds...")
    time.sleep(3)
    
    # --------------------------------------------------
    # Phase 2: Benchmark PAS-Offload (2-bit Sliced weights: 1.13MB)
    # --------------------------------------------------
    print("\nStarting Phase 2: Simulating PAS-Offload 2-bit Sliced Weight Streaming (1.13MB)...")
    time_stamps = []
    power_samples = []
    monitoring = True
    
    start_time = time.perf_counter()
    monitor_thread = threading.Thread(target=monitor_power_worker, args=(start_time,))
    monitor_thread.start()
    
    # Simulate streaming 1.13MB layers continuously for 4 seconds
    run_workload(1.13, 4)
    
    monitoring = False
    monitor_thread.join()
    
    pas_times = list(time_stamps)
    pas_powers = list(power_samples)
    print(f"Phase 2 Complete. Avg Power: {sum(pas_powers)/len(pas_powers):.2f} W | Peak: {max(pas_powers):.2f} W")
    
    # NVML Shutdown
    nvmlShutdown()
    
    # --------------------------------------------------
    # Save Comparison Plot
    # --------------------------------------------------
    print("\nGenerating power comparison plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(std_times, std_powers, label="Standard Full Streaming (90MB)", color="crimson", alpha=0.8, linewidth=2)
    plt.plot(pas_times, pas_powers, label="PAS-Offload Sliced Streaming (1.13MB)", color="limegreen", alpha=0.8, linewidth=2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("GPU Power Draw (Watts)")
    plt.title(f"GPU Power Consumption: Standard vs PAS-Offload ({device_name})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right")
    
    # Create images directory
    image_dir = "../images" if os.path.basename(os.getcwd()) == "experiments" else "images"
    os.makedirs(image_dir, exist_ok=True)
    plot_path = os.path.join(image_dir, "power_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Power comparison plot saved to: {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Error: CUDA is required to run the power benchmark.")
        sys.exit(1)
    run_power_experiment()
