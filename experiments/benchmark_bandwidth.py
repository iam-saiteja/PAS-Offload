import os
import sys
import matplotlib.pyplot as plt

def run_bandwidth_experiment():
    print("==================================================")
    print("EXPERIMENT: PCIe Bandwidth and Payload Reduction")
    print("==================================================")
    
    # Layer Dimensions (Llama-7B FFN: 4096 x 11008)
    in_features = 4096
    out_features = 11008
    num_layers = 32
    
    # Weights element count
    layer_elements = in_features * out_features
    total_elements = layer_elements * num_layers
    
    # --------------------------------------------------
    # 1. Calculate Payload Sizes (Per layer)
    # --------------------------------------------------
    # Float16 full transfer: 2 bytes per element
    f16_size_bytes = layer_elements * 2
    # Int8 full transfer: 1 byte per element
    int8_size_bytes = layer_elements * 1
    # PAS-Offload 2-bit sparse transfer: 10% active columns, 2 bits per element
    # Elements = 10% of out_features * in_features
    sparse_elements = int(out_features * 0.10) * in_features
    pas_size_bytes = (sparse_elements * 2 // 8) # 2 bits per weight packed into bytes
    
    print(f"Layer Size (Elements): {layer_elements:,}")
    print(f"Active Columns (10%):  {int(out_features * 0.10):,}")
    
    print("\n--- Data Payload Size Per FFN Layer ---")
    print(f"Standard Float16 Layer: {f16_size_bytes / (1024 * 1024):.2f} MB")
    print(f"Standard Int8 Layer:    {int8_size_bytes / (1024 * 1024):.2f} MB")
    print(f"PAS-Offload (2-bit):    {pas_size_bytes / (1024 * 1024):.2f} MB")
    
    reduction_factor = f16_size_bytes / pas_size_bytes
    reduction_pct = (1.0 - pas_size_bytes / f16_size_bytes) * 100
    print(f"Payload Size Reduction: {reduction_pct:.2f}% ({reduction_factor:.1f}x smaller)")
    
    # --------------------------------------------------
    # 2. Translate to PCIe Bus Consumption (for 32-layer generation of 1 token)
    # --------------------------------------------------
    print("\n--- PCIe Bandwidth Required Per Token Generated ---")
    # Total data per token generated
    f16_total_mb = (f16_size_bytes * num_layers) / (1024 * 1024)
    int8_total_mb = (int8_size_bytes * num_layers) / (1024 * 1024)
    pas_total_mb = (pas_size_bytes * num_layers) / (1024 * 1024)
    
    print(f"Standard Float16 Path: {f16_total_mb:.2f} MB / token")
    print(f"Standard Int8 Path:    {int8_total_mb:.2f} MB / token")
    print(f"PAS-Offload (2-bit):    {pas_total_mb:.2f} MB / token")
    
    # Estimate PCIe transfer times on PCIe Gen3 x4 SSD limits (~1500 MB/s actual throughput)
    bus_throughput = 1500.0  # MB/s
    f16_time = f16_total_mb / bus_throughput
    int8_time = int8_total_mb / bus_throughput
    pas_time = pas_total_mb / bus_throughput
    
    print(f"\n--- Estimated PCIe Transfer Latency per Token (at {bus_throughput} MB/s) ---")
    print(f"Standard Float16 Path: {f16_time * 1000:.2f} ms")
    print(f"Standard Int8 Path:    {int8_time * 1000:.2f} ms")
    print(f"PAS-Offload (2-bit):    {pas_time * 1000:.2f} ms")
    
    # Print comparison
    print("\n--------------------------------------------------")
    print("Bandwidth Results Summary")
    print("--------------------------------------------------")
    print(f"PAS-Offload reduces total PCIe traffic from {f16_total_mb:.2f} MB to {pas_total_mb:.2f} MB per token.")
    print("This satisfies the physical throughput constraints of client-grade hardware.")

    # --------------------------------------------------
    # Save Comparison Plot
    # --------------------------------------------------
    print("\nGenerating bandwidth comparison plot...")
    plt.figure(figsize=(9, 5))
    categories = ['Float16 Full Layer', 'Int8 Full Layer', 'PAS-Offload 2-bit (Ours)']
    sizes_mb = [f16_size_bytes / (1024 * 1024), int8_size_bytes / (1024 * 1024), pas_size_bytes / (1024 * 1024)]
    colors = ['#dc3545', '#ffc107', '#28a745'] # red, amber, green
    
    bars = plt.bar(categories, sizes_mb, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (max(sizes_mb) * 0.02), f'{height:.2f} MB', ha='center', va='bottom', fontweight='bold')
        
    plt.ylabel("Data Payload Size per FFN Layer (MB)")
    plt.title("Weight Streaming Payload Size Comparison (Lower is Better)")
    plt.ylim(0, max(sizes_mb) * 1.2)
    plt.grid(True, linestyle="--", alpha=0.5, axis='y')
    
    # Add annotation for reduction
    reduction_pct = (1.0 - pas_size_bytes / f16_size_bytes) * 100
    plt.text(1.0, max(sizes_mb) * 0.8, f"{reduction_pct:.2f}% Payload Reduction", ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", fc="#e2f0d9", ec="#385723", lw=2), fontsize=11, fontweight='bold')
             
    # Create images directory
    image_dir = "../images" if os.path.basename(os.getcwd()) == "experiments" else "images"
    os.makedirs(image_dir, exist_ok=True)
    plot_path = os.path.join(image_dir, "bandwidth_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Bandwidth comparison plot saved to: {os.path.abspath(plot_path)}")

if __name__ == "__main__":
    run_bandwidth_experiment()
