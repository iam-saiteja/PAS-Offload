# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "lz4",
#   "zstandard",
#   "matplotlib",
# ]
# ///

import os
import torch
import lz4.frame
import zstandard as zstd
import numpy as np
import time
import matplotlib.pyplot as plt
import math

def calculate_entropy(data_bytes):
    """Calculates the Shannon Entropy in bits per byte of a byte stream."""
    if not data_bytes:
        return 0
    counts = np.bincount(np.frombuffer(data_bytes, dtype=np.uint8))
    probabilities = counts[counts > 0] / len(data_bytes)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def benchmark_codec(name, compress_fn, decompress_fn, raw_bytes):
    raw_size_mb = len(raw_bytes) / (1024 * 1024)
    
    # 1. Compress
    start_t = time.perf_counter()
    compressed_bytes = compress_fn(raw_bytes)
    compress_time = time.perf_counter() - start_t
    
    comp_size_mb = len(compressed_bytes) / (1024 * 1024)
    ratio = raw_size_mb / comp_size_mb
    
    # 2. Decompress
    iters = 20
    start_t = time.perf_counter()
    for _ in range(iters):
        decomp_bytes = decompress_fn(compressed_bytes)
    decomp_time = (time.perf_counter() - start_t) / iters
    
    assert len(decomp_bytes) == len(raw_bytes), f"{name} Decompression failed!"
    
    decomp_throughput_mb_s = raw_size_mb / decomp_time
    
    return {
        "Name": name,
        "Ratio": f"{ratio:.2f}x",
        "Comp Size": f"{comp_size_mb:.2f} MB",
        "Decomp Time": f"{decomp_time*1000:.2f} ms",
        "Decomp Throughput": f"{decomp_throughput_mb_s:.0f} MB/s"
    }

def analyze_and_plot_distribution(weights, name, artifact_dir):
    flat_weights = weights.flatten().numpy()
    
    # Basic Stats
    mean = flat_weights.mean()
    std = flat_weights.std()
    min_val = flat_weights.min()
    max_val = flat_weights.max()
    
    # Zeros and clipping
    zero_pct = (flat_weights == 0).sum() / len(flat_weights) * 100
    clipped_neg = (flat_weights <= -128).sum() / len(flat_weights) * 100
    clipped_pos = (flat_weights >= 127).sum() / len(flat_weights) * 100
    
    # Unique values
    vals, counts = np.unique(flat_weights, return_counts=True)
    
    # Print statistics
    print(f"\n--- Distribution: {name} ---")
    print(f"Mean: {mean:.4f}, Std: {std:.4f}, Min: {min_val}, Max: {max_val}")
    print(f"Zeros: {zero_pct:.2f}%, Clipped (<= -128): {clipped_neg:.2f}%, Clipped (>= 127): {clipped_pos:.2f}%")
    
    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(flat_weights, bins=256, range=(-128, 127), color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Weight Distribution: {name}")
    plt.xlabel("Quantized Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save to artifact directory
    plot_path = os.path.join(artifact_dir, f"{name.lower().replace(' ', '_')}_dist.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved distribution plot to {plot_path}")
    
    return vals, counts

def main():
    # Set artifact directory path
    artifact_dir = r"C:\Users\iamsa\.gemini\antigravity\brain\3f36aee9-38b2-4855-81a7-7d26e2a5ef6f"
    os.makedirs(artifact_dir, exist_ok=True)
    
    dim1, dim2 = 4096, 4096
    
    # ==========================================
    # EXPERIMENT A: Float16 Weights
    # ==========================================
    print("==========================================")
    print("EXPERIMENT A: Float16 Benchmark")
    print("==========================================")
    weights_f16 = torch.randn(dim1, dim2, dtype=torch.float16)
    raw_bytes_f16 = weights_f16.numpy().tobytes()
    raw_mb_f16 = len(raw_bytes_f16) / (1024 * 1024)
    
    print(f"Raw f16 Layer Size: {raw_mb_f16:.2f} MB")
    
    entropy_f16 = calculate_entropy(raw_bytes_f16)
    max_ratio_f16 = 8.0 / entropy_f16
    print(f"Shannon Entropy (Byte-level): {entropy_f16:.4f} bits/byte (Max Theoretical Lossless Compression: {max_ratio_f16:.2f}x)")
    
    results_f16 = []
    
    # LZ4 Standard
    results_f16.append(benchmark_codec(
        "LZ4 (Standard)",
        lambda b: lz4.frame.compress(b, compression_level=lz4.frame.COMPRESSIONLEVEL_MIN),
        lambda b: lz4.frame.decompress(b),
        raw_bytes_f16
    ))
    
    # Zstd
    zstd_comp = zstd.ZstdCompressor(level=3)
    zstd_decomp = zstd.ZstdDecompressor()
    results_f16.append(benchmark_codec(
        "Zstd (Level 3)",
        lambda b: zstd_comp.compress(b),
        lambda b: zstd_decomp.decompress(b),
        raw_bytes_f16
    ))
    
    # Delta + LZ4
    def delta_lz4_compress_f16(b):
        arr = np.frombuffer(b, dtype=np.int16)
        delta = np.diff(arr, prepend=arr[0])
        return lz4.frame.compress(delta.tobytes(), compression_level=lz4.frame.COMPRESSIONLEVEL_MIN)
        
    def delta_lz4_decompress_f16(b):
        decomp = lz4.frame.decompress(b)
        delta = np.frombuffer(decomp, dtype=np.int16)
        return np.cumsum(delta, dtype=np.int16).tobytes()

    results_f16.append(benchmark_codec(
        "Delta + LZ4",
        delta_lz4_compress_f16,
        delta_lz4_decompress_f16,
        raw_bytes_f16
    ))
    
    print(f"\n{'Codec':<15} | {'Ratio':<6} | {'Comp Size':<10} | {'Decomp Time':<12} | {'Decomp Throughput'}")
    print("-" * 70)
    for r in results_f16:
        print(f"{r['Name']:<15} | {r['Ratio']:<6} | {r['Comp Size']:<10} | {r['Decomp Time']:<12} | {r['Decomp Throughput']}")

    # ==========================================
    # EXPERIMENT B: int8 Distribution & Entropy Analysis
    # ==========================================
    print("\n==========================================")
    print("EXPERIMENT B: int8 Distribution & Entropy Analysis")
    print("==========================================")
    
    # 1. Original Min-Max Quantization
    weights = torch.randn(dim1, dim2, dtype=torch.float32)
    scale = weights.abs().max() / 127.0
    weights_int8_minmax = torch.round(weights / scale).to(torch.int8)
    bytes_minmax = weights_int8_minmax.numpy().tobytes()
    entropy_minmax = calculate_entropy(bytes_minmax)
    max_ratio_minmax = 8.0 / entropy_minmax
    
    print(f"\nOriginal Min-Max Quantized int8:")
    print(f"Shannon Entropy (Byte-level): {entropy_minmax:.4f} bits/byte (Max Theoretical Lossless Compression: {max_ratio_minmax:.2f}x)")
    analyze_and_plot_distribution(weights_int8_minmax, "Min-Max Quantized", artifact_dir)
    
    # 2. User's Clipped Distribution
    weights_int8_clipped = (torch.randn(dim1, dim2) * 127).clamp(-128, 127).to(torch.int8)
    bytes_clipped = weights_int8_clipped.numpy().tobytes()
    entropy_clipped = calculate_entropy(bytes_clipped)
    max_ratio_clipped = 8.0 / entropy_clipped
    
    print(f"\nUser Clipped int8 (randn * 127, clamped):")
    print(f"Shannon Entropy (Byte-level): {entropy_clipped:.4f} bits/byte (Max Theoretical Lossless Compression: {max_ratio_clipped:.2f}x)")
    vals, counts = analyze_and_plot_distribution(weights_int8_clipped, "Clipped randn", artifact_dir)
    
    # Print sample of the histogram counts for the clipped distribution (as requested: "values, counts")
    print(f"\nClipped randn histogram sample (first 10 unique values and their counts):")
    for v, c in list(zip(vals, counts))[:10]:
        print(f"Value: {v:<4} | Count: {c}")
    print("...")
    print(f"Clipped randn histogram sample (middle 10 unique values around zero and their counts):")
    mid = len(vals) // 2
    for v, c in list(zip(vals, counts))[mid-5:mid+5]:
        print(f"Value: {v:<4} | Count: {c}")

if __name__ == "__main__":
    main()
