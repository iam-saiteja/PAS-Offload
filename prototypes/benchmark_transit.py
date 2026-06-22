# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "lz4",
#   "zstandard",
# ]
# ///

import torch
import lz4.frame
import zstandard as zstd
import numpy as np
import time

def generate_realistic_weights(dim1=4096, dim2=4096):
    """
    Generates a weight matrix mimicking a standard LLM layer.
    Weights are normally distributed around 0, then quantized to int8.
    """
    print(f"Generating {dim1}x{dim2} weight matrix...")
    # Standard normal distribution, typical for unscaled weights
    weights = torch.randn(dim1, dim2, dtype=torch.float32)
    
    # Simple symmetric min-max quantization to int8
    scale = weights.abs().max() / 127.0
    quantized_weights = torch.round(weights / scale).to(torch.int8)
    
    return quantized_weights.numpy().tobytes()

def benchmark_codec(name, compress_fn, decompress_fn, raw_bytes):
    raw_size_mb = len(raw_bytes) / (1024 * 1024)
    
    # 1. Compress
    start_t = time.perf_counter()
    compressed_bytes = compress_fn(raw_bytes)
    compress_time = time.perf_counter() - start_t
    
    comp_size_mb = len(compressed_bytes) / (1024 * 1024)
    ratio = raw_size_mb / comp_size_mb
    
    # 2. Decompress (Run multiple times to average out OS noise)
    iters = 20
    start_t = time.perf_counter()
    for _ in range(iters):
        decomp_bytes = decompress_fn(compressed_bytes)
    decomp_time = (time.perf_counter() - start_t) / iters
    
    assert len(decomp_bytes) == len(raw_bytes), "Decompression failed!"
    
    # 3. Calculate Throughput
    decomp_throughput_mb_s = raw_size_mb / decomp_time
    
    return {
        "Name": name,
        "Ratio": f"{ratio:.2f}x",
        "Comp Size": f"{comp_size_mb:.2f} MB",
        "Decomp Time": f"{decomp_time*1000:.2f} ms",
        "Decomp Throughput": f"{decomp_throughput_mb_s:.0f} MB/s"
    }

def main():
    raw_bytes = generate_realistic_weights(4096, 4096)
    raw_mb = len(raw_bytes) / (1024 * 1024)
    
    print(f"\n--- Baseline ---")
    print(f"Raw Layer Size: {raw_mb:.2f} MB (int8)")
    print(f"PCIe Gen3 target: ~3,000 MB/s")
    print(f"PCIe Gen4 target: ~7,000 MB/s\n")
    
    results = []
    
    # --- Codec 1: LZ4 (Optimized for speed) ---
    results.append(benchmark_codec(
        "LZ4 (Standard)",
        lambda b: lz4.frame.compress(b, compression_level=lz4.frame.COMPRESSIONLEVEL_MIN),
        lambda b: lz4.frame.decompress(b),
        raw_bytes
    ))
    
    # --- Codec 2: Zstd (Optimized for ratio, level 3 is default) ---
    zstd_comp = zstd.ZstdCompressor(level=3)
    zstd_decomp = zstd.ZstdDecompressor()
    results.append(benchmark_codec(
        "Zstd (Level 3)",
        lambda b: zstd_comp.compress(b),
        lambda b: zstd_decomp.decompress(b),
        raw_bytes
    ))
    
    # --- Codec 3: Delta + LZ4 (Naive domain-specific test) ---
    # Neural net weights often have small changes between adjacent values.
    # Delta encoding transforms them into smaller numbers, which LZ4 might compress better.
    def delta_lz4_compress(b):
        arr = np.frombuffer(b, dtype=np.int8)
        delta = np.diff(arr, prepend=arr[0])
        return lz4.frame.compress(delta.tobytes(), compression_level=lz4.frame.COMPRESSIONLEVEL_MIN)
        
    def delta_lz4_decompress(b):
        decomp = lz4.frame.decompress(b)
        delta = np.frombuffer(decomp, dtype=np.int8)
        return np.cumsum(delta, dtype=np.int8).tobytes()

    results.append(benchmark_codec(
        "Delta + LZ4",
        delta_lz4_compress,
        delta_lz4_decompress,
        raw_bytes
    ))
    
    # Print Table
    print(f"{'Codec':<15} | {'Ratio':<6} | {'Comp Size':<10} | {'Decomp Time':<12} | {'Decomp Throughput'}")
    print("-" * 70)
    for r in results:
        print(f"{r['Name']:<15} | {r['Ratio']:<6} | {r['Comp Size']:<10} | {r['Decomp Time']:<12} | {r['Decomp Throughput']}")

if __name__ == "__main__":
    main()
