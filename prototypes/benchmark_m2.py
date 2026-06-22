# /// script
# dependencies = [
#   "torch",
#   "numpy",
# ]
# ///

import os
import sys
import time
import numpy as np
import torch
import ctypes
from ctypes import wintypes

# ==========================================
# Windows Direct I/O Helper Setup
# ==========================================
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_WRITE_THROUGH = 0x80000000
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = -1

# Load kernel32
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# Define Windows APIs
CreateFileW = kernel32.CreateFileW
CreateFileW.argtypes = [
    wintypes.LPCWSTR,     # lpFileName
    wintypes.DWORD,      # dwDesiredAccess
    wintypes.DWORD,      # dwShareMode
    ctypes.c_void_p,     # lpSecurityAttributes
    wintypes.DWORD,      # dwCreationDisposition
    wintypes.DWORD,      # dwFlagsAndAttributes
    wintypes.HANDLE      # hTemplateFile
]
CreateFileW.restype = wintypes.HANDLE

ReadFile = kernel32.ReadFile
ReadFile.argtypes = [
    wintypes.HANDLE,     # hFile
    ctypes.c_void_p,     # lpBuffer
    wintypes.DWORD,      # nNumberOfBytesToRead
    ctypes.POINTER(wintypes.DWORD), # lpNumberOfBytesRead
    ctypes.c_void_p      # lpOverlapped
]
ReadFile.restype = wintypes.BOOL

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

VirtualAlloc = kernel32.VirtualAlloc
VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
VirtualAlloc.restype = ctypes.c_void_p

VirtualFree = kernel32.VirtualFree
VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD]
VirtualFree.restype = wintypes.BOOL

def read_file_direct(filepath, size):
    """Reads a file bypassing the OS page cache using Windows Direct I/O."""
    # Sector size alignment (usually 4096 bytes on modern systems)
    aligned_size = ((size + 4095) // 4096) * 4096
    
    # Open file with NO_BUFFERING flag
    handle = CreateFileW(
        filepath,
        GENERIC_READ,
        FILE_SHARE_READ,
        None,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING,
        None
    )
    if handle == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error())
        
    # Windows requires the buffer to be sector-aligned.
    # VirtualAlloc allocates pages, which are 4KB-aligned.
    MEM_COMMIT = 0x00001000
    MEM_RESERVE = 0x00002000
    PAGE_READWRITE = 0x04
    
    buf = VirtualAlloc(None, aligned_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    if not buf:
        CloseHandle(handle)
        raise MemoryError("Failed to allocate page-aligned buffer.")
        
    bytes_read = wintypes.DWORD(0)
    
    try:
        start_t = time.perf_counter()
        success = ReadFile(handle, buf, aligned_size, ctypes.byref(bytes_read), None)
        end_t = time.perf_counter()
        
        if not success:
            raise ctypes.WinError(ctypes.get_last_error())
            
        read_time = end_t - start_t
        return read_time, bytes_read.value
    finally:
        CloseHandle(handle)
        MEM_RELEASE = 0x00008000
        VirtualFree(buf, 0, MEM_RELEASE)

def run_attention_benchmark():
    print("==========================================")
    print("GPU Attention Compute Benchmark")
    print("==========================================")
    
    if not torch.cuda.is_available():
        print("Error: CUDA-enabled GPU not available to PyTorch!")
        return None
        
    d_model = 4096
    seq_len = 512
    num_heads = 32
    head_dim = d_model // num_heads

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Parameters: d_model={d_model}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    
    # Generate random Q, K, V tensors on GPU in float16
    Q = torch.randn(1, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    K = torch.randn(1, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    V = torch.randn(1, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float16)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    torch.cuda.synchronize()
    
    # Benchmark
    print("Running 100 attention iterations...")
    start = time.perf_counter()
    for _ in range(100):
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    end = time.perf_counter()

    attn_time_ms = (end - start) / 100 * 1000
    print(f"Attention compute time: {attn_time_ms:.3f} ms")
    return attn_time_ms

def run_nvme_benchmark():
    print("\n==========================================")
    print("Expert Load Disk Benchmark (100MB)")
    print("==========================================")
    
    expert_size_mb = 100
    file_size_bytes = expert_size_mb * 1024 * 1024
    
    # We will write the file in the current workspace directory
    filename = "expert_test.bin"
    
    print(f"Generating and writing {expert_size_mb}MB dummy file to '{filename}'...")
    data = np.random.bytes(file_size_bytes)
    
    with open(filename, 'wb') as f:
        f.write(data)
    
    # --- Test 1: Standard Python Read (Buffered/Cached by OS) ---
    print("\nTest 1: Standard OS-buffered Read (Likely Cached)")
    start = time.perf_counter()
    with open(filename, 'rb') as f:
        _ = f.read()
    end = time.perf_counter()
    
    buffered_time_ms = (end - start) * 1000
    buffered_throughput = expert_size_mb / (end - start)
    print(f"Buffered load time: {buffered_time_ms:.2f} ms")
    print(f"Buffered throughput: {buffered_throughput:.0f} MB/s")
    
    # --- Test 2: Windows Direct I/O Read (Bypasses OS Page Cache) ---
    print("\nTest 2: Direct I/O Read (Bypasses Page Cache)")
    try:
        direct_time, bytes_read = read_file_direct(filename, file_size_bytes)
        direct_time_ms = direct_time * 1000
        direct_throughput = (bytes_read / (1024 * 1024)) / direct_time
        print(f"Direct load time (Cold): {direct_time_ms:.2f} ms")
        print(f"Direct throughput: {direct_throughput:.0f} MB/s")
        print(f"Bytes successfully read: {bytes_read}")
    except Exception as e:
        print(f"Direct I/O read failed: {e}")
        direct_time_ms = None
        direct_throughput = None
        
    # Clean up file
    try:
        os.remove(filename)
        print("\nCleaned up test file.")
    except Exception as e:
        print(f"Could not remove temp file: {e}")
        
    return {
        "buffered_time_ms": buffered_time_ms,
        "buffered_throughput": buffered_throughput,
        "direct_time_ms": direct_time_ms,
        "direct_throughput": direct_throughput
    }

def main():
    attn_ms = run_attention_benchmark()
    disk_results = run_nvme_benchmark()
    
    if attn_ms and disk_results["direct_time_ms"]:
        ratio = disk_results["direct_time_ms"] / attn_ms
        print("\n==========================================")
        print("Comparison Summary")
        print("==========================================")
        print(f"Attention Compute Window: {attn_ms:.3f} ms")
        print(f"Cold Expert Load Time:   {disk_results['direct_time_ms']:.2f} ms")
        print(f"Ratio (Load Time / Attn Window): {ratio:.2f}x")
        
        if disk_results["direct_time_ms"] <= attn_ms:
            print("\n[SUCCESS] The Expert loading time is within the Attention compute window!")
            print("Mechanism 2 is viable for fully hiding disk-to-RAM prefetch latency.")
        else:
            overhead = disk_results["direct_time_ms"] - attn_ms
            print(f"\n[FAIL/LIMITATION] Expert loading time exceeds the Attention compute window by {overhead:.2f} ms.")
            print("Mechanism 2 cannot fully hide the prefetch latency under these parameters.")

if __name__ == "__main__":
    main()
