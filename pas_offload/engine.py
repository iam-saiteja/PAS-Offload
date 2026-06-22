import torch
import torch.nn as nn
from .predictor import LowRankPredictor
from .quantizer import pack_2bit, unpack_2bit_vectorized

class PASOffloadEngine:
    """
    PAS-Offload Coordination Engine (Optimized).
    Handles host-side transposed weight caching, CPU-side active-column prediction,
    pinned memory DMA transfer, and on-GPU weight unpacking and execution.
    
    Includes the following optimizations:
    1. Pre-allocated page-locked (pinned) CPU host buffers for zero-copy DMA.
    2. Double-buffering pipelining to overlap compute with transfer.
    3. Pre-allocated GPU buffers to eliminate dynamic memory allocations.
    4. Custom stream pipelining.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 16):
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        self.predictor = LowRankPredictor(in_features, rank, out_features)
        
        # Calculate packed 2-bit bytes per column vector (4 weights per byte)
        self.bytes_per_col = in_features * 2 // 8
        
        self.use_cuda = torch.cuda.is_available()
        
        # 1. Page-locked (pinned) CPU host cache for full model weights
        if self.use_cuda:
            self.packed_weights_cpu = torch.zeros(out_features, self.bytes_per_col, dtype=torch.uint8).pin_memory()
            self.scales_cpu = torch.zeros(out_features, dtype=torch.float16).pin_memory()
            self.min_vals_cpu = torch.zeros(out_features, dtype=torch.float16).pin_memory()
            
            # 2. Pipelined double-buffers on host CPU (pinned)
            self.sliced_packed_pinned = [
                torch.zeros(out_features, self.bytes_per_col, dtype=torch.uint8).pin_memory(),
                torch.zeros(out_features, self.bytes_per_col, dtype=torch.uint8).pin_memory()
            ]
            self.scales_pinned = [
                torch.zeros(out_features, dtype=torch.float16).pin_memory(),
                torch.zeros(out_features, dtype=torch.float16).pin_memory()
            ]
            self.min_vals_pinned = [
                torch.zeros(out_features, dtype=torch.float16).pin_memory(),
                torch.zeros(out_features, dtype=torch.float16).pin_memory()
            ]
            
            # 3. Pre-allocated GPU buffers to eliminate dynamic allocation overhead
            self.sliced_packed_gpu = [
                torch.zeros(out_features, self.bytes_per_col, dtype=torch.uint8, device='cuda'),
                torch.zeros(out_features, self.bytes_per_col, dtype=torch.uint8, device='cuda')
            ]
            self.scales_gpu = [
                torch.zeros(out_features, dtype=torch.float16, device='cuda'),
                torch.zeros(out_features, dtype=torch.float16, device='cuda')
            ]
            self.min_vals_gpu = [
                torch.zeros(out_features, dtype=torch.float16, device='cuda'),
                torch.zeros(out_features, dtype=torch.float16, device='cuda')
            ]
            self.unpacked_weights_gpu = [
                torch.zeros(out_features, in_features, dtype=torch.float16, device='cuda'),
                torch.zeros(out_features, in_features, dtype=torch.float16, device='cuda')
            ]
            self.temp_uint8_gpu = [
                torch.zeros(out_features, self.bytes_per_col, 4, dtype=torch.uint8, device='cuda'),
                torch.zeros(out_features, self.bytes_per_col, 4, dtype=torch.uint8, device='cuda')
            ]
            
            # 4. Independent CUDA streams for double-buffering
            self.streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        else:
            # Fallback for CPU-only execution (testing/dev)
            self.packed_weights_cpu = torch.zeros(out_features, self.bytes_per_col, dtype=torch.uint8)
            self.scales_cpu = torch.zeros(out_features, dtype=torch.float16)
            self.min_vals_cpu = torch.zeros(out_features, dtype=torch.float16)
            
            self.sliced_packed_pinned = [
                torch.zeros(out_features, self.bytes_per_col, dtype=torch.uint8),
                torch.zeros(out_features, self.bytes_per_col, dtype=torch.uint8)
            ]
            self.scales_pinned = [
                torch.zeros(out_features, dtype=torch.float16),
                torch.zeros(out_features, dtype=torch.float16)
            ]
            self.min_vals_pinned = [
                torch.zeros(out_features, dtype=torch.float16),
                torch.zeros(out_features, dtype=torch.float16)
            ]
            
            self.sliced_packed_gpu = self.sliced_packed_pinned
            self.scales_gpu = self.scales_pinned
            self.min_vals_gpu = self.min_vals_pinned
            self.unpacked_weights_gpu = [
                torch.zeros(out_features, in_features, dtype=torch.float16),
                torch.zeros(out_features, in_features, dtype=torch.float16)
            ]
            self.temp_uint8_gpu = [
                torch.zeros(out_features, self.bytes_per_col, 4, dtype=torch.uint8),
                torch.zeros(out_features, self.bytes_per_col, 4, dtype=torch.uint8)
            ]
            self.streams = [None, None]
            
    def load_weights(self, weights_f16: torch.Tensor):
        """
        Compresses and loads weight matrix of shape (out_features, in_features)
        into the transposed CPU host cache.
        """
        assert weights_f16.shape == (self.out_features, self.in_features), \
            f"Weight shape mismatch. Expected {(self.out_features, self.in_features)}, got {weights_f16.shape}"
        
        packed_temp = torch.zeros(self.out_features, self.bytes_per_col, dtype=torch.uint8)
        scales_temp = torch.zeros(self.out_features, dtype=torch.float16)
        min_vals_temp = torch.zeros(self.out_features, dtype=torch.float16)
        
        for i in range(self.out_features):
            col_weights = weights_f16[i]
            packed_col, scale, min_val = pack_2bit(col_weights)
            packed_temp[i] = packed_col
            scales_temp[i] = scale
            min_vals_temp[i] = min_val
            
        self.packed_weights_cpu.copy_(packed_temp)
        self.scales_cpu.copy_(scales_temp)
        self.min_vals_cpu.copy_(min_vals_temp)

    def submit_forward(self, x: torch.Tensor, threshold: float = 0.15, buffer_idx: int = 0, top_k: int = None, x_cpu_hint: torch.Tensor = None) -> dict:
        """
        Asynchronously schedules active column prediction, CPU slicing, 
        PCIe DMA weight streaming, and GPU dequantization on a dedicated stream.
        
        Args:
            x (torch.Tensor): Input hidden state vector, shape (1, in_features).
            threshold (float): Prediction activation threshold.
            buffer_idx (int): Buffer slot index (0 or 1) to use.
            top_k (int): Optional top-k constraint for column selection.
            x_cpu_hint (torch.Tensor): Optional CPU tensor for the input to bypass slow explicit .cpu() copy.
            
        Returns:
            ticket (dict): Metadata about the scheduled job.
        """
        assert x.shape == (1, self.in_features), f"Input shape mismatch. Expected {(1, self.in_features)}, got {x.shape}"
        
        # 1. Run CPU Predictor
        if x_cpu_hint is not None:
            x_cpu = x_cpu_hint[0] if x_cpu_hint.dim() > 1 else x_cpu_hint
        else:
            x_cpu = x[0].cpu().to(dtype=self.predictor.w1.weight.dtype)
            
        indices = self.predictor.predict_indices(x_cpu, threshold, top_k=top_k)
        active_cols = len(indices)
        
        # 2. Slice directly into the pre-allocated pinned CPU buffer using out parameter
        torch.index_select(
            self.packed_weights_cpu, 0, indices,
            out=self.sliced_packed_pinned[buffer_idx][:active_cols]
        )
        torch.index_select(
            self.scales_cpu, 0, indices,
            out=self.scales_pinned[buffer_idx][:active_cols]
        )
        torch.index_select(
            self.min_vals_cpu, 0, indices,
            out=self.min_vals_pinned[buffer_idx][:active_cols]
        )
        
        # 3. Stream asynchronously via DMA to GPU and execute unpack
        stream = self.streams[buffer_idx]
        
        if self.use_cuda:
            with torch.cuda.stream(stream):
                # Copy from host pinned buffers to pre-allocated GPU buffers
                self.sliced_packed_gpu[buffer_idx][:active_cols].copy_(
                    self.sliced_packed_pinned[buffer_idx][:active_cols], non_blocking=True
                )
                self.scales_gpu[buffer_idx][:active_cols].copy_(
                    self.scales_pinned[buffer_idx][:active_cols], non_blocking=True
                )
                self.min_vals_gpu[buffer_idx][:active_cols].copy_(
                    self.min_vals_pinned[buffer_idx][:active_cols], non_blocking=True
                )
                
                # Unpack 2-bit weights directly into the pre-allocated GPU output buffer
                unpack_2bit_vectorized(
                    self.sliced_packed_gpu[buffer_idx][:active_cols],
                    self.scales_gpu[buffer_idx][:active_cols],
                    self.min_vals_gpu[buffer_idx][:active_cols],
                    (active_cols, self.in_features),
                    out_tensor=self.unpacked_weights_gpu[buffer_idx][:active_cols],
                    temp_uint8_buffer=self.temp_uint8_gpu[buffer_idx]
                )
        else:
            # Fallback dequantization on CPU
            unpack_2bit_vectorized(
                self.sliced_packed_gpu[buffer_idx][:active_cols],
                self.scales_gpu[buffer_idx][:active_cols],
                self.min_vals_gpu[buffer_idx][:active_cols],
                (active_cols, self.in_features),
                out_tensor=self.unpacked_weights_gpu[buffer_idx][:active_cols],
                temp_uint8_buffer=self.temp_uint8_gpu[buffer_idx]
            )
            
        return {
            'buffer_idx': buffer_idx,
            'active_cols': active_cols,
            'indices': indices,
            'stream': stream
        }

    def execute_forward(self, x: torch.Tensor, ticket: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Synchronizes the streaming pipeline and executes the matrix multiplication.
        """
        buffer_idx = ticket['buffer_idx']
        active_cols = ticket['active_cols']
        indices = ticket['indices']
        stream = ticket['stream']
        
        # Wait for the async transfer & dequantization to complete
        if self.use_cuda and stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
            
        weights_gpu = self.unpacked_weights_gpu[buffer_idx][:active_cols]
        out = torch.matmul(x, weights_gpu.t())
        
        return out, indices
        
    def forward(self, x: torch.Tensor, threshold: float = 0.15, top_k: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Synchronous wrapper matching the original engine's API.
        """
        ticket = self.submit_forward(x, threshold, buffer_idx=0, top_k=top_k)
        return self.execute_forward(x, ticket)
