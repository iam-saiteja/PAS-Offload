import torch
import torch.nn as nn
from pas_offload.predictor import LowRankPredictor
from .quantizer_v3 import pack_2bit, unpack_2bit_lut_v3, get_lut

class PASOffloadEngineV3:
    """
    V3 PAS-Offload coordination engine:
    - Double-buffered pipelining.
    - Zero-allocation pinned memory slices.
    - LUT-based ultra-fast dequantization on the GPU.
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
            # Ensure LUT is initialized
            get_lut('cuda')
            
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
            
            # 3. Pre-allocated GPU buffers
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
            
            # 4. Independent CUDA streams for double-buffering
            self.streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        else:
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
            self.streams = [None, None]
            
    def load_weights(self, weights_f16: torch.Tensor):
        assert weights_f16.shape == (self.out_features, self.in_features)
        
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

    def submit_forward(self, x: torch.Tensor, threshold: float = 0.15, buffer_idx: int = 0, x_cpu_hint: torch.Tensor = None, top_k: int = None) -> dict:
        if x_cpu_hint is not None:
            x_cpu = x_cpu_hint[0] if x_cpu_hint.dim() > 1 else x_cpu_hint
        else:
            x_cpu = x[0]
            
        indices = self.predictor.predict_indices(x_cpu, threshold, top_k=top_k)
        active_cols = len(indices)
        
        if active_cols == 0:
            return {'buffer_idx': buffer_idx, 'active_cols': 0, 'indices': indices, 'stream': self.streams[buffer_idx]}
            
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
        
        stream = self.streams[buffer_idx]
        
        if self.use_cuda:
            with torch.cuda.stream(stream):
                self.sliced_packed_gpu[buffer_idx][:active_cols].copy_(
                    self.sliced_packed_pinned[buffer_idx][:active_cols], non_blocking=True
                )
                self.scales_gpu[buffer_idx][:active_cols].copy_(
                    self.scales_pinned[buffer_idx][:active_cols], non_blocking=True
                )
                self.min_vals_gpu[buffer_idx][:active_cols].copy_(
                    self.min_vals_pinned[buffer_idx][:active_cols], non_blocking=True
                )
                
                unpack_2bit_lut_v3(
                    self.sliced_packed_gpu[buffer_idx][:active_cols],
                    self.scales_gpu[buffer_idx][:active_cols],
                    self.min_vals_gpu[buffer_idx][:active_cols],
                    (active_cols, self.in_features),
                    out_tensor=self.unpacked_weights_gpu[buffer_idx][:active_cols]
                )
        else:
            unpack_2bit_lut_v3(
                self.sliced_packed_gpu[buffer_idx][:active_cols],
                self.scales_gpu[buffer_idx][:active_cols],
                self.min_vals_gpu[buffer_idx][:active_cols],
                (active_cols, self.in_features),
                out_tensor=self.unpacked_weights_gpu[buffer_idx][:active_cols]
            )
            
        return {
            'buffer_idx': buffer_idx,
            'active_cols': active_cols,
            'indices': indices,
            'stream': stream
        }

    def execute_forward(self, x: torch.Tensor, ticket: dict) -> tuple[torch.Tensor, torch.Tensor]:
        buffer_idx = ticket['buffer_idx']
        active_cols = ticket['active_cols']
        indices = ticket['indices']
        stream = ticket['stream']
        
        if active_cols == 0:
            return torch.zeros((1, self.out_features), dtype=torch.float16, device=x.device), indices
            
        if self.use_cuda and stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
            
        weights_gpu = self.unpacked_weights_gpu[buffer_idx][:active_cols]
        out = torch.matmul(x, weights_gpu.t())
        
        return out, indices
        
    def forward(self, x: torch.Tensor, threshold: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
        ticket = self.submit_forward(x, threshold, buffer_idx=0)
        return self.execute_forward(x, ticket)
