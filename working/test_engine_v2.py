import unittest
import torch
from working.quantizer_v2 import pack_2bit, unpack_2bit_vectorized_v2
from working.engine_v2 import PASOffloadEngineV2

class TestPASOffloadEngineV2(unittest.TestCase):
    def setUp(self):
        self.in_features = 128
        self.out_features = 256
        self.rank = 8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_pack_unpack_v2(self):
        # 1. Test quantize & dequantize loop correctness
        tensor = torch.randn(self.in_features, dtype=torch.float16, device=self.device)
        packed, scale, min_val = pack_2bit(tensor)
        
        # Unpack
        # packed is packed column vector of shape (bytes_per_col,)
        # For vectorized unpack, let's shape it as 2D: (1, bytes_per_col)
        packed_2d = packed.unsqueeze(0).to(self.device)
        scales = torch.tensor([scale], dtype=torch.float16, device=self.device)
        min_vals = torch.tensor([min_val], dtype=torch.float16, device=self.device)
        
        unpacked = unpack_2bit_vectorized_v2(
            packed_2d, scales, min_vals, (1, self.in_features)
        ).squeeze(0)
        
        # Compute quantization error
        error = torch.mean(torch.abs(tensor - unpacked)).item()
        # Max theoretical quantization error for 2-bit uniform is 0.5 * step_size
        step_size = (tensor.max() - tensor.min()).item() / 3.0
        self.assertLess(error, step_size)

    def test_engine_forward_v2(self):
        # 2. Test standard forward pass correctness of V2 engine
        engine = PASOffloadEngineV2(self.in_features, self.out_features, self.rank)
        
        weights = torch.randn(self.out_features, self.in_features, dtype=torch.float16)
        engine.load_weights(weights)
        
        x = torch.randn(1, self.in_features, dtype=torch.float16)
        if torch.cuda.is_available():
            x = x.cuda()
            
        # Run forward pass
        out, indices = engine.forward(x, threshold=0.15)
        
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], len(indices))
        self.assertGreater(len(indices), 0)

    def test_engine_pipeline_v2(self):
        # 3. Test pipelined submit & execute workflow
        engine = PASOffloadEngineV2(self.in_features, self.out_features, self.rank)
        
        weights = torch.randn(self.out_features, self.in_features, dtype=torch.float16)
        engine.load_weights(weights)
        
        x = torch.randn(1, self.in_features, dtype=torch.float16)
        if torch.cuda.is_available():
            x = x.cuda()
            
        # Submit step
        ticket = engine.submit_forward(x, threshold=0.15, buffer_idx=0)
        # Execute step
        out, indices = engine.execute_forward(x, ticket)
        
        # Test correctness compared to synchronous forward wrapper
        out_sync, indices_sync = engine.forward(x, threshold=0.15)
        
        self.assertTrue(torch.allclose(out, out_sync))
        self.assertTrue(torch.equal(indices, indices_sync))

if __name__ == '__main__':
    unittest.main()
