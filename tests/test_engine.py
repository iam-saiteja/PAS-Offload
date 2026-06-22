import unittest
import torch
from pas_offload.engine import PASOffloadEngine

class TestPASOffloadEngine(unittest.TestCase):
    def setUp(self):
        self.in_features = 512
        self.out_features = 1024
        self.rank = 16
        self.engine = PASOffloadEngine(self.in_features, self.out_features, self.rank)
        
    def test_weight_loading(self):
        # Generate random float16 weights
        weights = torch.randn(self.out_features, self.in_features, dtype=torch.float16)
        
        # Load weights into CPU cache
        self.engine.load_weights(weights)
        
        # Verify sizes in page-locked memory
        self.assertEqual(self.engine.packed_weights_cpu.shape, (self.out_features, self.in_features * 2 // 8))
        self.assertEqual(self.engine.scales_cpu.shape, (self.out_features,))
        self.assertEqual(self.engine.min_vals_cpu.shape, (self.out_features,))
        
    def test_forward_pass_cuda(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA device not available. Skipping GPU streaming test.")
            
        # Initialize weights
        weights = torch.randn(self.out_features, self.in_features, dtype=torch.float16)
        self.engine.load_weights(weights)
        
        # Generate input hidden state on GPU
        x = torch.randn(1, self.in_features, device='cuda', dtype=torch.float16)
        
        # Execute forward pass
        out, indices = self.engine.forward(x, threshold=0.15)
        
        # Verify outputs
        self.assertEqual(out.dim(), 2)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], len(indices))
        self.assertEqual(out.device.type, 'cuda')
        self.assertEqual(indices.device.type, 'cpu')

if __name__ == "__main__":
    unittest.main()
