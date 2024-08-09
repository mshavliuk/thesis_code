import numpy as np
import pytest
import torch

from src.util.collator import Collator


class TestCollator:
    @pytest.fixture
    def collator(self):
        return Collator()
    
    def test_collates_batch_correctly(self, collator):
        batch = [
            {
                'values': np.array([1.1, 2.2, 3.3], dtype=np.float32),
                'variables': np.array([2, 3, 4]),
                'times': np.array([3, 4, 5])
            },
            {
                'values': np.array([4.4, 5.5]),
                'variables': np.array([1, 2]),
                'times': np.array([1, 2])
            }
        ]
        result = collator(batch)
        assert torch.equal(result['values'],
                           torch.tensor([[1.1, 2.2, 3.3], [4.4, 5.5, 0]], dtype=torch.float32))
        assert torch.equal(result['variables'],
                           torch.tensor([[2, 3, 4], [1, 2, 0]], dtype=torch.int))
        assert torch.equal(result['times'], torch.tensor([[3, 4, 5], [1, 2, 0]], dtype=torch.int))
        assert torch.equal(result['input_mask'],
                           torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.int))
    
    def test_handles_empty_batch(self, collator: Collator):
        batch = []
        result = collator(batch)
        assert result == {}
    
    def test_handles_single_element_batch(self, collator: Collator):
        batch = [{
            'values': np.array([4, 5, 6]),
            'variables': np.array([1, 2, 3]),
            'times': np.array([1, 2, 4])
        }]
        result = collator(batch)
        assert torch.equal(result['values'], torch.tensor([[4, 5, 6]], dtype=torch.int))
        assert torch.equal(result['variables'], torch.tensor([[1, 2, 3]], dtype=torch.int))
        assert torch.equal(result['times'], torch.tensor([[1, 2, 4]], dtype=torch.int))
        assert torch.equal(result['input_mask'], torch.tensor([[1, 1, 1]], dtype=torch.int))
