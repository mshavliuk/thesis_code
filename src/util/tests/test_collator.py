import numpy as np
import pytest
import torch

from src.util.collator import Collator


class TestCollator:
    padding_value = 12345
    
    @pytest.fixture(scope='function')
    def collator(self):
        return Collator(padding_variable_value=self.padding_value)
    
    @pytest.fixture(scope='function')
    def batch(self):
        return [
            {
                'values': np.array([1.1, 2.2, 3.3], dtype=np.float32),
                'variables': np.array([2, 3, 4], dtype=np.int32),
                'times': np.array([0.3, 0.4, 0.5], dtype=np.float32),
                'extra': np.array([0.1, 0.2], dtype=np.float64)
            },
            {
                'values': np.array([4.4, 5.5], dtype=np.float32),
                'variables': np.array([1, 2], dtype=np.int32),
                'times': np.array([0.1, 0.2], dtype=np.float32),
                'extra': np.array([0.4, 0.5], dtype=np.float64)
            }
        ]

    
    @pytest.fixture(scope='function')
    def collated_batch(self, collator, batch):
        return collator(batch)
    
    
    def test_collated_batch_has_correct_keys(self, collated_batch):
        assert collated_batch.keys() == {'values', 'variables', 'times', 'input_mask', 'extra'}
    
    def test_collated_batch_has_correct_types(self, collated_batch):
        assert collated_batch['values'].dtype == torch.float32
        assert collated_batch['variables'].dtype == torch.int32
        assert collated_batch['times'].dtype == torch.float32
        assert collated_batch['input_mask'].dtype == torch.bool
        assert collated_batch['extra'].dtype == torch.float64
    
    def test_triplet_values_collated(self, collated_batch):
        assert torch.equal(
            collated_batch['values'],
            torch.tensor([
                [1.1, 2.2, 3.3] + [self.padding_value] * 5,
                [4.4, 5.5] + [self.padding_value] * 6
            ])
        )
        assert torch.equal(
            collated_batch['variables'],
            torch.tensor([
                [2, 3, 4] + [self.padding_value] * 5,
                [1, 2] + [self.padding_value] * 6
            ]))
        assert torch.equal(
            collated_batch['times'],
            torch.tensor([
                [0.3, 0.4, 0.5] + [self.padding_value] * 5,
                [0.1, 0.2] + [self.padding_value] * 6
            ]))
        assert torch.equal(
            collated_batch['input_mask'],
            torch.tensor([
                [1, 1, 1] + [0] * 5,
                [1, 1] + [0] * 6
            ]))
    
    def test_concatenates_non_collated_keys(self, collated_batch):
        assert torch.equal(
            collated_batch['extra'],
            torch.tensor([
                [0.1, 0.2],
                [0.4, 0.5]
            ], dtype=torch.float64)
        )
    
    def test_handles_single_element_batch(self, collator: Collator):
        batch = [{
            'values': np.array([4, 5, 6]),
            'variables': np.array([1, 2, 3]),
            'times': np.array([1, 2, 4])
        }]
        result = collator(batch)
        pad = [self.padding_value] * 5
        assert torch.equal(result['values'], torch.tensor([[4, 5, 6] + pad], dtype=torch.int))
        assert torch.equal(result['variables'], torch.tensor([[1, 2, 3] + pad], dtype=torch.int))
        assert torch.equal(result['times'], torch.tensor([[1, 2, 4] + pad], dtype=torch.int))
        assert torch.equal(
            result['input_mask'],
            torch.tensor([[1, 1, 1] + [0] * 5], dtype=torch.int))
