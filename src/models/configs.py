from dataclasses import dataclass


@dataclass(frozen=True)
class StratsConfig:
    hid_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    attention_dropout: float
    head_layers: list[str]
    
    def __post_init__(self):
        assert self.hid_dim > 0, 'Hidden dimension must be positive'
        assert self.num_layers > 0, 'Number of layers must be positive'
        assert self.num_heads > 0, 'Number of heads must be positive'
        assert 0 <= self.dropout < 1, 'Dropout must be in [0, 1)'
        assert 0 <= self.attention_dropout < 1, 'Attention dropout must be in [0, 1)'


@dataclass(frozen=True)
class FeaturesInfo:
    demographics_num: int
    features_num: int
    
    def __post_init__(self):
        assert self.demographics_num > 0, 'Demographics number must be positive'
        assert self.features_num > 0, 'Features number must be positive'
