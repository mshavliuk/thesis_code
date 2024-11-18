from dataclasses import dataclass

from pydantic import (
    BaseModel,
    Field,
)


class StratsConfig(BaseModel):
    hid_dim: int = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    dropout: float = Field(..., ge=0, lt=1)
    attention_dropout: float = Field(..., ge=0, lt=1)
    head_layers: list[str]
    ablate: list[str] = Field(default_factory=list)
    
    # ts - entire timeseries embedding
    #   cve_time - time embedding
    #   cve_value - value embedding
    #   variable_emb - variable embedding
    # demo - demographics embedding
    
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
