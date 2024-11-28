from pydantic import (
    BaseModel,
    Field,
)


class StratsConfig(BaseModel):
    quantized: bool = False
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


class FeaturesInfo(BaseModel):
    demographics_num: int = Field(..., gt=0)
    features_num: int = Field(..., gt=0)
