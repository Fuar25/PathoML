# PathoML/models

## 1. Purpose
MIL model architectures. Each model maps a bag of patch features to classification logits.

## 2. BaseModel Contract
Defined in `PathoML.optimization.interfaces`. Two base classes:
- `BaseModel(nn.Module)` — generic model; `forward(data_dict) -> {'logits': Tensor}`
- `BaseMIL(BaseModel)` — MIL-specific; requires `aggregator: Aggregator` and `classifier: Classifier`

All models must:
- Accept `data_dict: dict` in `forward()` (at minimum key `'features'`)
- Return a dict with key `'logits'` (shape `(B, num_classes)`)

## 3. Available Models

| Class | Registry key | Description |
|-------|-------------|-------------|
| `ABMIL` | `abmil` | Gated attention MIL (Ilse et al. 2018). Pipeline: FeatureEncoder → GatedAttention → LinearClassifier |
| `LinearProbe` | `linear_probe` | Mean-pool baseline; single linear layer over bag features |
| `MLP` | `mlp` | Single hidden-layer MLP with GELU activation and Dropout |

### ABMIL model-specific kwargs (via `ModelConfig.model_kwargs`)
- `gated: bool = True` — enable gated attention (sigmoid branch)
- `attention_dim: Optional[int] = None` — attention hidden dim; defaults to `hidden_dim // 2`
- `encoder_dropout: float` — encoder-specific dropout (fallback: global `dropout`)
- `classifier_dropout: float` — classifier-specific dropout (fallback: global `dropout`)

## 4. Registration and Usage
```python
# Models self-register on import via @register_model decorator.
from PathoML.optimization.registry import create_model

model = create_model("abmil", input_dim=1536, hidden_dim=512, num_classes=1,
                     gated=True, attention_dim=256)
```

## Decided
- `DataDict = Dict[str, torch.Tensor]` type alias keeps forward signatures readable.
- `external_impl` hook on `ABMIL` allows injecting a custom paper implementation without changing the registry interface.

## TODO
1. Interpretability outputs: attention maps, bag embeddings — specify export interface once `interpretability/` module is designed.
