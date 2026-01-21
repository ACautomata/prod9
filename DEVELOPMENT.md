# DEVELOPMENT.md

Development guide for the prod9 medical imaging research project. This document provides practical tutorials and guidelines for extending and modifying the codebase.

**For architectural details**, see [ARCHITECTURE.md](ARCHITECTURE.md).

**For user-facing documentation**, see [README.md](README.md).

## Table of Contents

- [Code Conventions](#code-conventions)
- [Development Workflow](#development-workflow)
- [Extending MaskGiT](#extending-maskgit)
- [Extending MAISI](#extending-maisi)
- [Using Advanced Features](#using-advanced-features)
- [Adding New Components](#adding-new-components)
- [Debugging and Optimization](#debugging-and-optimization)
- [Testing Guidelines](#testing-guidelines)

---

## Code Conventions

### PyTorch Patterns

- **Persistent tensors**: Use `register_buffer()` for model state that isn't a parameter
  - Examples: FSQ levels, position embeddings, normalization statistics
  ```python
  self.register_buffer("levels", torch.tensor([1, 2, 3]))
  ```

- **Module lists**: Use `nn.ModuleList` for repeated blocks (not Python lists)
  ```python
  self.layers = nn.ModuleList([Block() for _ in range(num_layers)])
  ```

- **Inference methods**: Decorate with `@torch.no_grad()`
  ```python
  @torch.no_grad()
  def inference(self, x):
      # No gradient computation needed
  ```

- **Gradient checkpointing**: Use `use_reentrant=False` for safety
  ```python
  torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
  ```

- **Device detection**: Apple Silicon GPU support
  ```python
  device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
  ```

### Type Checking

The project uses **Pyright standard mode** (enforced in `.vscode/settings.json`).

**Rules**:
- Add type hints to all new code
- **NEVER use `# type: ignore`** - use `typing.cast()` instead
  ```python
  # BAD
  result = some_function()  # type: ignore

  # GOOD
  from typing import cast
  result = cast(ExpectedType, some_function())
  ```

Run type checking before committing:
```bash
pyright
```

### Import Organization

Use `isort` to maintain consistent imports:

```bash
# Organize all imports
isort .

# Check without modifying
isort --check-only .
```

Configuration in `.isort.cfg`.

### Testing Patterns

**Test structure**:
```python
import unittest
import torch

class MyTest(unittest.TestCase):
    def setUp(self):
        # Set up test fixtures
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def test_something(self):
        # Test implementation
        pass
```

**Test categories**:
- **Smoke tests**: Basic forward/backward execution
- **Shape tests**: Input/output dimension verification
- **Roundtrip tests**: Encode-decode consistency
- **Gradient tests**: Backward pass correctness
- **Device tests**: MPS/CPU compatibility

**Mock external dependencies** in sampler tests:
```python
from unittest.mock import Mock

def test_sampler():
    mock_transformer = Mock()
    mock_transformer.return_value = expected_output
    # Test sampler with mock
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `AutoencoderFSQ`, `MaskGiTSampler`)
- **Methods/variables**: `snake_case` (e.g., `compute_losses`, `num_layers`)
- **Constants (buffers)**: `UPPER_SNAKE_CASE` (e.g., `LEVELS`, `POSITIONS`)

---

## Development Workflow

### Quick Development Setup

For fast iteration during development:

```yaml
# In any config file
trainer:
  max_epochs: 3                    # Quick test runs
  logging:
    limit_train_batches: 10         # Use only 10 batches per epoch
    limit_val_batches: 2            # Use only 2 val batches
  accelerator: "gpu"                # Use GPU if available
  devices: 1                        # Single GPU

data:
  num_workers: 4                    # Parallel data loading
  cache_rate: 1.0                   # Cache all data in memory
```

### Profiling

Enable profiling to identify bottlenecks:

```yaml
trainer:
  profiler: "simple"    # Basic profiling
  # profiler: "advanced"  # Detailed profiling (slower)
```

### Gradient Debugging

Detect anomalous operations (slow, use only for debugging):

```yaml
trainer:
  detect_anomaly: true
```

### Testing with Synthetic Data

Use small synthetic datasets for quick tests:

```bash
pytest tests/system/test_autoencoder_training.py -v -s
```

---

## Extending MaskGiT

### Adding a New Scheduler Type

MaskGiT uses schedulers to control masking during training. Add custom schedules by modifying `MaskGiTScheduler`.

**Steps**:

1. **Add scheduler name** to `_SCHEDULE_FUNCTIONS` in [`src/prod9/generator/maskgit.py`](src/prod9/generator/maskgit.py):
   ```python
   class MaskGiTScheduler:
       _SCHEDULE_FUNCTIONS = {
           "cosine": _cosine_schedule,
           "my_custom": _my_custom_schedule,  # Add here
       }
   ```

2. **Implement the schedule function**:
   ```python
   def _my_custom_schedule(step: int, total_steps: int) -> float:
       # Return ratio of masked tokens (0.0 to 1.0)
       # Should be non-decreasing
       progress = step / total_steps
       return custom_function(progress)
   ```

3. **Add tests** in [`tests/unit/generator/test_scheduler.py`](tests/unit/generator/test_scheduler.py):
   ```python
   def test_my_custom_schedule():
       scheduler = MaskGiTScheduler(scheduler_type="my_custom")

       # Test shape (0 to 1 range)
       ratios = [scheduler(step, 100) for step in range(100)]
       assert all(0 <= r <= 1 for r in ratios)

       # Test monotonicity (should be non-decreasing)
       assert all(ratios[i] <= ratios[i+1] for i in range(len(ratios)-1))

       # Test edge cases
       assert scheduler(0, 100) >= 0
       assert scheduler(100, 100) <= 1
   ```

**Reference**: See [`ARCHITECTURE.md`](ARCHITECTURE.md) for scheduler design details.

### Extending the Transformer

When adding new blocks to `TransformerDecoder`, follow the established `AdaLNZeroBlock` pattern.

**Steps**:

1. **Modify `TransformerDecoder`** in [`src/prod9/generator/transformer.py`](src/prod9/generator/transformer.py)

2. **Follow the `AdaLNZeroBlock` pattern**:
   ```python
   class MyCustomBlock(nn.Module):
       def __init__(self, hidden_dim: int, num_heads: int):
           super().__init__()
           self.norm1 = nn.LayerNorm(hidden_dim)
           self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
           self.norm2 = nn.LayerNorm(hidden_dim)
           self.mlp = nn.Sequential(...)
           # Zero-initialized gating for conditioning
           self.gate = ZeroInitLinear(hidden_dim, hidden_dim)

       def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
           # Accept condition argument
           residual = x
           x = self.norm1(x)
           x = self.attn(x, x, x)[0] + residual

           residual = x
           x = self.norm2(x)
           x = self.mlp(x) + residual

           # Apply condition via zero-init gate
           return x + self.gate(condition)

       def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
           # Return hidden_state with same shape as input
           return output
   ```

3. **Add tests** in [`tests/unit/generator/test_transformer.py`](tests/unit/generator/test_transformer.py):
   ```python
   def test_custom_block_forward():
       block = MyCustomBlock(hidden_dim=512, num_heads=8)
       x = torch.randn(2, 100, 512)
       condition = torch.randn(2, 512)

       output = block(x, condition)

       # Shape consistency
       assert output.shape == x.shape

       # Gradient flow
       output.sum().backward()
       assert block.gate.weight.grad is not None
   ```

**Key requirements**:
- Accept `condition` argument for conditional generation
- Return `hidden_state` with same shape as input
- Use adaptive layer normalization with zero-initialization
- Test gradient flow when adding learnable parameters

### Modifying Quantization

The `FiniteScalarQuantizer` (FSQ) converts continuous latents to discrete codes. Modifications must preserve gradient flow via straight-through estimator.

**Key methods** in [`src/prod9/autoencoder/autoencoder_fsq.py`](src/prod9/autoencoder/autoencoder_fsq.py):

- `forward()`: Quantize latent codes with STE gradients
- `embed_code()`: Convert discrete indices to latent vectors
- `_broadcast_indices()`: Reshape indices for multi-dimensional quantization

**Steps**:

1. **Update `FiniteScalarQuantizer`**:
   ```python
   class FiniteScalarQuantizer(nn.Module):
       def forward(self, z: torch.Tensor) -> torch.Tensor:
           # Quantize: continuous z -> discrete z_q
           # Use STE for gradients: d(L)/d(z) = d(L)/d(z_q)
           z_q = self.quantize(z)
           return z + (z_q - z).detach()  # Straight-through estimator

       def embed_code(self, indices: torch.Tensor) -> torch.Tensor:
           # Convert discrete indices to continuous latents
           return self.codebook[indices]

       def _broadcast_indices(self, indices: torch.Tensor) -> torch.Tensor:
           # Reshape for product quantization
           return self._LEVELS.prod() * indices + ...
   ```

2. **Test roundtrip**:
   ```python
   def test_quantizer_roundtrip():
       quantizer = FiniteScalarQuantizer(levels=[8, 8, 8, 6, 5])

       # Generate random indices
       indices = torch.randint(0, 100, (10,))

       # Roundtrip: indices -> latent -> indices
       latent = quantizer.embed_code(indices)
       reconstructed_indices = quantizer.quantize(latent)

       assert torch.all(indices == reconstructed_indices)
   ```

**Critical requirement**: Ensure gradient flow via straight-through estimator. Test with:
```python
def test_quantizer_gradients():
    quantizer = FiniteScalarQuantizer(...)
    z = torch.randn(4, 5, requires_grad=True)

    z_q = quantizer(z)
    loss = z_q.sum()
    loss.backward()

    assert z.grad is not None  # Gradients should flow
```

---

## Extending MAISI

### Configuring Rectified Flow

Rectified Flow balances inference speed vs quality through the number of sampling steps.

**Speed vs quality trade-off**:

```yaml
# In configs/maisi/diffusion/*.yaml
scheduler:
  num_train_timesteps: 1000    # Training timesteps (fixed)
  num_inference_steps: 10      # Inference steps (10-30 typical)
  # Lower = faster but lower quality
  # Higher = slower but better quality
```

**Typical values**:
- `num_inference_steps: 10` - Fast generation, moderate quality
- `num_inference_steps: 20` - Balanced speed/quality (recommended)
- `num_inference_steps: 30` - High quality, slower

**Example usage**:
```bash
# Generate with 10 steps (fast)
prod9-train-maisi-diffusion generate \
    --config configs/maisi/diffusion/brats_diffusion.yaml \
    --num-inference-steps 10 \
    --output outputs/fast_generation

# Generate with 30 steps (high quality)
prod9-train-maisi-diffusion generate \
    --config configs/maisi/diffusion/brats_diffusion.yaml \
    --num-inference-steps 30 \
    --output outputs/high_quality
```

**Reference**: See [`ARCHITECTURE.md`](ARCHITECTURE.md) for Rectified Flow design details.

---

## Using Advanced Features

### Pure In-Context Learning (MaskGiT Stage 2b)

Pure in-context learning enables modality translation and few-shot generation by conditioning on source modalities.

**Concept**: Generate target modality conditioned on source modalities using context sequences.

**Configuration** example:

```yaml
# configs/maskgit/brats/stage2/in_context.yaml
model:
  transformer_type: "single_stream"  # Use TransformerDecoderSingleStream
  hidden_dim: 512
  num_layers: 12
  num_heads: 8

modality_processor:
  latent_dim: 5              # FSQ levels [8,8,8,6,5]
  hidden_dim: 512
  num_classes: 4             # T1, T1ce, T2, FLAIR
  patch_size: 2
  max_context_length: 8      # Max number of (label, latent) pairs

training:
  # Source modalities for in-context learning
  source_modalities: [0, 1, 2]  # T1, T1ce, T2
  target_modality: 3            # FLAIR
  batch_size: 4
```

**Training**:
```bash
prod9-train-transformer train \
    --config configs/maskgit/brats/stage2/in_context.yaml \
    --output-dir outputs/stage2_incontext
```

**Inference**:
```bash
# Generate T2 from T1 (modality translation)
prod9-train-transformer generate \
    --config configs/maskgit/brats/stage2/in_context.yaml \
    --checkpoint outputs/stage2_incontext/best.ckpt \
    --output outputs/generated \
    --num-samples 10 \
    --source-modality T1 \
    --target-modality T2
```

**Use cases**:
- **Modality translation**: Synthesize missing modalities (e.g., T1 → T2)
- **Few-shot generation**: Generate with limited examples
- **Multi-modal synthesis**: Combine multiple source modalities

**Reference**: See [`ARCHITECTURE.md`](ARCHITECTURE.md) for `ModalityProcessor` and `TransformerDecoderSingleStream` design.

### Algorithm-Lightning Architecture

The new algorithm-lightning separation enables testing training logic without Lightning overhead.

**For developers**: When adding new training algorithms, follow this three-step pattern:

**Step 1: Create algorithm class** (`training/algorithms/my_algorithm.py`)

```python
from typing import Dict, Any
import torch

class MyAlgorithmTrainer:
    """Pure training logic - no Lightning dependencies."""
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.Module):
        self.model = model
        self.loss_fn = loss_fn

    def compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pure PyTorch logic - testable without Lightning."""
        inputs = batch["input"]
        targets = batch["target"]
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        return {"loss": loss}

    def compute_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Metric computation - testable without Lightning."""
        return {"accuracy": (outputs == batch["target"]).float().mean()}
```

**Step 2: Create Lightning adapter** (`training/lightning/my_lightning.py`)

```python
import lightning as L
from training.algorithms.my_algorithm import MyAlgorithmTrainer

class MyLightningModule(L.LightningModule):
    """Thin wrapper for Lightning integration."""
    def __init__(self, algorithm_trainer: MyAlgorithmTrainer):
        super().__init__()
        self.algorithm = algorithm_trainer

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Delegate to algorithm
        losses = self.algorithm.compute_losses(batch)
        self.log("train_loss", losses["loss"])
        return losses["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        losses = self.algorithm.compute_losses(batch)
        metrics = self.algorithm.compute_metrics(batch, batch.get("output"))
        self.log_dict(metrics)
        self.log("val_loss", losses["loss"])
```

**Step 3: Test algorithm without Lightning** (`tests/unit/training/test_my_algorithm.py`)

```python
import pytest
import torch
from training.algorithms.my_algorithm import MyAlgorithmTrainer

def test_compute_losses():
    """Test pure logic - no Lightning overhead."""
    model = torch.nn.Linear(10, 5)
    loss_fn = torch.nn.MSELoss()
    algorithm = MyAlgorithmTrainer(model, loss_fn)

    batch = {
        "input": torch.randn(4, 10),
        "target": torch.randn(4, 5)
    }
    losses = algorithm.compute_losses(batch)

    assert "loss" in losses
    assert losses["loss"].item() >= 0

def test_compute_metrics():
    """Test metric computation."""
    algorithm = MyAlgorithmTrainer(
        torch.nn.Linear(10, 5),
        torch.nn.MSELoss()
    )

    batch = {"target": torch.tensor([0, 1, 2, 3])}
    outputs = torch.tensor([0, 1, 2, 3])

    metrics = algorithm.compute_metrics(batch, outputs)

    assert "accuracy" in metrics
    assert metrics["accuracy"].item() == 1.0
```

**Benefits**:
- **Faster testing**: No Lightning overhead in unit tests
- **Better modularity**: Clear separation of concerns
- **Easier debugging**: Test training logic in isolation
- **Code reuse**: Algorithms can be used outside Lightning

**Reference**: See [`ARCHITECTURE.md`](ARCHITECTURE.md) for algorithm-lightning pattern details.

---

## Adding New Components

### Adding a New Dataset

The centralized data module makes adding new datasets straightforward.

**Step 1: Create dataset class** in `src/prod9/data/datasets/my_dataset.py`:

```python
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path

class MyDataset(Dataset):
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        # Load data files
        self.files = list(self.data_dir.glob("*.nii.gz"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        # Load and preprocess sample
        img = nib.load(self.files[idx]).get_fdata()
        sample = {"image": img, "path": str(self.files[idx])}

        if self.transform:
            sample = self.transform(sample)

        return sample
```

**Step 2: Add builder method** in `src/prod9/data/builders.py`:

```python
from prod9.data.transforms import TransformBuilder
from prod9.data.datasets.my_dataset import MyDataset

class DatasetBuilder:
    # ... existing methods ...

    @staticmethod
    def build_my_dataset(config: dict, split: str):
        """Build MyDataset for training or validation."""
        from prod9.training.config import resolve_env_vars

        data_dir = resolve_env_vars(config["data"]["data_dir"])
        transform_builder = TransformBuilder()
        transform = transform_builder.build_my_transforms(config, split)

        dataset = MyDataset(data_dir, transform)

        # Split into train/val if needed
        if split == "train":
            train_size = int(0.8 * len(dataset))
            return torch.utils.data.Subset(dataset, range(train_size))
        else:  # validation
            train_size = int(0.8 * len(dataset))
            return torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
```

**Step 3: Add config** in `src/prod9/configs/maskgit/my_dataset/stage1/base.yaml`:

```yaml
data:
  data_dir: ${MY_DATA_DIR}  # Environment variable
  train_val_split: 0.8
  cache_rate: 1.0
  num_workers: 4

model:
  # Model architecture
```

**Step 4: Add tests** in `tests/unit/data/test_my_dataset.py`:

```python
import pytest
from prod9.data.builders import DatasetBuilder

def test_my_dataset_construction():
    """Test dataset can be built from config."""
    config = {
        "data": {
            "data_dir": "/path/to/data",
            "train_val_split": 0.8
        }
    }

    dataset = DatasetBuilder.build_my_dataset(config, "train")

    assert len(dataset) > 0

def test_my_dataset_transforms():
    """Test transforms are applied correctly."""
    # Test transform logic
    pass
```

**Reference**: See [`ARCHITECTURE.md`](ARCHITECTURE.md) for data module architecture.

### Adding a New CLI

Add new CLI entry points following the pattern in `src/prod9/cli/`.

**Structure**:
```python
# src/prod9/cli/my_feature.py
import typer
from pathlib import Path

app = typer.Typer(help="My feature CLI")

@app.command()
def train(
    config: Path = typer.Option(..., "--config", help="Path to config file"),
    output_dir: Path = typer.Option("outputs", "--output-dir", help="Output directory")
):
    """Train my feature."""
    # Load config
    # Build model
    # Train with Lightning
    pass

@app.command()
def validate(
    config: Path = typer.Option(..., "--config"),
    checkpoint: Path = typer.Option(..., "--checkpoint")
):
    """Validate my feature."""
    # Load checkpoint
    # Run validation
    pass

if __name__ == "__main__":
    app()
```

Register in `setup.cfg` or `pyproject.toml`:
```toml
[project.scripts]
prod9-my-feature = "prod9.cli.my_feature:app"
```

---

## Debugging and Optimization

### Common Training Issues

#### 1. OOM Errors During Training

**Symptoms**: `CUDA out of memory` or `MPS out of memory`

**Solutions**:
```yaml
# Reduce batch size
training:
  batch_size: 2  # Lower from 4 or 8

# Enable sliding window
sliding_window:
  enabled: true
  roi_size: [32, 32, 32]  # Smaller patches

# Reduce model size
model:
  hidden_dim: 256  # Lower from 512
  num_layers: 6    # Lower from 12
```

#### 2. Slow Training

**Symptoms**: Training takes much longer than expected

**Solutions**:
```yaml
# Check data loading
data:
  num_workers: 4  # Parallel data loading
  cache_rate: 1.0  # Cache data in memory
  pin_memory: true  # Faster GPU transfer

# Profile to find bottleneck
trainer:
  profiler: "simple"

# Reduce precision (if GPU supports)
trainer:
  precision: 16  # Mixed precision training
```

#### 3. Poor Convergence

**Symptoms**: Loss doesn't decrease or training is unstable

**Solutions**:
```yaml
# Adjust learning rate
training:
  optimizer:
    lr: 1e-4  # Try 1e-4 or 1e-5

# Check loss weights
loss:
  reconstruction_weight: 1.0
  perceptual_weight: 0.1  # Lower if unstable
  adversarial_weight: 0.01

# Enable gradient anomaly detection
trainer:
  detect_anomaly: true  # Debug mode (slow)
```

#### 4. Type Checking Errors

**Symptoms**: `pyright` reports type errors

**Solutions**:
```bash
# Run pyright to see exact errors
pyright

# Add type hints to fix issues
def my_function(x: int) -> str:
    return str(x)

# Use typing.cast() when needed
from typing import cast
result = cast(ExpectedType, some_function())
```

**Never use `# type: ignore`** - this is strictly forbidden.

### Performance Optimization

#### Quick Iteration Configuration

For rapid development cycles:

```yaml
trainer:
  max_epochs: 3                    # Quick test runs
  logging:
    limit_train_batches: 10         # Use only 10 batches per epoch
    limit_val_batches: 2            # Use only 2 val batches
  accelerator: "gpu"                # Use GPU if available
  devices: 1                        # Single GPU

data:
  num_workers: 4                    # Parallel data loading
  cache_rate: 1.0                   # Cache all data in memory
```

#### Production Configuration

For full training runs:

```yaml
trainer:
  max_epochs: 100
  logging:
    every_n_train_steps: 50         # Log every 50 steps
  accelerator: "gpu"
  devices: 1                        # Or use all GPUs
  gradient_clip_val: 1.0            # Clip gradients

data:
  num_workers: 8                    # More workers
  cache_rate: 1.0
  pin_memory: true                  # Faster GPU transfer

training:
  batch_size: 8                     # Maximize GPU utilization
```

---

## Testing Guidelines

### Test Organization

Tests are organized into three levels:

```
tests/
├── unit/              # Fast component tests (seconds)
│   ├── autoencoder/
│   ├── generator/
│   ├── training/
│   ├── data/
│   └── cli/
├── integration/       # Module interaction tests (minutes)
│   ├── test_cli_autoencoder.py
│   ├── test_cli_transformer.py
│   └── ...
└── system/            # End-to-end workflows (tens of minutes)
    ├── test_autoencoder_training.py
    ├── test_transformer_training.py
    └── ...
```

### Running Tests

```bash
# Run all tests
pytest

# Run by level
pytest tests/unit/           # Fast component tests
pytest tests/integration/    # Module interaction tests
pytest tests/system/         # Slow end-to-end tests

# Run specific test
pytest tests/unit/autoencoder/test_fsq.py

# With coverage
pytest --cov=prod9
pytest --cov-report=html     # HTML report in htmlcov/

# Verbose output
pytest -v -s
```

### Test Coverage Goals

- **Unit tests**: Fast component tests (seconds)
  - Test individual functions and classes
  - No external dependencies
  - High coverage of edge cases

- **Integration tests**: Module interaction (minutes)
  - Test component integration
  - Mock expensive operations
  - Verify data flow

- **System tests**: End-to-end workflows (tens of minutes)
  - Test complete training loops
  - Use real data (small samples)
  - Verify metrics and outputs

### New Algorithm Tests

When adding algorithm classes (following the algorithm-lightning pattern):

```python
# tests/unit/training/test_my_algorithm.py
import pytest
import torch
from training.algorithms.my_algorithm import MyAlgorithmTrainer

def test_compute_losses():
    """Test pure logic - no Lightning overhead."""
    model = torch.nn.Linear(10, 5)
    loss_fn = torch.nn.MSELoss()
    algorithm = MyAlgorithmTrainer(model, loss_fn)

    batch = {
        "input": torch.randn(4, 10),
        "target": torch.randn(4, 5)
    }
    losses = algorithm.compute_losses(batch)

    assert "loss" in losses
    assert losses["loss"].item() >= 0

def test_compute_metrics():
    """Test metric computation."""
    # Test implementation
    pass
```

**Benefits**:
- Fast execution (no Lightning overhead)
- Easy debugging (pure PyTorch)
- Better test coverage (test edge cases)

### Data Module Tests

Test dataset builders and transforms:

```python
# tests/unit/data/test_my_dataset.py
import pytest
from prod9.data.builders import DatasetBuilder

def test_my_dataset_construction():
    """Test dataset can be built from config."""
    config = {
        "data": {
            "data_dir": "/path/to/data",
            "train_val_split": 0.8
        }
    }

    dataset = DatasetBuilder.build_my_dataset(config, "train")

    assert len(dataset) > 0

def test_my_dataset_transforms():
    """Test transforms are applied correctly."""
    # Test transform logic
    pass
```

---

## Additional Resources

### Documentation

- [README.md](README.md) - User-facing overview and quickstart
- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed system architecture
- [CLAUDE.md](CLAUDE.md) - Complete project documentation for AI agents
- [REFACTORING.md](REFACTORING.md) - Architecture refactor details

### Configuration

- [configs/README.md](src/prod9/configs/README.md) - Configuration system overview
- [configs/maskgit/README.md](src/prod9/configs/maskgit/README.md) - MaskGiT config details
- [configs/maisi/README.md](src/prod9/configs/maisi/README.md) - MAISI config details

### Key Modules

- [`autoencoder/`](src/prod9/autoencoder/) - Stage 1: Autoencoder implementations
- [`generator/`](src/prod9/generator/) - Stage 2 & 3: MaskGiT generation
- [`controlnet/`](src/prod9/controlnet/) - MAISI Stage 3: ControlNet
- [`data/`](src/prod9/data/) - Centralized data handling
- [`training/`](src/prod9/training/) - Training infrastructure

### Development Commands

```bash
# Installation
pip install -e .[dev]

# Type checking
pyright

# Testing
pytest --cov=prod9

# Import organization
isort .

# Build package
pip install hatch
hatch build
```

---

**For questions or issues**, refer to the main [README.md](README.md) or [CLAUDE.md](CLAUDE.md).
