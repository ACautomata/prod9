# Refactoring Report: Decoupling Lightning Modules

## Overview
The PyTorch Lightning modules in `prod9` have been refactored to separate business logic (algorithms) from framework boilerplate (Lightning glue) and data processing.

## Architecture
The system now follows a 3-layer architecture:

### 1. Business Logic Layer (`prod9.training.algorithms`)
Contains pure PyTorch implementation of training loops, loss calculations, and sampling algorithms. These classes are independent of PyTorch Lightning and can be unit-tested using standard PyTorch tensors and mocks.
- `AutoencoderTrainer`
- `TransformerTrainer`
- `VAEGANTrainer`
- `DiffusionTrainer`
- `ControlNetTrainer`

### 2. Lightning Adapter Layer (`prod9.training.lightning`)
Contains thin Lightning wrappers that delegate to the business logic layer. They handle Lightning-specific concerns like logging, manual optimization, and lifecycle hooks.
- `AutoencoderLightning`
- `TransformerLightning`
- `MAISIVAELightning`
- `MAISIDiffusionLightning`
- `ControlNetLightning`

### 3. Data Processing Layer (`prod9.data`)
Contains decoupled datasets, transforms, and builders.
- `DatasetBuilder`: Centralized factory for creating datasets.
- `TransformBuilder`: Centralized factory for creating MONAI transform pipelines.
- `PreEncoder`: Logic for pre-encoding data for Stage 2 training.

## Backward Compatibility
The existing modules in `src/prod9/training/` (e.g., `autoencoder.py`, `transformer.py`, `brats_data.py`) have been kept as shims. They maintain the same public API and constructor arguments, but delegate their implementation to the new layers. Existing CLI commands and YAML configurations remain fully functional.

## Benefits
- **Testability**: Business logic can now be unit-tested in isolation without a complex Lightning Trainer mock.
- **Maintainability**: Reduced class sizes and clear separation of concerns.
- **Flexibility**: Easier to swap components or use the business logic in different frameworks.

## Verification
- All existing unit tests pass.
- New unit tests added for all business logic trainers in `prod9.training.algorithms`.
- Comprehensive integration tests using `LightningTestHarness` added in `tests/integration/` to verify:
  - `AutoencoderLightning` / `TransformerLightning` (FSQ)
  - `MAISIVAELightning` / `MAISIDiffusionLightning` / `ControlNetLightning` (MAISI)
- All Lightning modules correctly delegate to standalone Trainers while maintaining backward compatibility.
- Type safety enforced across the new architecture (no `# type: ignore`).
