import torch


def get_device() -> torch.device:
    """
    Get the best available device for computation.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def resolve_device(device_config: str | None) -> torch.device:
    """Resolve device from config or auto-detect."""
    if device_config is not None:
        return torch.device(device_config)
    return get_device()
