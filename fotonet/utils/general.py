import torch

def check_device(device=None):
    """Choose a usable device without mutating global PyTorch runtime policy."""
    if device is not None:
        chosen = torch.device(device)
        if chosen.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if chosen.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return chosen
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_cudnn_benchmark(enabled):
    """Explicit opt-in for fixed-shape convolution throughput tuning."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(enabled)

