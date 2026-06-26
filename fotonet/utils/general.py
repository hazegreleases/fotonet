import torch

def check_device():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        # cuDNN benchmark can spend many seconds autotuning the first new input
        # shape on Windows/RTX setups. Heuristic selection is much better for
        # low-latency inference and short epoch_cut tuning trials.
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

