import torch


def sanitize_tensor(tensor, replacement_val=0.):
    """Replace non-finite entries with replacement_val."""
    return torch.where(torch.isfinite(tensor), tensor, torch.full_like(tensor, replacement_val))
