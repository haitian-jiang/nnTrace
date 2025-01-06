import torch

def initialize(
        model: torch.nn.Module, 
        prefix: str = "",
        mapping=None,
    ) -> torch.nn.Module:
    """Reinitialize a model with consistent random weights.
    
    Args:
        model (torch.nn.Module): A PyTorch model to initialize.
        prefix (str): Prefix for the parameter name.
    
    Returns:
        torch.nn.Module: The initialized model.
    """
    # for name, parameter in model.named_parameters():
        