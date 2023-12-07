import torch
from torch import nn


class SingleLinearProbe(nn.Module):
    """A single-layer neural probe performing an affine transformation.
    The dimension of the input remains the same.

    Args:
        dim_input (int): The dimension of a single input embedding.
    """
    def __init__(self, dim_input: int) -> None:
        super().__init__()
        self._linear = nn.Linear(dim_input, dim_input)
    
    def __str__(self) -> str:
        return "single_linear_probe"

    def forward(self, cls_repr) -> torch.Tensor:
        """Forward pass through the module.

        Args:
            cls_repr (torch.Tensor): A batch of input embeddings of shape
                (batch_size, dim_input).

        Returns:
            torch.Tensor: A transformed batch of input embeddings of shape (batch_size, dim_input).
        """
        x = self._linear(cls_repr)
        return x


class DoubleLinearProbe(nn.Module):
    """A double-layer neural probe performing a non-linear transformation.
    The dimension of the input embedding may change during the transformation
    through the `dim_output` parameter. However, the result of the transformation
    must have the same dimension as the input.

    Args:
        dim_input (int): The dimension of a single input embedding.
        dim_output (int): The dimension to which the input embedding is mapped to.
    """
    def __init__(self, dim_input: int, dim_output: int) -> None:
        super().__init__()
        self._linear_stack = nn.Sequential(
            nn.Linear(dim_input, dim_output),
            nn.ReLU(),
            nn.Linear(dim_output, dim_input)
        )

    def __str__(self) -> str:
        return "double_linear_probe"

    def forward(self, cls_repr) -> torch.Tensor:
        """Forward pass through the module.

        Args:
            cls_repr (torch.Tensor): A batch of input embeddings of shape
                (batch_size, dim_input).

        Returns:
            torch.Tensor: A transformed batch of input embeddings of shape (batch_size, dim_input).
        """
        x = self._linear_stack(cls_repr)
        return x
