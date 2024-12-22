import torch.nn as nn
import torch


# The scale factor for the positional encoding
SCALE = 10000


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding module injects information about the relative or absolute position of the tokens in the sequence.

    Attributes:
        input_size (int): The size of the input tensor.
        col_arrangement (Tensor): A tensor containing a range of column indices.
    """

    def __init__(self, input_size):
        """
        Initializes the PositionalEncoding module.

        Args:
            input_size (int): The size of the input tensor.
        """
        super().__init__()
        self.input_size = input_size

        # Dimensionality stays the same regardless the sequence length
        self.col_arrangement = torch.arange(input_size)

    def encode_position(self, row_index, col_index):
        """
        Encodes the position using a formula from my AI lecture.
        Differentiates between even and odd columns.

        Args:
            row_index (Tensor): A tensor containing row indices.
            col_index (Tensor): A tensor containing column indices.

        Returns:
            Tensor: A tensor containing the positional encoding.
        """
        encoding = torch.where(col_index % 2 == 0,
                               torch.sin(
                                   torch.div(row_index, torch.pow(SCALE, torch.div(col_index, self.input_size)))),
                               torch.cos(
                                   torch.div(row_index, torch.pow(SCALE, torch.div((col_index - 1), self.input_size)))))
        return encoding

    def forward(self, x):
        """
        Forward pass for the PositionalEncoding module.
        At first, a meshgrid is created from the row and column indices. The row indices depend on the sequence length.
        Then, the positional encoding is calculated and added to the input tensor

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The input tensor with positional encoding added.
        """
        row_arrangement = torch.arange(x.shape[-2])
        row_indices, col_indices = torch.meshgrid(row_arrangement, self.col_arrangement, indexing='ij')
        encoding = self.encode_position(row_indices, col_indices)
        encoding_added = torch.add(x, encoding)
        return encoding_added
