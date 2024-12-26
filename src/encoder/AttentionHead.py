import math

import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    # TODO Dropout

    """
    AttentionHead class implements a single attention head for the multi-head attention mechanism.

    Attributes:
        scale_factor (float): Scaling factor for the attention scores.
        query_weight_linear_layer (nn.Linear): Linear layer to project input to query tensor.
        key_weight_linear_layer (nn.Linear): Linear layer to project input to key tensor.
        value_weight_linear_layer (nn.Linear): Linear layer to project input to value tensor.
    """

    def __init__(self, input_size, hidden_size, dropout_prob=0):
        """
        Initializes the AttentionHead.

        Args:
            input_size (int): Dimension of the input tensor.
            hidden_size (int): Dimension of the hidden layer.
            dropout_prob (float): Dropout probability (currently unused).
        """
        super().__init__()
        self.scale_factor = 1.0 / math.sqrt(hidden_size)
        self.query_weight_linear_layer = nn.Linear(input_size, hidden_size)
        self.key_weight_linear_layer = nn.Linear(input_size, hidden_size)
        self.value_weight_linear_layer = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        """
        Perform the forward pass of the attention head.
        Each row is one element of the sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, sequence length, input dimension).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, sequence length, hidden dimension).
        """
        query_tensor = self.query_weight_linear_layer(x)
        key_tensor = self.key_weight_linear_layer(x)
        value_tensor = self.value_weight_linear_layer(x)

        attention_tensor = torch.matmul(query_tensor, transposeBatchTensor(key_tensor))

        # Scale the attention tensor for numerical stability
        scaled_attention_tensor = torch.mul(attention_tensor, self.scale_factor)

        # Softmax applied for each row vector (each element of the sequence)
        softmax_attention_tensor = nn.functional.softmax(scaled_attention_tensor, dim=-1)

        output = torch.matmul(softmax_attention_tensor, value_tensor)

        return output


def transposeBatchTensor(tensor):
    """
    Transpose the last two dimensions of the tensor.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Transposed tensor.
    """
    return tensor.transpose(-2, -1)
