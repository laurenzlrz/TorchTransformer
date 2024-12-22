import torch
import torch.nn as nn

from src.encoder.AttentionHead import AttentionHead

class MultiHeadAttention(nn.Module):
    """
    Performs Multi-head Attention using the Attention Head class. Takes inputs of input_size
    and returns outputs of output_size.

    Attributes:
        head_sizes (list): List of hidden sizes for each attention head.
        input_size (int): Dimension of the input tensor.
        attention_heads (nn.ModuleList): List of AttentionHead instances.
        sum_hidden_dim (int): Sum of all hidden dimensions of the attention heads.
        weights_rescale (nn.Linear): Linear layer to rescale the concatenated output.
    """

    def __init__(self, input_size, output_size, head_sizes):
        """
        Initializes the MultiHeadAttention.

        Args:
            input_size (int): Dimension of the input tensor.
            output_size (int): Dimension of the output tensor.
            head_sizes (list[int]): List of hidden sizes for each attention head.
        """
        super(MultiHeadAttention, self).__init__()
        self.head_sizes = head_sizes
        self.input_size = input_size
        self.sum_hidden_dim = sum(self.head_sizes)

        self.attention_heads = nn.ModuleList([AttentionHead(input_size, size) for size in self.head_sizes])

        self.weights_rescale = nn.Linear(self.sum_hidden_dim, output_size)

    def forward(self, x):
        """
        Perform the forward pass of the multi-head attention.
        Calculates the output of each attention head and concatenates them.
        Afterward, the concatenated output is rescaled with a linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, sequence length, input dimension).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, sequence length, output dimension).
        """
        outputs = [attention_head(x) for attention_head in self.attention_heads]
        concat_output = torch.concat(outputs, dim=-1)
        rescaled_output = self.weights_rescale(concat_output)
        return rescaled_output