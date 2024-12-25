from torch import nn
import torch
from torch.nn.modules.module import T

from src.encoder.EncoderStack import EncoderStack
from src.encoder.PositionalEncoding import PositionalEncoding
from src.encoder.Masking import Masking

class TimeSeriesTransformer(nn.Module):
    """
    TimeSeriesTransformer is a neural network model for time series prediction.
    It consists of an encoder2 stack and a linear layer for regression.

    Attributes:
        input_size (int): The size of the input tensor.
        positional_encoding (PositionalEncoding): The positional encoding module.
        encoder_stack (EncoderStack): The stack of encoder2 blocks.
        regressor (nn.Linear): The linear layer for regression.
    """

    def __init__(self, input_size, head_sizes, mask):
        """
        Initializes the TimeSeriesTransformer model.

        Args:
            input_size (int): The size of the input tensor.
            head_sizes (List[List[int]]): The number of attention heads for each block in the encoder2 stack.
        """
        super(TimeSeriesTransformer, self).__init__()
        self.input_size = input_size
        self.head_sizes = head_sizes
        self.mask = mask
        self.positional_encoding = PositionalEncoding(self.input_size)
        self.encoder_stack = EncoderStack(input_size, head_sizes)
        self.regressor = nn.Linear(self.input_size, 1)

    def forward(self, x):
        """
        Forward pass for the TimeSeriesTransformer model.
        Positional encoding is applied to the input tensor, then passed through the encoder2 stack.
        The output is pooled and passed through the regressor.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The predicted output tensor.
        """
        masked = self.mask.mask(x)
        positional_encoded = self.positional_encoding(masked)
        encoded = self.encoder_stack(positional_encoded)
        pooled = encoded.mean(dim=-2)
        pred = self.regressor(pooled)

        return pred

    def train(self: T, mode: bool = True) -> T:
        self.mask.training = True
        return super().train(mode)

    def eval(self: T) -> T:
        self.mask.training = False
        return super().eval()


# Example usage
test_transformer = TimeSeriesTransformer(32, [[16, 16, 16] for _ in range(0, 3)])
test_tensor = torch.rand(8, 32)
print(test_transformer(test_tensor))