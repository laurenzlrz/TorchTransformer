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

    def __init__(self, input_size, head_sizes, output_size, mask, task='regression'):
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
        if task == 'regression':
            self.output = RegressionModule(input_size, input_size, output_size)
        elif task == 'reconstruction':
            self.output = ReconstructionModule(input_size, output_size)
        else:
            raise ValueError(f'{task} is illegal argument for task')

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
        out = self.output(encoded)

        return out

    def train(self: T, mode: bool = True) -> T:
        self.mask.training = True
        return super().train(mode)

    def eval(self: T) -> T:
        self.mask.training = False
        return super().eval()


# Example usage
'''
test_transformer = TimeSeriesTransformer(32, [[16, 16, 16] for _ in range(0, 3)])
test_tensor = torch.rand(8, 32)
print(test_transformer(test_tensor))
'''
class ReconstructionModule(nn.Module):
    '''
    This mudule is built to reconstruct the full sequence length.
    For a sequence with dimension (sequence_length, input_size) the module
    will return an output of dimensions (sequence_length, output_size)
    '''
    def __init__(self, input_size, output_size):
        super(ReconstructionModule, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_size, output_size)
        )
    def forward(self, encoded):
        return self.module(encoded)

class RegressionModule(nn.Module):
    '''
    This module will take the encoded inputs and return a regression prediction.
    An input of dimensions (sequence_length, input_size) will turn into (output_size)
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(RegressionModule, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.regressor = nn.Linear(hidden_size*3, output_size)

    def forward(self, encoded):
        encoded = self.hidden(encoded)
        encoded_mean = encoded.mean(dim=-2)
        encoded_max, encoded_max_indices = encoded.max(dim=-2)
        encoded_max_indices = torch.div(encoded_max_indices, encoded.size(1))

        regressor_inputs = torch.concat([encoded_mean, encoded_max, encoded_max_indices], dim=-1)
        pred = self.regressor(regressor_inputs)

        return pred


