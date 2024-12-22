from encoder.EncoderBlock import EncoderBlock
from encoder.PositionalEncoding import PositionalEncoding
from torch import nn
import torch

class TimeSeriesTransformer(nn.Module):
    #TODO Variable AttentionHeads und Anzahl Encoder Blocks
    def __init__(self, input_size, head_sizes, num_blocks, output_size_attention):

        super(TimeSeriesTransformer, self).__init__()
        self.input_size = input_size
        self.encoder_blocks = nn.ModuleList([EncoderBlock(input_size, head_sizes, output_size_attention, input_size) for _ in range(num_blocks)])
        self.encoder = nn.Sequential(*self.encoder_blocks)
        self.regressor = nn.Linear(input_size, 1)
        self.positional_encoding = PositionalEncoding(self.input_size)

    def forward(self, x):
        positional_encoded = self.positional_encoding(x)
        encoded = self.encoder(positional_encoded)
        pooled = encoded.mean(dim=-2)
        pred = self.regressor(pooled)

        return pred

test_transformer = TimeSeriesTransformer(input_size=32, head_sizes=[16,16,16], num_blocks=2, output_size_attention=32)

test_tensor = torch.rand(8, 32)

print(test_transformer(test_tensor))