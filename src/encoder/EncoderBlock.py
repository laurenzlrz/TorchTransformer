from encoder.MultiHeadAttention import MultiHeadAttention
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, input_size, head_sizes, output_size_attention, output_size_ffn):
        super().__init__()
        self.input_size = input_size
        self.output_size_attention = output_size_attention
        self.head_sized = head_sizes
        self.output_size_ffn = output_size_ffn

        # TODO
        """
        if output_size_attention and output_size_ffn is None:
            self.output_size_ffn = output_size
            self.output_size_attention = output_size
        else:
            self.output_size_ffn = output_size_ffn
        """

        self.multihead_attention = MultiHeadAttention(input_size, output_size_attention, head_sizes)
        self.ffn = nn.Sequential(nn.Linear(self.output_size_attention, self.output_size_ffn),
                                 nn.ReLU())
        self.layernorm_attention = nn.LayerNorm(normalized_shape=self.output_size_attention)
        self.layernorm_ffn = nn.LayerNorm(normalized_shape=self.output_size_ffn)

    def forward(self, x):
        residual_attention = x
        out_attention = self.multihead_attention(x)
        out_attention += residual_attention
        out1 = self.layernorm_attention(out_attention)

        residual_ffn = out1
        out_ffn = self.ffn(out1)
        out_ffn += residual_ffn
        out2 = self.layernorm_ffn(out_ffn)

        return out2
