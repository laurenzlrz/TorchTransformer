import torch
import torch.nn as nn

from src.encoder.MultiHeadAttention import MultiHeadAttention


class EncoderBlock(nn.Module):
    """
    EncoderBlock module for the Transformer architecture.
    Consists of a multi-head attention mechanism and a feed-forward network.

    Attributes:
        input_size (int): The size of the input tensor.
        head_sizes (List[int]): The number of attention heads.

        multihead_attention (MultiHeadAttention): The multi-head attention mechanism.
        feed_forward (nn.Sequential): The feed-forward network.
        layernorm_attention (nn.LayerNorm): Layer normalization after the attention mechanism.
        layer-norm_feed_forward (nn.LayerNorm): Layer normalization after the feed-forward network.
    """

    def __init__(self, input_size, head_sizes):
        """
        Initializes the EncoderBlock module.

        Args:
            input_size (int): The size of the input tensor.
            head_sizes (List[int]): The number of attention heads.
        """
        super().__init__()
        self.input_size = input_size
        self.head_sizes = head_sizes

        # TODO
        """
        if output_size_attention and output_size_ffn is None:
            self.output_size_ffn = output_size
            self.output_size_attention = output_size
        else:
            self.output_size_ffn = output_size_ffn
        """

        # Because of residual connections, the output size of the attention mechanism must be the same as the input size
        self.multihead_attention = MultiHeadAttention(self.input_size, self.input_size, self.head_sizes)
        self.feed_forward = nn.Sequential(nn.Linear
                                          (self.input_size, self.input_size), nn.ReLU())
        self.layernorm_attention = nn.LayerNorm(normalized_shape=self.input_size)
        self.layernorm_feed_forward = nn.LayerNorm(normalized_shape=self.input_size)

    def forward(self, x):
        """
        Forward pass for the EncoderBlock module.
        Applies the multi-head attention, then summed with residual connection, followed by layer normalization.
        Then, the output is fed through a feed-forward network, and again layer normalized.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the multi-head attention, residual connection,
            feed-forward network, and layer normalization.
        """
        residual_attention = x
        multihead_attention_output = self.multihead_attention(x)
        multihead_residual_output = torch.add(multihead_attention_output, residual_attention)
        normalized_multihead_residual_output = self.layernorm_attention(multihead_residual_output)

        feed_forward_output = self.feed_forward(normalized_multihead_residual_output)
        feed_forward_residual_output = torch.add(feed_forward_output, normalized_multihead_residual_output)
        normed_feed_forward_residual_output = self.layernorm_feed_forward(feed_forward_residual_output)

        return normed_feed_forward_residual_output
