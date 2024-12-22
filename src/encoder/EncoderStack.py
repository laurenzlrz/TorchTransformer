from torch import nn

from src.encoder.EncoderBlock import EncoderBlock

INVALID_BLOCK_SIZES_MSG = ("The number of attention heads, "
                           "feed-forward output sizes, and attention output sizes must be the same.")


class EncoderStack(nn.Module):
    """
    MultiEncoder module is a stack of EncoderBlock modules. It applies the EncoderBlock module multiple times.

    Attributes:
        encoder_blocks_list (List[EncoderBlock]): List of EncoderBlock instances.
        encoder_stack (nn.ModuleList): ModuleList containing the EncoderBlock instances.
    """

    def __init__(self, input_size, head_sizes, attention_output_sizes, feed_forward_output_sizes):
        """
        Initializes the MultiEncoder module.

        Args:
            input_size (int): The size of the input tensor.
            head_sizes (List[List[int]]): The number of attention heads for each block.
            attention_output_sizes (List[int]): The output sizes of the attention mechanism for each block.
            feed_forward_output_sizes (List[int]): The output sizes of the feed-forward network for each block.
        """
        super().__init__()

        # Check if the number of attention heads, feed-forward output sizes, and attention output sizes are the same
        if len(attention_output_sizes) != len(feed_forward_output_sizes):
            raise ValueError(INVALID_BLOCK_SIZES_MSG)
        if len(head_sizes) != len(attention_output_sizes):
            raise ValueError(INVALID_BLOCK_SIZES_MSG)

        # Initialize the encoder blocks, for each block the input size is the output size of the previous block
        self.encoder_blocks_list = []
        block_input_size = input_size
        for head_sizes, attention_output_size, feed_forward_output_size in zip(head_sizes, attention_output_sizes,
                                                                               feed_forward_output_sizes):
            self.encoder_blocks_list.append(EncoderBlock(block_input_size, head_sizes, attention_output_size,
                                                         feed_forward_output_size))
            block_input_size = feed_forward_output_size

        self.encoder_stack = nn.Sequential(*self.encoder_blocks_list)

    def forward(self, x):
        """
        Forward pass for the MultiEncoder module.
        Applies the encoder blocks sequentially.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the encoder blocks.
        """
        return self.encoder_stack(x)
