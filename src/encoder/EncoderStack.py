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
        input_size (int): The size of the input tensor.
        output_size (int): The size of the output tensor after applying the encoder2 blocks.
    """

    def __init__(self, input_size, head_sizes):
        """
        Initializes the MultiEncoder module.

        Args:
            input_size (int): The size of the input tensor.
            head_sizes (List[List[int]]): The number of attention heads for each block.
        """
        super().__init__()

        self.input_size = input_size
        self.encoder_blocks_list = [EncoderBlock(input_size, head_sizes) for head_sizes in head_sizes]
        self.encoder_stack = nn.Sequential(*self.encoder_blocks_list)

    def forward(self, x):
        """
        Forward pass for the MultiEncoder module.
        Applies the encoder2 blocks sequentially.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the encoder2 blocks.
        """
        return self.encoder_stack(x)
