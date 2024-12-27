from torch import nn
import torch.nn.functional as F

from src.encoder.EncoderStack import EncoderStack

class TreeTransformer(nn.Module):

    def __init__(self, recurrent_encoder: EncoderStack, output_encoder: EncoderStack, block_size: int, pool_size: int):
        super().__init__()
        self.recurrent_encoder = recurrent_encoder
        self.output_encoder = output_encoder
        self.pool_size = pool_size
        self.block_size = block_size
        self.tree = None

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        while seq_len >= self.d:
            x = self.apply_transformer_in_blocks(x)
            x = self.pool(x)
            seq_len = x.size(1)

        return self.output_encoder(x)

    def apply_transformer_in_blocks(self, x):
        batch_size, seq_len, dim = x.size()

        # Padding
        if seq_len % self.block_size != 0:
            pad_len = self.block_size - (seq_len % self.block_size)
            x = F.pad(x, (0, 0, 0, pad_len), value=0)
            seq_len = x.size(1)

        x = x.view(batch_size, seq_len // self.block_size, self.block_size, dim)
        x = x.permute(0, 2, 1, 3).contiguous().view(-1, self.block_size, dim)

        x = self.transformer_block(x)

        x = x.view(batch_size, -1, self.block_size, dim).permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, dim)

        return x

    def pool(self, x):
        x = F.adaptive_avg_pool1d(x.permute(0, 2, 1), self.pool_size).permute(0, 2, 1)
        return x
