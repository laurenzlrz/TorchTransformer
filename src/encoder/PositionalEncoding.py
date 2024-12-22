import torch.nn as nn
import torch

SCALE = 10000


class PositionalEncoding(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.col_arrangement = torch.arange(input_size)

    def encode_position(self, row_index, col_index):
        encoding = torch.where(col_index % 2 == 0,
                               torch.sin(torch.div(row_index, torch.pow(SCALE, torch.div(col_index, self.input_size)))),
                               torch.cos(
                                   torch.div(row_index, torch.pow(SCALE, torch.div((col_index - 1), self.input_size)))))
        return encoding

    def forward(self, x):
        row_arrangement = torch.arange(x.shape[-2])
        row_indices, col_indices = torch.meshgrid(row_arrangement, self.col_arrangement, indexing='ij')
        encoding = self.encode_position(row_indices, col_indices)
        encoding_added = torch.add(x, encoding)
        return encoding_added
