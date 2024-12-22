from encoder.AttentionHead import AttentionHead
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    Performs Multihead Attention using the Attention Head class. Takes inputs of input_size
    returns outputs of output_size
    """

    def __init__(self, input_size, output_size, head_sizes):
        super(MultiHeadAttention, self).__init__()
        self.head_sizes = head_sizes
        self.input_size = input_size
        self.num_heads = len(self.head_sizes)
        self.attention_heads = nn.ModuleList([AttentionHead(input_size, size) for size in self.head_sizes])
        self.sum_hidden_dim = torch.sum(torch.tensor(head_sizes), dtype=torch.int32).item()
        self.weights_rescale = nn.Linear(self.sum_hidden_dim, output_size)

    def forward(self, x):
        outputs = [attention_head(x) for attention_head in self.attention_heads]
        concat_output = torch.concat(outputs, dim=-1)
        rescaled_output = self.weights_rescale(concat_output)
        return rescaled_output

