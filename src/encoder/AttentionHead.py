import math

import torch
import torch.nn as nn

from TensorCreation import TensorCreationMode, create_tensor, create_linear


class AttentionHead(nn.Module):
    """
    Expected Input Dimension: Batch x sequence length x input dimension
    Output Dimension: Batch x sequence length x hidden dimension
    """
    #TODO Dropout?
    def __init__(self, input_size, hidden_size, initialization=TensorCreationMode.UNIFORM, dropout_prob=0):
        super().__init__()
        self.scale_factor = 1.0 / math.sqrt(hidden_size)
        self.query_weight_linear_layer = nn.Linear(input_size, hidden_size)
        self.key_weight_linear_layer =  nn.Linear(input_size, hidden_size)
        self.value_weight_linear_layer =  nn.Linear(input_size, hidden_size)


    def forward(self, x):
        query_tensor = self.query_weight_linear_layer(x)
        key_tensor = self.key_weight_linear_layer(x)
        value_tensor = self.value_weight_linear_layer(x)

        attention_tensor = torch.matmul(query_tensor, transposeBatchTensor(key_tensor))

        scaled_attention_tensor = torch.mul(attention_tensor, self.scale_factor)

        softmax_attention_tensor = nn.functional.softmax(scaled_attention_tensor, dim=-1)

        output = torch.matmul(softmax_attention_tensor, value_tensor)

        return output


def transposeBatchTensor(tensor):
    return tensor.transpose(-2,-1)
