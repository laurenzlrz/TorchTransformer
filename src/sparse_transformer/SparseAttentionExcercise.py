# Excercise to understand unsqueeze, broadcasting and einsum in PyTorch
# Goal is to implement the forward pass of the SparseAttention module
# Challenge is the mask of the keys and values, because they are different for each query

import torch


tensor = torch.linspace(1, 6, steps=6).view(2, 3)

a = torch.ones(2)
b = 0 * torch.ones(2)
c = torch.cat((a.unsqueeze(0),b.unsqueeze(0)), dim=0).unsqueeze(1)


row_filter = torch.tensor([[1,2], [0,2]])
filtered = tensor[:, row_filter].permute(1,0,2)
print(tensor)
print(filtered)


print(c)

print(torch.matmul(c, filtered).squeeze(1))

# End of practice