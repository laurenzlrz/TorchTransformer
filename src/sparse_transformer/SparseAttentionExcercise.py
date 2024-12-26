# Excercise to understand unsqueeze, broadcasting and einsum in PyTorch
# Goal is to implement the forward pass of the SparseAttention module
# Challenge is the mask of the keys and values, because they are different for each query

from torch import masked
import torch
import time

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

# '# measure time taken for matrix multiplication with or without unsqueeze and transpose
# start_time = time.time()
#
# '# Create a tensor with shape (10000, 10000)
# a = torch.rand(1000, 1000)
# b = torch.rand(1000, 1000)
# c = torch.matmul(a, b)
#
# end_time = time.time()
# print(f"Time taken: {end_time - start_time}")
#
# start_time = time.time()
#
# a = torch.rand(1000, 1, 1000)
# b = torch.rand(1000, 1000)
# c = torch.matmul(a, b).transpose(0,1).unsqueeze(0)
#
# end_time = time.time()
# print(f"Time taken: {end_time - start_time}")

# Test filtering instead of masking
k = 3000
a = torch.rand(k, k)
b = torch.rand(k, k)

start_time = time.time()
c = torch.matmul(a, b)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")
print(c.shape)

a = torch.rand(k, k)
b = torch.rand(k, k)
filter = [[i] for i in range(0,k)]

start_time = time.time()

filtered = a[filter, :].transpose(1,2)
b = b.unsqueeze(1)
d = torch.matmul(b, filtered).squeeze(2)

end_time = time.time()

print(f"Time taken: {end_time - start_time}")
print(d.shape)

# Test sucessful, filtered matrix multiplication is faster than masking


