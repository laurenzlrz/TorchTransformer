# Excercise to understand unsqueeze, broadcasting and einsum in PyTorch
# Goal is to implement the forward pass of the SparseAttention module
# Challenge is the mask of the keys and values, because they are different for each query

import torch
import time

# Test filtering and transposing
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

# Test create filter for each query

def select_nodes(i, k, num_elements):
    """
    Selects a subset of nodes according to the specified rules.

    Args:
        i (int): The index of the current node.
        k (int): The number of neighboring elements.
        num_elements (int): The total number of elements in the tree.

    Returns:
        List[int]: A list of indices of the selected nodes.
    """
    selected_indices = []
    current_layer = 1
    current_index = i

    while current_index < num_elements:
        # Calculate the range of neighboring nodes to be selected
        start_index = max(0, current_index - k)
        end_index = min(current_index + k + 1, num_elements)

        # Select the neighboring nodes
        selected_indices.extend(range(start_index, end_index))

        # Skip the next 2k neighboring nodes and move to the next layer
        current_index = end_index + 2 * k
        current_layer += 1
        k *= 2

    # Remove duplicates and sort the indices
    selected_indices = sorted(set(selected_indices))

    return selected_indices

# Example usage
i = 3  # Current node index
k = 2  # Number of neighboring elements
num_elements = 15  # Total number of elements in the tree

selected_nodes = select_nodes(i, k, num_elements)
print(selected_nodes)


# Tree  



