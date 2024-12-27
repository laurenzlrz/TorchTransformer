import torch

from src.sparse_transformer.hierachical_transformer.BackendTree import TreeNode, Tree


def create_tree(n: int):
    """
    Public method to build a tree of n leaves. It delegates the actual
    construction to the recursive `_build_level` helper function and
    returns the final list of leaf nodes.

    Args:
        n (int): The number of leaf nodes to create.

    Returns:
        List[TreeNode]: A list of TreeNode objects representing the leaves.
    """
    # Step 1: Create the bottom-most leaves.
    leaves = [TreeNode(i, []) for i in range(n)]

    # Step 2: Recursively link siblings and build parent levels
    #         until only one level remains.
    _build_level(leaves, start_index=n)

    node = leaves[0]
    parent = node.parent
    while parent is not None:
        node = node.parent
        parent = node.parent

    # The leaves themselves never change; we always return them as-is.
    return Tree(node)


def _build_level(current_level, start_index):
    """
    Recursively build one "parent level" above the current_level, linking
    siblings (left/right) and forming parents for pairs of nodes. Continues
    until there's only one node in the new level (the ultimate root).

    This function modifies `current_level` in place to add sibling links,
    then constructs a new list of parents. If that new list has more than
    one node, it recursively calls itself again.

    Args:
        current_level (List[TreeNode]): The current list of nodes (siblings).
        start_index (int): The next available integer index to assign
                           to newly created parent nodes.

    Returns:
        None. (The recursion stops when there's only one node at the
        newest level.)
    """
    level_size = len(current_level)
    # Base case: if there's only one node, no further levels to build.
    if level_size <= 1:
        return

    # 1) Link each node to its next neighbor (siblings).
    for i in range(level_size - 1):
        current_level[i].right = current_level[i + 1]
        current_level[i + 1].left = current_level[i]

    # 2) Build the parent layer from pairs in current_level.
    parents = []
    # We'll use a standard for-loop stepping by 2, to make pairs:
    for i in range(0, level_size, 2):
        left_child = current_level[i]
        # If there's a right child at i+1, pair them. Otherwise, only one child.
        if i + 1 < level_size:
            right_child = current_level[i + 1]
            new_parent = TreeNode(start_index, [left_child, right_child])
        else:
            new_parent = TreeNode(start_index, [left_child])
        parents.append(new_parent)
        start_index += 1

    # 3) Recursively build the level above this newly formed parent layer.
    _build_level(parents, start_index)


def create_surroundings_tensor(tree: Tree, elements_per_layer: int = 1):
    """
    Create a tensor representing the aggregated surroundings of each leaf node in the binary tree.

    Args:
        tree (BinaryTree): The binary tree containing the leaf nodes.
        elements_per_layer (int): The number of siblings to consider on each side per layer.

    Returns:
        torch.Tensor: A tensor where each row corresponds to the surroundings of a leaf node,
                      padded with -1 to ensure uniform length.
    """
    n = len(tree.get_number_of_leaves())
    surroundings_list = []

    # Collect surroundings for each leaf node
    for i in range(n):
        surroundings = tree.calculate_aggregated_surroundings(i, elements_per_layer)
        surroundings_list.append(surroundings)

    # Determine the maximum length of surroundings
    max_length = max(len(surroundings) for surroundings in surroundings_list)

    # Pad all surroundings to the same length (use -1 as padding value)
    padded_surroundings = [
        surroundings + [-1] * (max_length - len(surroundings))
        for surroundings in surroundings_list
    ]

    # Convert to PyTorch tensor
    tensor = torch.tensor(padded_surroundings, dtype=torch.int64)

    return tensor


# Example usage (test)
if __name__ == "__main__":
    tree = create_tree(23)
    result = tree.calculate_aggregated_surroundings(7, tree, elements_per_layer=1)
    print("Aggregated surroundings for node 7:", result)
    print(create_surroundings_tensor(tree, 1))
