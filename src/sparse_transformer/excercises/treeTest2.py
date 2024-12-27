import torch


class BinaryTree:
    """
    BinaryTree is a structure that builds a tree of n leaves, then successively
    combines pairs of nodes into parents until only one level remains (the root
    level). Each node can point to its immediate left and right siblings, as
    well as its parent (if any). The final tree ensures that all leaves are at
    the bottom level, with internal parent nodes formed by pairing children.

    Attributes:
        leaves (List[TreeNode]): A list of the bottom-most TreeNode objects.
                                 These represent the "leaves" of the tree.
    """

    def __init__(self, n: int):
        """
        Initialize a new BinaryTree with n leaves. Each leaf node will have an
        index from 0 to n-1, inclusive.

        Args:
            n (int): The number of leaf nodes to create initially.
        """
        self.leaves = self.create_tree(n)

    def create_tree(self, n: int):
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
        self._build_level(leaves, start_index=n)

        # The leaves themselves never change; we always return them as-is.
        return leaves

    def _build_level(self, current_level, start_index):
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
        self._build_level(parents, start_index)

    def get_element(self, index: int):
        """
        Retrieve a leaf node by its index (0 <= index < len(leaves)).

        Args:
            index (int): The index of the leaf node to retrieve.

        Returns:
            TreeNode or None: The leaf node if index is valid; otherwise None.
        """
        if 0 <= index < len(self.leaves):
            return self.leaves[index]
        return None


class TreeNode:
    """
    A node in the binary tree structure that can contain either:
      - no children (leaf),
      - one child, or
      - two children.

    Additionally, each node keeps track of its left and right siblings
    (if any), and its parent node (if any).
    """

    def __init__(self, index: int, childs: list):
        """
        Create a new TreeNode.

        Args:
            index (int): The integer index assigned to this node.
            childs (list): A list of TreeNode children (either empty, 1, or 2).
        """
        self.index = index
        self.childs = childs  # sub-nodes (either 0, 1, or 2)
        self.parent = None  # reference to parent TreeNode
        self.left = None  # reference to the immediate left sibling
        self.right = None  # reference to the immediate right sibling

        # Link each child back to this node as its parent.
        for child in childs:
            child.parent = self

    def is_rightest_child(self) -> bool:
        """
        Check if this node is the right-most child of its parent.

        Returns:
            bool: True if this node is the last child in parent's child list;
                  False otherwise or if no parent exists.
        """
        return self.parent and self.parent.childs[-1] == self

    def is_leftest_child(self) -> bool:
        """
        Check if this node is the left-most child of its parent.

        Returns:
            bool: True if this node is the first child in parent's child list;
                  False otherwise or if no parent exists.
        """
        return self.parent and self.parent.childs[0] == self

    def get_node(self, index: int):
        """
        Recursively search for a node with a given index in the subtree
        rooted at this node.

        Args:
            index (int): The index to look for.

        Returns:
            TreeNode or None: The node with the given index, if found.
                              Otherwise, None.
        """
        if self.index == index:
            return self
        if not self.childs:
            return None
        for child in self.childs:
            found = child.get_node(index)
            if found:
                return found
        return None


def calculate_aggregated_surroundings(index: int, tree: BinaryTree, elements_per_layer: int = 1):
    """
    Given the index of a leaf node in the tree, collect the "surroundings"
    (neighbors to the left and right, and their parents above them),
    aggregated recursively up the tree.

    Args:
        index (int): Index of the starting leaf node in `tree`.
        tree (BinaryTree): The BinaryTree object containing the node of interest.
        elements_per_layer (int): How many siblings to grab on each side
                                  before moving up to the parent.

    Returns:
        List[int]: A list of node indices in left-to-right order representing
                   the collected surroundings of the node at `index`.
    """
    element = tree.get_element(index)
    if element is None:
        return []

    surroundings = []
    _add_left(element, surroundings, elements_per_layer)
    _add_right(element, surroundings, elements_per_layer)
    return [node.index for node in surroundings]


def _add_left(element: TreeNode, surroundings: list, elements_per_layer: int):
    """
    Recursively collect the left surroundings for a given node.
    This involves:
        1. Taking up to `elements_per_layer - 1` immediate left siblings.
        2. Going all the way left until this node is the leftest child
           in its parent.
        3. Repeating the same logic on the parent (going upward).

    Args:
        element (TreeNode): The node whose left side we are collecting.
        surroundings (List[TreeNode]): A list to which the found nodes
                                       will be added at the front.
        elements_per_layer (int): The number of siblings to take per layer.
    """
    if element.left is None:
        return

    # 1) Collect immediate left siblings (up to elements_per_layer - 1).
    left_neighbor = element.left
    for _ in range(elements_per_layer - 1):
        surroundings.insert(0, left_neighbor)
        if left_neighbor.left is None:
            return  # No further left neighbors
        left_neighbor = left_neighbor.left

    # 2) Recursively go all the way left within the same parent.
    left_neighbor = _go_all_the_way_left(left_neighbor, surroundings)

    # 3) Move up one layer (to the parent) and repeat.
    _add_left(left_neighbor.parent, surroundings, elements_per_layer)


def _add_right(element: TreeNode, surroundings: list, elements_per_layer: int):
    """
    Recursively collect the right surroundings for a given node.
    This involves:
        1. Taking up to `elements_per_layer - 1` immediate right siblings.
        2. Going all the way right until this node is the rightest child
           in its parent.
        3. Repeating the same logic on the parent (going upward).

    Args:
        element (TreeNode): The node whose right side we are collecting.
        surroundings (List[TreeNode]): A list to which the found nodes
                                       will be appended.
        elements_per_layer (int): The number of siblings to take per layer.
    """
    if element.right is None:
        return

    # 1) Collect immediate right siblings (up to elements_per_layer - 1).
    right_neighbor = element.right
    for _ in range(elements_per_layer - 1):
        surroundings.append(right_neighbor)
        if right_neighbor.right is None:
            return  # No further right neighbors
        right_neighbor = right_neighbor.right

    # 2) Recursively go all the way right within the same parent.
    right_neighbor = _go_all_the_way_right(right_neighbor, surroundings)

    # 3) Move up one layer (to the parent) and repeat.
    _add_right(right_neighbor.parent, surroundings, elements_per_layer)


def _go_all_the_way_left(node: TreeNode, surroundings: list) -> TreeNode:
    """
    Recursively move left (via node.left) until we reach the left-most child
    in its parent. Along the way, insert each encountered node at the front
    of the surroundings list.

    Args:
        node (TreeNode): The starting node for left traversal.
        surroundings (List[TreeNode]): The list into which visited nodes
                                       will be inserted at the front.

    Returns:
        TreeNode: The final left-most node encountered.
    """
    # If this is the left-most child, insert and return it.
    if node.is_leftest_child():
        surroundings.insert(0, node)
        return node

    # Otherwise, insert the current node and move further left.
    surroundings.insert(0, node)
    return _go_all_the_way_left(node.left, surroundings)


def _go_all_the_way_right(node: TreeNode, surroundings: list) -> TreeNode:
    """
    Recursively move right (via node.right) until we reach the right-most child
    in its parent. Along the way, append each encountered node to the end
    of the surroundings list.

    Args:
        node (TreeNode): The starting node for right traversal.
        surroundings (List[TreeNode]): The list into which visited nodes
                                       will be appended.

    Returns:
        TreeNode: The final right-most node encountered.
    """
    # If this is the right-most child, append and return it.
    if node.is_rightest_child():
        surroundings.append(node)
        return node

    # Otherwise, append the current node and move further right.
    surroundings.append(node)
    return _go_all_the_way_right(node.right, surroundings)


def create_surroundings_tensor(tree: BinaryTree, elements_per_layer: int = 1):
    """
    Create a tensor representing the aggregated surroundings of each leaf node in the binary tree.

    Args:
        tree (BinaryTree): The binary tree containing the leaf nodes.
        elements_per_layer (int): The number of siblings to consider on each side per layer.

    Returns:
        torch.Tensor: A tensor where each row corresponds to the surroundings of a leaf node,
                      padded with -1 to ensure uniform length.
    """
    n = len(tree.leaves)
    surroundings_list = []

    # Collect surroundings for each leaf node
    for i in range(n):
        surroundings = calculate_aggregated_surroundings(i, tree, elements_per_layer)
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
    tree = BinaryTree(23)
    result = calculate_aggregated_surroundings(7, tree, elements_per_layer=1)
    print("Aggregated surroundings for node 7:", result)
    print(create_surroundings_tensor(tree, 1))
