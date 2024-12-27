# This file is a part of the Hierarchical Transformer project.

class TreeNode:
    """
    A node in the binary tree structure that can contain either:
      - no children (leaf),
      - children (internal node),

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

    def get_number_of_leaves(self) -> int:
        if self.childs.empty():
            return 1
        return sum([child.get_number_of_leaves() for child in self.childs])

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

class Tree:

    def __init__(self, root: TreeNode):
        self.root = root

    def get_number_of_leaves(self):
        self.root.get_number_of_leaves()

    def calculate_aggregated_surroundings(self, index: int, elements_per_layer: int = 1):
        """
        Given the index of a leaf node in the tree, collect the "surroundings"
        (neighbors to the left and right, and their parents above them),
        aggregated recursively up the tree.

        Args:
            index (int): Index of the starting leaf node in `tree`.
            elements_per_layer (int): How many siblings to grab on each side
                                      before moving up to the parent.

        Returns:
            List[int]: A list of node indices in left-to-right order representing
                       the collected surroundings of the node at `index`.
        """
        element = self.root.get_node(index)
        if element is None:
            return []

        surroundings = []
        self._add_left(element, surroundings, elements_per_layer)
        self._add_right(element, surroundings, elements_per_layer)
        return [node.index for node in surroundings]

    def _add_left(self, element: TreeNode, surroundings: list, elements_per_layer: int):
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
        left_neighbor = self._go_all_the_way_left(left_neighbor, surroundings)

        # 3) Move up one layer (to the parent) and repeat.
        self._add_left(left_neighbor.parent, surroundings, elements_per_layer)

    def _add_right(self, element: TreeNode, surroundings: list, elements_per_layer: int):
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
        right_neighbor = self._go_all_the_way_right(right_neighbor, surroundings)

        # 3) Move up one layer (to the parent) and repeat.
        self._add_right(right_neighbor.parent, surroundings, elements_per_layer)

    def _go_all_the_way_left(self, node: TreeNode, surroundings: list) -> TreeNode:
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
        return self._go_all_the_way_left(node.left, surroundings)

    def _go_all_the_way_right(self, node: TreeNode, surroundings: list) -> TreeNode:
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
        return self._go_all_the_way_right(node.right, surroundings)