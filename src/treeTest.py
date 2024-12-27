class TreeNode:
    def __init__(self, index, left=None, right=None):
        self.index = index
        self.left = left
        self.right = right
        self.parent = None
        self.left_cousin = None
        self.right_cousin = None
        if left:
            left.parent = self
        if right:
            right.parent = self

    def is_right_child(self):
        return self.parent and self.parent.right == self

    def is_left_child(self):
        return self.parent and self.parent.left == self


class BinaryTree:
    def __init__(self, n):
        self.root = self._create_tree(n)

    def _create_tree(self, n):
        leaves = [TreeNode(i) for i in range(n)]
        current_level = leaves
        index = n
        while len(current_level) > 1:
            next_level = []
            for i in range(len(current_level)):
                if i > 0:
                    current_level[i].left_cousin = current_level[i - 1]
                if i < len(current_level) - 1:
                    current_level[i].right_cousin = current_level[i + 1]

                if i % 2 == 0:
                    left = current_level[i]
                    right = current_level[i + 1] if i + 1 < len(current_level) else None
                    parent = TreeNode(index, left, right)
                    next_level.append(parent)
                    index += 1
            current_level = next_level
        return current_level[0]

    def get_node(self, index):
        return self._find_node(self.root, index)

    def _find_node(self, node, index):
        if not node:
            return None
        if node.index == index:
            return node
        left_result = self._find_node(node.left, index)
        if left_result:
            return left_result
        return self._find_node(node.right, index)


def calculate_aggregated_surroundings(index, tree):
    surroundings = []
    element = tree.get_node(index)
    if not element:
        return surroundings

    add_left(element, surroundings)
    add_right(element, surroundings)
    return [node.index for node in surroundings]


def add_left(element, surroundings):
    if element.left_cousin is None:
        return
    left_element = element.left_cousin
    surroundings.insert(0, left_element)
    if left_element.is_right_child() and left_element.parent.left:
        surroundings.insert(0, left_element.parent.left)
    add_left(left_element.parent, surroundings)


def add_right(element, surroundings):
    if element.right_cousin is None:
        return
    right_element = element.right_cousin
    surroundings.append(right_element)
    if right_element.is_left_child() and right_element.parent.right:
        surroundings.append(right_element.parent.right)
    add_right(right_element.parent, surroundings)


# Testfunktion
def test_algorithm():
    tree = BinaryTree(16)  # Ein Baum mit 8 Blättern (vollständig bis zur Ebene 3)
    result = calculate_aggregated_surroundings(3, tree)  # Test mit Knoten 3
    print("Surroundings of node 3:", result)


# Test ausführen
test_algorithm()
