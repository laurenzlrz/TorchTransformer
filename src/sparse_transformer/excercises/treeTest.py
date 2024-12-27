class BinaryTree:
    def __init__(self, n):
        self.leaves = self.create_tree(n)

    def create_tree(self, n):
        leaves = [TreeNode(i, []) for i in range(n)]
        current_level = leaves
        index = n

        while len(current_level) > 1:
            next_level = []
            i = 0
            while i < len(current_level):
                left_child = current_level[i]
                right_child = current_level[i + 1] if i + 1 < len(current_level) else None
                childs = [left_child, right_child] if right_child else [left_child]
                parent = TreeNode(index, childs)
                child_indices = [child.index for child in childs]
                print(str(parent.index) + " " + str(child_indices))
                next_level.append(parent)
                index += 1
                i += 2

            for i in range(1, len(current_level) - 1):
                current_level[i].left = current_level[i - 1]
                current_level[i].right = current_level[i + 1]

            current_level = next_level
        return leaves

    def get_element(self, index):
        return self.leaves[index]

class TreeNode:
    def __init__(self, index, childs):
        self.index = index
        self.childs = childs
        self.parent = None
        self.left = None
        self.right = None

        for child in self.childs:
            child.parent = self

    def is_rightest_child(self):
        return self.parent and self.parent.childs[-1] == self

    def is_leftest_child(self):
        return self.parent and self.parent.childs[0] == self

    def get_node(self, index):
        if self.index == index:
            return self
        if self.childs.empty():
            return None
        for child in self.childs:
            result = child.get_node(index)
            if result:
                return result

def calculate_aggregated_surroundings(index, tree, elements_per_layer=1):
    surroundings = []
    element = tree.get_element(index)
    if not element:
        return surroundings

    add_left(element, surroundings, elements_per_layer)
    add_right(element, surroundings, elements_per_layer)
    return [node.index for node in surroundings]


def add_left(element, surroundings, elements_per_layer):
    if element.left is None:
        return

    left_element = element.left
    for _ in range(elements_per_layer - 1):
        surroundings.insert(0, left_element)
        if left_element.left is None:
            return
        left_element = left_element.left

    while not left_element.is_leftest_child():
        surroundings.insert(0, left_element)
        left_element = left_element.left
    surroundings.insert(0, left_element)
    add_left(left_element.parent, surroundings, elements_per_layer)


def add_right(element, surroundings, elements_per_layer):
    if element.right is None:
        return

    right_element = element.right
    for _ in range(elements_per_layer - 1):
        surroundings.append(right_element)
        if right_element.right is None:
            return
        right_element = right_element.right

    while not right_element.is_rightest_child():
        surroundings.append(right_element)
        right_element = right_element.right
    surroundings.append(right_element)

    add_right(right_element.parent, surroundings, elements_per_layer)


# Testfunktion

# Example usage


tree = BinaryTree(23)  # Ein Baum mit 29 Blättern (nicht zwingend Zweierpotenz)
result = calculate_aggregated_surroundings(7, tree)  # Test mit Knoten 7
print(result)
# Test ausführen

[35, 25, 6, 8, 9, 28, 38, 43]
