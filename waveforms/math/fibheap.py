class FibNode():
    __slots__ = ('key', 'degree', 'left', 'right', 'child', 'parent', 'marked')

    def __init__(self,
                 key=None,
                 degree=0,
                 left=None,
                 right=None,
                 child=None,
                 parent=None,
                 marked=False):
        self.key, self.degree, self.left, self.right, self.child, self.parent, self.marked = (
            key, degree, left, right, child, parent, marked)

    def add(self, node):
        self.left.right = node
        node.left = self.left
        self.left = node
        node.right = self
        node.parent = self.parent

    def cat(self, node):
        self.right.left, node.right.left = node, self
        self.right, node.right = node.right, self.right

    def remove(self):
        self.left.right, self.right.left = self.right, self.left
        self.left, self.right = self, self

    def remove_child(self, node):
        self.degree -= 1
        self.child = node.right
        node.remove()
        if self.degree == 0:
            self.child = None
        node.parent = None

    def link(self, node):
        node.remove()
        self.degree += 1
        if self.child is None:
            self.child = node
            node.left, node.right = node, node
        else:
            self.child.add(node)
            self.child = node
        node.parent = self
        node.marked = False

    def view(self):
        print(
            f'\t{self.parent}\t\n{self.left}\t{self.key}\t{self.right}\n\t{self.child}({self.degree})'
        )

    def __repr__(self):
        return f'{self.key}'  #' at {hex(id(self))}'


class FibHeap():
    """
    Fibonacci Heap

    Attributes:
        key_number: number of keys in the heap
        min_node: the node with the minimum key

    Methods:
        insert(node): insert a node into the heap
        extract_min(): extract the node with the minimum key
        consolidate(): consolidate the heap
    """

    def __init__(self, key_number=0, min_node=None):
        self.key_number, self.min_node = key_number, min_node

    def insert(self, node):
        if self.min_node is None:
            self.min_node = node
            node.left, node.right = node, node
        else:
            self.min_node.add(node)
            if node.key < self.min_node.key:
                self.min_node = node
        self.key_number += 1

    def extract_min(self):
        ret = self.min_node
        if ret is not None:
            if ret.degree:
                x = ret.child
                while ret.degree:
                    x.parent = None
                    x = x.right
                    ret.degree -= 1
                ret.cat(ret.child)
                ret.child = None
            right = ret.right
            ret.remove()
            self.key_number -= 1
            if ret == right:
                self.min_node = None
            else:
                self.min_node = right
                self.consolidate()
        return ret

    def consolidate(self):
        if self.min_node is None:
            return
        from math import log2
        max_degree = round(log2(self.key_number) + 0.5) + 1
        cons = [None] * max_degree

        root_list = [self.min_node]
        right = self.min_node.right
        while right != self.min_node:
            root_list.append(right)
            right = right.right

        for i in range(len(root_list)):
            x = root_list[i]
            while cons[x.degree] is not None:
                y = cons[x.degree]
                if y.key < x.key:
                    x, y = y, x
                x.link(y)
                cons[y.degree] = None
            cons[x.degree] = x

        self.min_node = None
        for i in range(max_degree):
            if cons[i] is not None:
                if self.min_node is None:
                    self.min_node = cons[i]
                else:
                    if cons[i].key < self.min_node.key:
                        self.min_node = cons[i]

    def cut(self, node, parent):
        parent.remove_child(node)
        self.insert(node)
        node.marked = False

    def cascade(self, node):
        parent = node.parent
        if parent is not None:
            if node.marked:
                self.cut(node, parent)
                self.cascade(parent)
            else:
                node.marked = True

    def decrease(self, node, key):
        node.key = key
        parent = node.parent
        if parent is not None and node.key < parent.key:
            self.cut(node, parent)
            self.cascade(parent)
        if node.key < self.min_node.key:
            self.min_node = node

    def increase(self, node, key):
        while node.child is not None:
            self.cut(node.child, node)
        node.degree = 0
        node.key = key
        parent = node.parent
        if parent is not None:
            self.cut(node, parent)
            self.cascade(parent)
        elif self.min_node == node:
            right = node.right
            while right != node:
                if right.key < self.min_node.key:
                    self.min_node = right
                right = right.right

    def change(self, node, key):
        if key < node.key:
            self.decrease(node, key)
        elif key > node.key:
            self.increase(node, key)

    def remove(self, node):
        self.decrease(node, self.min_node.key)
        self.min_node = node
        self.extract_min()


def heap_union(a, b):
    """
    Union two Fibonacci Heaps
    """
    if a is None or a.min_node is None:
        return b
    if b is None or b.min_node is None:
        return a
    if b.max_degree > a.max_degree:
        a, b = b, a
    a.min_node.cat(b.min_node)
    if b.min_node.key < a.min_node.key:
        a.min_node = b.min_node
    a.key_number += b.key_number
    return a


def heap_view(nodes):
    from treelib import Node, Tree

    ret = Tree()
    ret.add_node(Node(identifier='root'))
    while True:
        flag = True
        for node in nodes:
            if ret.contains(node.__repr__()):
                continue
            parent = node.parent.__repr__(
            ) if node.parent is not None else 'root'
            if ret.contains(parent):
                ret.add_node(Node(identifier=node.__repr__()), parent=parent)
            else:
                flag = False
        if flag:
            break
    return ret.show()